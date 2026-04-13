"""
Realtime simultaneous interpretation demo.

Run:
    python realtime_web_demo.py --ref-audio test_ref.wav --ref-text "参考音频文本"

Open:
    http://127.0.0.1:7870/realtime
"""

from __future__ import annotations

import argparse
import base64
import cgi
import collections
import difflib
import hashlib
import io
import json
import mimetypes
import multiprocessing as mp
import os
import queue
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

from demo_pre_mlx_history import SimultaneousTranslator, split_english_for_tts, split_for_simul


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
RUNS_DIR = ROOT / "web_runs"
RUNS_DIR.mkdir(exist_ok=True)

TRANSLATOR = None
TRANSLATOR_LOCK = threading.Lock()

SESSIONS: Dict[str, "RealtimeSession"] = {}
SESSIONS_LOCK = threading.Lock()

LOG_LOCK = threading.Lock()
LOG_BUFFER = collections.deque(maxlen=500)
LOG_SEQ = 0
SENTENCE_END_CHARS = "，。！？；,.!?;:"
MLX_INFERENCE_LOCK = threading.Lock()


def _tts_process_main(model_id: str, streaming_interval_s: float, request_queue, response_queue):
    from demo_pre_mlx_history import TTSModule

    tts = TTSModule(model_path=model_id)
    tts.streaming_interval_s = streaming_interval_s
    active_ref_audio_path = None
    active_ref_text = None
    response_queue.put({"type": "ready"})

    while True:
        request = request_queue.get()
        if request is None:
            break

        request_id = request.get("request_id")
        try:
            ref_audio_path = request["ref_audio_path"]
            ref_text = request.get("ref_text") or ""
            if ref_audio_path != active_ref_audio_path or ref_text != active_ref_text:
                tts.create_voice_reference(ref_audio_path, ref_text)
                active_ref_audio_path = ref_audio_path
                active_ref_text = ref_text

            for waveform, sample_rate, is_final_chunk in tts.synthesize_stream(
                request["english_text"],
                ref_audio_path,
                ref_text,
            ):
                response_queue.put(
                    {
                        "request_id": request_id,
                        "type": "chunk",
                        "audio_pcm": np.asarray(waveform, dtype=np.float32).tobytes(),
                        "sample_rate": sample_rate,
                        "is_final_chunk": bool(is_final_chunk),
                    }
                )
            response_queue.put({"request_id": request_id, "type": "done"})
        except Exception as exc:
            response_queue.put(
                {
                    "request_id": request_id,
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


class IsolatedTTSWorker:
    def __init__(
        self,
        model_id: str,
        streaming_interval_s: float,
        first_chunk_timeout_s: float = 6.0,
        idle_timeout_s: float = 4.0,
        total_timeout_s: float = 20.0,
    ):
        self.model_id = model_id
        self.streaming_interval_s = streaming_interval_s
        self.first_chunk_timeout_s = first_chunk_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.total_timeout_s = total_timeout_s
        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._call_lock = threading.Lock()

    def _start(self):
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_tts_process_main,
            args=(self.model_id, self.streaming_interval_s, self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()
        append_log(f"[rt-tts] Started isolated TTS worker pid={self._process.pid}.")
        self._wait_until_ready()

    def _wait_until_ready(self):
        start = time.time()
        while True:
            if self._process is None or not self._process.is_alive():
                raise RuntimeError("isolated TTS worker exited during startup")
            try:
                response = self._response_queue.get(timeout=0.2)
            except queue.Empty:
                if (time.time() - start) > 20.0:
                    self._stop()
                    raise TimeoutError("isolated TTS worker startup timeout")
                continue
            if response.get("type") == "ready":
                return

    def _stop(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None
        self._request_queue = None
        self._response_queue = None

    def close(self):
        self._stop()

    def _restart(self, reason: str):
        append_log(f"[rt-tts] Restarting isolated TTS worker: {reason}")
        self._stop()
        self._start()

    def _ensure_started(self):
        if self._process is None or not self._process.is_alive():
            self._start()

    def synthesize_stream(self, english_text: str, ref_audio_path: str, ref_text: str):
        with self._call_lock:
            self._ensure_started()
            request_id = uuid.uuid4().hex
            self._request_queue.put(
                {
                    "request_id": request_id,
                    "english_text": english_text,
                    "ref_audio_path": ref_audio_path,
                    "ref_text": ref_text,
                }
            )

            start_time = time.time()
            last_message_time = start_time
            got_chunk = False

            while True:
                timeout_s = self.idle_timeout_s if got_chunk else self.first_chunk_timeout_s
                try:
                    response = self._response_queue.get(timeout=0.2)
                except queue.Empty:
                    now = time.time()
                    if self._process is None or not self._process.is_alive():
                        self._restart("worker exited unexpectedly")
                        raise RuntimeError("isolated TTS worker exited")
                    if (now - start_time) > self.total_timeout_s:
                        self._restart(f"total timeout on text={english_text!r}")
                        raise TimeoutError("TTS total timeout")
                    if (now - last_message_time) > timeout_s:
                        self._restart(f"idle timeout on text={english_text!r}")
                        raise TimeoutError("TTS idle timeout")
                    continue

                if response.get("request_id") != request_id:
                    continue

                last_message_time = time.time()
                response_type = response.get("type")
                if response_type == "chunk":
                    got_chunk = True
                    yield (
                        np.frombuffer(response["audio_pcm"], dtype=np.float32).copy(),
                        int(response["sample_rate"]),
                        bool(response["is_final_chunk"]),
                    )
                    continue
                if response_type == "done":
                    return
                if response_type == "error":
                    self._restart(f"worker error on text={english_text!r}")
                    raise RuntimeError(response.get("message") or "isolated TTS worker error")


def append_log(message: str):
    global LOG_SEQ
    text = (message or "").replace("\x1b[A", "").rstrip()
    if not text:
        return
    with LOG_LOCK:
        for line in text.splitlines():
            if not line.strip():
                continue
            LOG_SEQ += 1
            LOG_BUFFER.append({"seq": LOG_SEQ, "line": line, "ts": time.time()})


class TeeStream:
    def __init__(self, original):
        self.original = original

    def write(self, data):
        if data:
            append_log(data)
            self.original.write(data)
        return len(data or "")

    def flush(self):
        self.original.flush()


def install_log_capture():
    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(sys.stdout)
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(sys.stderr)


def send_static_file(handler: BaseHTTPRequestHandler, base_dir: Path, relative_path: str):
    target = (base_dir / relative_path).resolve()
    if base_dir not in target.parents and target != base_dir:
        handler.send_error(HTTPStatus.FORBIDDEN)
        return
    if not target.exists() or not target.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    body = target.read_bytes()
    content_type, _ = mimetypes.guess_type(str(target))
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_page() -> bytes:
    return (STATIC_DIR / "realtime.html").read_bytes()


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_uploaded_file(field_item, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="simul-rt-")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(field_item.file.read())
    return path


def infer_suffix(filename: str, fallback: str = ".wav") -> str:
    if not filename:
        return fallback
    suffix = Path(filename).suffix.lower()
    return suffix if suffix else fallback


def convert_audio_with_ffmpeg(input_path: str, sample_rate: int, channels: int = 1) -> str:
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="simul-rt-norm-")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {result.stderr.strip()}")
    return output_path


def get_translator(args) -> SimultaneousTranslator:
    global TRANSLATOR
    with TRANSLATOR_LOCK:
        if TRANSLATOR is None:
            TRANSLATOR = SimultaneousTranslator(
                asr_model_size=args.asr_model,
                mt_backend=args.mt_backend,
                mt_model=args.mt_model,
                mt_base_url=args.mt_base_url,
            )
        return TRANSLATOR


def encode_pcm_base64(waveform: np.ndarray) -> str:
    pcm = np.asarray(waveform, dtype=np.float32, order="C")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int):
    sf.write(str(path), waveform, sample_rate)


def choose_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((host, preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def normalize_translation_text(text: str) -> str:
    text = re.sub(r"<think\b[^>]*>.*?</think\b[^>]*>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"(?i)^okay[,.\s]+", "", text).strip()
    text = re.sub(r"(?i)^['’]ll\b", "I will", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_reference_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", "", text).lower()
    placeholder_values = {
        "参考音频文本",
        "参考文本",
        "reftext",
        "referencetext",
        "sampletext",
    }
    if normalized in placeholder_values:
        return ""
    return text


def find_overlap_chars(left: str, right: str, min_chars: int = 2) -> int:
    max_size = min(len(left), len(right))
    for size in range(max_size, min_chars - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


class StableChunkCommitter:
    """Merge overlapping sliding-window ASR results and hold back an unstable tail."""

    def __init__(
        self,
        flush_chars: int = 12,
        tail_guard_chars: int = 6,
        merge_context_chars: int = 28,
        min_overlap_chars: int = 1,
    ):
        self.flush_chars = flush_chars
        self.tail_guard_chars = tail_guard_chars
        self.merge_context_chars = merge_context_chars
        self.min_overlap_chars = min_overlap_chars
        self.recent_text = ""
        self.pending = ""

    def _normalize(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", "", text)
        return text

    def _working_text(self) -> str:
        return self.recent_text + self.pending

    def _merge_fragment(self, fragment: str) -> str:
        base = self._working_text()
        if not base:
            return fragment

        exact_overlap = find_overlap_chars(base, fragment, min_chars=self.min_overlap_chars)
        if exact_overlap > 0:
            return base + fragment[exact_overlap:]

        search_tail = base[-max(self.merge_context_chars, len(fragment)) :]
        if fragment in search_tail:
            return base

        tail = base[-self.merge_context_chars :]
        match = difflib.SequenceMatcher(None, tail, fragment).find_longest_match(0, len(tail), 0, len(fragment))
        if match.size >= max(3, self.min_overlap_chars) and match.a + match.size == len(tail):
            return base + fragment[match.b + match.size :]

        return base + fragment

    def _next_commit_index(self, force: bool) -> int:
        if not self.pending:
            return 0
        if force:
            return len(self.pending)

        last_break = -1
        for ch in SENTENCE_END_CHARS:
            last_break = max(last_break, self.pending.rfind(ch))
        if last_break >= 0:
            return last_break + 1

        if len(self.pending) >= self.flush_chars:
            return max(0, len(self.pending) - self.tail_guard_chars)

        return 0

    def _flush_pending(self, force: bool) -> List[str]:
        commit_index = self._next_commit_index(force)
        if commit_index <= 0:
            return []
        stable_text = self.pending[:commit_index].strip()
        self.pending = self.pending[commit_index:]
        if not stable_text:
            return []
        self.recent_text = (self.recent_text + stable_text)[-self.merge_context_chars :]
        return split_for_simul(stable_text, max_chars=18)

    def ingest(self, partial_text: str) -> List[str]:
        partial_text = self._normalize(partial_text)
        if not partial_text:
            return []
        merged = self._merge_fragment(partial_text)
        if len(merged) < len(self.recent_text):
            return []
        self.pending = merged[len(self.recent_text) :]
        return self._flush_pending(force=False)

    def finalize(self) -> List[str]:
        return self._flush_pending(force=True)


class RealtimeSentenceAccumulator:
    """Accumulate committed Chinese fragments and only emit sentence-like units."""

    FILLER_SET = {
        "嗯",
        "嗯。",
        "啊",
        "啊。",
        "哦",
        "哦。",
        "呃",
        "呃。",
        "是",
        "是。",
        "对",
        "对。",
        "好",
        "好。",
        "你好",
        "你好。",
    }

    def __init__(self, min_chars: int = 8, force_chars: int = 18):
        self.min_chars = min_chars
        self.force_chars = force_chars
        self.parts: List[str] = []
        self.first_asr_ms: float = 0.0

    def _combined(self) -> str:
        return "".join(part for part in self.parts if part).strip()

    def _reset(self):
        self.parts = []
        self.first_asr_ms = 0.0

    def _should_flush(self, text: str, force: bool) -> bool:
        if not text:
            return False
        if force:
            return True
        if text in self.FILLER_SET:
            return False
        if len(text) >= self.force_chars:
            return True
        if re.search(r"[。！？!?；;]$", text) and len(text) >= self.min_chars:
            return True
        if re.search(r"[，,]", text) and len(text) >= self.min_chars + 2:
            return True
        return False

    def feed(self, text: str, asr_ms: float, force: bool = False) -> List[tuple[str, float]]:
        text = (text or "").strip()
        if text:
            if not self.parts:
                self.first_asr_ms = asr_ms
            self.parts.append(text)
        combined = self._combined()
        if not self._should_flush(combined, force):
            return []
        emitted = split_for_simul(combined, max_chars=18) or [combined]
        emitted_asr_ms = self.first_asr_ms
        self._reset()
        return [(item, emitted_asr_ms if idx == 0 else 0.0) for idx, item in enumerate(emitted)]

    def finalize(self) -> List[tuple[str, float]]:
        return self.feed("", self.first_asr_ms, force=True)


@dataclass
class AudioChunkTask:
    seq: int
    start_ms: int
    end_ms: int
    samples: np.ndarray
    sample_rate: int


class RealtimeSession:
    _INPUT_DONE = object()
    _TEXT_DONE = object()

    def __init__(self, session_id: str, translator: SimultaneousTranslator, ref_audio_path: str, ref_text: str):
        self.session_id = session_id
        self.translator = translator
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.created_at = time.time()
        self.input_queue: queue.Queue = queue.Queue()
        self.text_queue: queue.Queue = queue.Queue()
        self.events: Deque[Dict] = collections.deque(maxlen=2000)
        self.event_seq = 0
        self.last_chunk_seq = -1
        self.segment_index = 0
        self.first_audio_ms: Optional[float] = None
        self.source_cursor_ms = 0
        self.playout_cursor_ms = 0.0
        self.source_audio: List[np.ndarray] = []
        self.source_sample_rate: Optional[int] = None
        self.wave_chunks: List[np.ndarray] = []
        self.sample_rate: Optional[int] = None
        self.done = False
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.committer = StableChunkCommitter()
        self.sentence_accumulator = RealtimeSentenceAccumulator()
        self.tts_worker = IsolatedTTSWorker(
            model_id=self.translator.tts.model_id,
            streaming_interval_s=self.translator.tts.streaming_interval_s,
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def emit_event(self, payload: Dict):
        with self.event_lock:
            self.event_seq += 1
            payload["seq"] = self.event_seq
            self.events.append(payload)

    def list_events(self, since: int) -> Dict:
        items = [item for item in self.events if item["seq"] > since]
        return {
            "events": items,
            "latest_seq": self.event_seq,
            "done": self.done,
            "error": self.error,
        }

    def enqueue_chunk(self, task: AudioChunkTask):
        self.input_queue.put(task)

    def finalize(self):
        self.input_queue.put(self._INPUT_DONE)

    def _record_error(self, label: str, exc: BaseException):
        if self.error is None:
            self.error = str(exc)
            append_log(f"{label}:\n" + traceback.format_exc())
            self.emit_event({"type": "error", "message": self.error})
        self.stop_event.set()

    def _transcribe_chunk(self, samples: np.ndarray, sample_rate: int) -> tuple[str, float]:
        if sample_rate != self.translator.asr.sample_rate:
            raise ValueError(f"unexpected sample rate: {sample_rate}")
        t0 = time.time()
        with MLX_INFERENCE_LOCK:
            result = self.translator.asr.model.generate(
                samples.astype(np.float32, copy=False),
                language="zh",
                stream=False,
                verbose=False,
            )
        return self.translator.asr._output_text(result), (time.time() - t0) * 1000

    def _choose_tts_units(self, english_text: str) -> List[str]:
        lag_ms = max(0.0, self.source_cursor_ms - self.playout_cursor_ms)
        if lag_ms > 4000:
            return split_english_for_tts(english_text, max_words=4, max_chars=22)[:1]
        if lag_ms > 2800:
            return split_english_for_tts(english_text, max_words=5, max_chars=28)
        return split_english_for_tts(english_text, max_words=8, max_chars=48)

    def _process_committed_text(self, chinese_text: str, asr_ms: float):
        mt_t0 = time.time()
        english_text = normalize_translation_text(self.translator.mt.translate(chinese_text))
        mt_ms = (time.time() - mt_t0) * 1000
        if not english_text:
            return

        for index, english_unit in enumerate(self._choose_tts_units(english_text)):
            self.segment_index += 1
            tts_t0 = time.time()
            tts_first_chunk_ms: Optional[float] = None
            segment_wave_chunks: List[np.ndarray] = []
            sample_rate: Optional[int] = None

            for chunk_index, (waveform_chunk, sample_rate, is_final_chunk) in enumerate(
                self.tts_worker.synthesize_stream(english_unit, self.ref_audio_path, self.ref_text),
                start=1,
            ):
                now = time.time()
                if tts_first_chunk_ms is None:
                    tts_first_chunk_ms = (now - tts_t0) * 1000
                if self.first_audio_ms is None:
                    self.first_audio_ms = (now - self.created_at) * 1000
                segment_wave_chunks.append(waveform_chunk)
                self.emit_event(
                    {
                        "type": "audio_chunk",
                        "segment_index": self.segment_index,
                        "chunk_index": chunk_index,
                        "sample_rate": sample_rate,
                        "is_final_chunk": bool(is_final_chunk),
                        "first_audio_ms": self.first_audio_ms,
                        "tts_first_chunk_ms": tts_first_chunk_ms,
                        "audio_pcm_base64": encode_pcm_base64(waveform_chunk),
                    }
                )

            if not segment_wave_chunks or sample_rate is None:
                continue

            waveform = np.concatenate(segment_wave_chunks).astype(np.float32, copy=False)
            self.wave_chunks.append(waveform)
            self.sample_rate = sample_rate
            tts_ms = (time.time() - tts_t0) * 1000
            self.playout_cursor_ms += len(waveform) / sample_rate * 1000
            lag_ms = max(0.0, self.source_cursor_ms - self.playout_cursor_ms)

            self.emit_event(
                {
                    "type": "segment",
                    "segment": {
                        "index": self.segment_index,
                        "chinese": chinese_text if index == 0 else "",
                        "english": english_unit,
                        "asr_ms": asr_ms if index == 0 else 0.0,
                        "mt_ms": mt_ms if index == 0 else 0.0,
                        "tts_ms": tts_ms,
                        "tts_first_chunk_ms": tts_first_chunk_ms,
                        "lag_ms": lag_ms,
                        "source_cursor_ms": self.source_cursor_ms,
                        "playout_cursor_ms": self.playout_cursor_ms,
                    },
                }
            )

            print(
                f"  [rt] source={self.source_cursor_ms:.0f}ms | lag={lag_ms:.0f}ms | "
                f"ASR={asr_ms:.0f}ms | MT={mt_ms if index == 0 else 0:.0f}ms | "
                f"TTS-first={tts_first_chunk_ms or 0:.0f}ms | TTS-total={tts_ms:.0f}ms | "
                f"CN={chinese_text if index == 0 else ''} | EN={english_unit}"
            )

    def _append_source_audio(self, task: AudioChunkTask):
        if self.source_sample_rate is None:
            self.source_sample_rate = task.sample_rate
        current_samples = int(sum(len(chunk) for chunk in self.source_audio))
        start_sample = int(round(task.start_ms * task.sample_rate / 1000))
        if current_samples < start_sample:
            self.source_audio.append(np.zeros(start_sample - current_samples, dtype=np.float32))
            current_samples = start_sample
        overlap = max(0, current_samples - start_sample)
        if overlap < len(task.samples):
            self.source_audio.append(task.samples[overlap:].astype(np.float32, copy=False))

    def _run_finalize_fallback(self):
        if self.segment_index > 0 or not self.source_audio or self.source_sample_rate is None:
            return
        merged = np.concatenate(self.source_audio).astype(np.float32, copy=False)
        if merged.size == 0:
            return
        fallback_t0 = time.time()
        full_text, _ = self._transcribe_chunk(merged, self.source_sample_rate)
        full_text = (full_text or "").strip()
        print(f"  [rt-fallback] full-asr={((time.time() - fallback_t0) * 1000):.0f}ms | text={full_text}")
        if not full_text:
            return
        for chinese_text in split_for_simul(full_text, max_chars=18):
            self._process_committed_text(chinese_text, 0.0)

    def _run_asr_commit(self):
        try:
            while True:
                task = self.input_queue.get()
                if task is self._INPUT_DONE:
                    break
                if self.stop_event.is_set():
                    continue

                self.source_cursor_ms = max(self.source_cursor_ms, task.end_ms)
                self._append_source_audio(task)
                partial_text, asr_ms = self._transcribe_chunk(task.samples, task.sample_rate)
                print(
                    f"  [rt-asr] seq={task.seq} | source={task.end_ms}ms | "
                    f"asr={asr_ms:.0f}ms | partial={partial_text}"
                )
                self.emit_event(
                    {
                        "type": "partial",
                        "partial": partial_text,
                        "asr_ms": asr_ms,
                        "source_cursor_ms": self.source_cursor_ms,
                    }
                )
                for chinese_text in self.committer.ingest(partial_text):
                    for sentence_text, sentence_asr_ms in self.sentence_accumulator.feed(chinese_text, asr_ms):
                        self.text_queue.put((sentence_text, sentence_asr_ms))

            for chinese_text in self.committer.finalize():
                for sentence_text, sentence_asr_ms in self.sentence_accumulator.feed(chinese_text, 0.0):
                    self.text_queue.put((sentence_text, sentence_asr_ms))
            for sentence_text, sentence_asr_ms in self.sentence_accumulator.finalize():
                self.text_queue.put((sentence_text, sentence_asr_ms))
        except Exception as exc:
            self._record_error("Realtime ASR worker exception", exc)
        finally:
            self.text_queue.put(self._TEXT_DONE)

    def _run_tts(self):
        try:
            while True:
                item = self.text_queue.get()
                if item is self._TEXT_DONE:
                    break
                if self.stop_event.is_set():
                    continue
                while self.input_queue.qsize() > 2 and not self.stop_event.is_set():
                    time.sleep(0.05)
                chinese_text, asr_ms = item
                self._process_committed_text(chinese_text, asr_ms)
        except Exception as exc:
            self._record_error("Realtime TTS worker exception", exc)

    def _run(self):
        asr_thread = threading.Thread(target=self._run_asr_commit, daemon=True)
        tts_thread = threading.Thread(target=self._run_tts, daemon=True)
        asr_thread.start()
        tts_thread.start()
        try:
            asr_thread.join()
            tts_thread.join()
            if self.error:
                return
            self._run_finalize_fallback()

            audio_url = ""
            if self.wave_chunks and self.sample_rate:
                merged = np.concatenate(self.wave_chunks).astype(np.float32, copy=False)
                output_path = RUNS_DIR / f"{self.session_id}.wav"
                write_wav(output_path, merged, self.sample_rate)
                audio_url = f"/{output_path.relative_to(ROOT)}"

            self.emit_event(
                {
                    "type": "done",
                    "audio_url": audio_url,
                    "timings": {
                        "first_audio_ms": self.first_audio_ms,
                        "segment_count": self.segment_index,
                        "source_cursor_ms": self.source_cursor_ms,
                        "playout_cursor_ms": self.playout_cursor_ms,
                        "lag_ms": max(0.0, self.source_cursor_ms - self.playout_cursor_ms),
                    },
                }
            )
        except Exception as exc:
            self._record_error("Realtime session exception", exc)
        finally:
            self.tts_worker.close()
            self.done = True


class DemoServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, args):
        super().__init__(server_address, RequestHandlerClass)
        self.args = args


class RealtimeHandler(BaseHTTPRequestHandler):
    server_version = "RealtimeSimulDemo/0.1"

    def log_message(self, fmt, *args):
        message = fmt % args
        if "/api/realtime/events" in message or "/api/logs" in message:
            return
        print(f"[realtime-web] {self.address_string()} - {message}")

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/realtime"}:
            body = html_page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/realtime/config":
            payload = {
                "default_ref_text": self.server.args.ref_text or "",
                "default_ref_audio": bool(self.server.args.ref_audio),
                "window_ms": self.server.args.window_ms,
                "hop_ms": self.server.args.hop_ms,
                "sample_rate": 16000,
            }
            self._send_json(payload)
            return

        if parsed.path == "/api/realtime/events":
            params = parse_qs(parsed.query)
            session_id = (params.get("session_id") or [""])[0]
            since = int((params.get("since") or ["0"])[0])
            session = self._get_session(session_id)
            if not session:
                self._send_json({"error": "session not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(session.list_events(since))
            return

        if parsed.path == "/api/logs":
            since = int((parse_qs(parsed.query).get("since") or ["0"])[0])
            with LOG_LOCK:
                items = [item for item in LOG_BUFFER if item["seq"] > since]
                latest_seq = LOG_SEQ
            self._send_json({"logs": items, "latest_seq": latest_seq})
            return

        if parsed.path.startswith("/static/"):
            send_static_file(self, STATIC_DIR, parsed.path[len("/static/") :])
            return

        if parsed.path.startswith("/runs/") or parsed.path.startswith("/web_runs/"):
            relative = parsed.path.lstrip("/")
            target = (ROOT / relative).resolve()
            if ROOT not in target.parents and target != ROOT:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = target.read_bytes()
            content_type = "audio/wav" if target.suffix == ".wav" else "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/realtime/session":
            self._handle_create_session()
            return
        if parsed.path == "/api/realtime/chunk":
            self._handle_chunk()
            return
        if parsed.path == "/api/realtime/finalize":
            self._handle_finalize()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_json(self, payload: Dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_session(self, session_id: str) -> Optional[RealtimeSession]:
        with SESSIONS_LOCK:
            return SESSIONS.get(session_id)

    def _handle_create_session(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )

        ref_audio_path = self.server.args.ref_audio
        ref_text = normalize_reference_text(form.getfirst("ref_text", self.server.args.ref_text or ""))
        window_ms = int(form.getfirst("window_ms", str(self.server.args.window_ms)))
        hop_ms = int(form.getfirst("hop_ms", str(self.server.args.hop_ms)))
        if window_ms < 600 or hop_ms < 300 or hop_ms >= window_ms:
            self._send_json({"error": "invalid window/hop settings"}, status=HTTPStatus.BAD_REQUEST)
            return

        uploaded_ref = form["ref_audio"] if "ref_audio" in form and getattr(form["ref_audio"], "filename", "") else None
        temp_files: List[str] = []

        try:
            if uploaded_ref is not None:
                ref_audio_path = save_uploaded_file(uploaded_ref, suffix=infer_suffix(uploaded_ref.filename, ".wav"))
                temp_files.append(ref_audio_path)
                normalized = convert_audio_with_ffmpeg(ref_audio_path, sample_rate=24000, channels=1)
                temp_files.append(normalized)
                ref_audio_path = normalized
            elif ref_audio_path:
                normalized = convert_audio_with_ffmpeg(ref_audio_path, sample_rate=24000, channels=1)
                temp_files.append(normalized)
                ref_audio_path = normalized

            if not ref_audio_path:
                self._send_json({"error": "reference audio is required"}, status=HTTPStatus.BAD_REQUEST)
                return

            translator = get_translator(self.server.args)
            translator.tts.create_voice_reference(ref_audio_path, "")

            session_id = uuid.uuid4().hex
            session = RealtimeSession(session_id, translator, ref_audio_path, "")
            with SESSIONS_LOCK:
                SESSIONS[session_id] = session

            self._send_json(
                {
                    "session_id": session_id,
                    "window_ms": window_ms,
                    "hop_ms": hop_ms,
                    "sample_rate": 16000,
                }
            )
        except Exception as exc:
            append_log("Create session exception:\n" + traceback.format_exc())
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_chunk(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        session_id = (params.get("session_id") or [""])[0]
        seq = int((params.get("seq") or ["0"])[0])
        start_ms = int((params.get("start_ms") or ["0"])[0])
        end_ms = int((params.get("end_ms") or ["0"])[0])

        session = self._get_session(session_id)
        if not session:
            self._send_json({"error": "session not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        if not body:
            self._send_json({"error": "empty chunk"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            samples, sample_rate = sf.read(io.BytesIO(body), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            samples = np.asarray(samples, dtype=np.float32)
            with session.lock:
                if seq <= session.last_chunk_seq:
                    self._send_json({"ok": True, "ignored": True})
                    return
                session.last_chunk_seq = seq
            session.enqueue_chunk(AudioChunkTask(seq, start_ms, end_ms, samples, sample_rate))
            self._send_json({"ok": True})
        except Exception as exc:
            append_log("Chunk ingest exception:\n" + traceback.format_exc())
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_finalize(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = {}
        if content_length:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        session_id = payload.get("session_id", "")
        session = self._get_session(session_id)
        if not session:
            self._send_json({"error": "session not found"}, status=HTTPStatus.NOT_FOUND)
            return
        session.finalize()
        self._send_json({"ok": True})


def build_parser():
    parser = argparse.ArgumentParser(description="Realtime simultaneous translation web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--window-ms", type=int, default=1000)
    parser.add_argument("--hop-ms", type=int, default=700)
    parser.add_argument("--ref-audio", help="Default reference audio")
    parser.add_argument("--ref-text", default="", help="Default reference transcript")
    parser.add_argument("--asr-model", default="mlx-community/Qwen3-ASR-0.6B-bf16")
    parser.add_argument("--mt-backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--mt-model")
    parser.add_argument("--mt-base-url")
    return parser


def main():
    install_log_capture()
    parser = build_parser()
    args = parser.parse_args()
    port = choose_port(args.host, args.port)
    server = DemoServer((args.host, port), RealtimeHandler, args)
    print("\n" + "=" * 56)
    print(f"Realtime demo running at http://{args.host}:{port}/realtime")
    print("=" * 56)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

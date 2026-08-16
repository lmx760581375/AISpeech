"""
Local web demo for simultaneous translation.

Run:
    python web_demo.py --ref-audio test_ref.wav --ref-text "参考音频文本"

Open:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import base64
import cgi
import collections
import hashlib
import json
import mimetypes
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from demo import SimultaneousTranslator, TTSModule


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
RUNS_DIR = ROOT / "web_runs"
RUNS_DIR.mkdir(exist_ok=True)

TRANSLATOR = None
TRANSLATOR_LOCK = threading.Lock()
VOICE_REF_LOCK = threading.Lock()
VOICE_REF_STATE = {"signature": None}
WARMUP_STATE = {
    "status": "idle",
    "message": "waiting",
    "started_at": None,
    "completed_at": None,
    "error": None,
}
LOG_LOCK = threading.Lock()
LOG_BUFFER = collections.deque(maxlen=400)
LOG_SEQ = 0
WEB_TTS_STREAMING_INTERVAL_S = 0.6
WEB_AUDIO_CHUNK_MIN_DURATION_S = 0.6


class ClientDisconnectedError(ConnectionError):
    """Raised when the browser closes the streaming response early."""


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


def html_page() -> bytes:
    return (STATIC_DIR / "index.html").read_bytes()


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


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int):
    import soundfile as sf

    sf.write(str(path), waveform, sample_rate)


def encode_pcm_base64(waveform: np.ndarray) -> str:
    pcm = np.asarray(waveform, dtype=np.float32, order="C")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def merge_waveforms(chunks: List[np.ndarray]) -> np.ndarray:
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=np.float32, order="C")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def save_uploaded_file(field_item, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="simul-web-")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(field_item.file.read())
    return path


def convert_audio_with_ffmpeg(input_path: str, sample_rate: int, channels: int = 1) -> str:
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="simul-norm-")
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


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_translator(args) -> SimultaneousTranslator:
    global TRANSLATOR
    with TRANSLATOR_LOCK:
        if TRANSLATOR is None:
            TRANSLATOR = SimultaneousTranslator(
                asr_model_size=args.asr_model,
                mt_backend=args.mt_backend,
                mt_model=args.mt_model,
                mt_base_url=args.mt_base_url,
                tts_model=args.tts_model,
            )
        if getattr(TRANSLATOR.tts, "streaming_interval_s", 0.0) < WEB_TTS_STREAMING_INTERVAL_S:
            TRANSLATOR.tts.streaming_interval_s = WEB_TTS_STREAMING_INTERVAL_S
        return TRANSLATOR


def iter_web_audio_events(
    events: Iterable[Dict],
    min_chunk_duration_s: float = WEB_AUDIO_CHUNK_MIN_DURATION_S,
):
    buffered_chunks: List[np.ndarray] = []
    buffered_samples = 0
    buffered_event: Optional[Dict] = None

    def flush_buffer():
        nonlocal buffered_chunks, buffered_samples, buffered_event
        if not buffered_chunks or buffered_event is None:
            return None
        waveform = merge_waveforms(buffered_chunks)
        merged_event = dict(buffered_event)
        merged_event["waveform"] = waveform
        merged_event["num_samples"] = int(len(waveform))
        buffered_chunks = []
        buffered_samples = 0
        buffered_event = None
        return merged_event

    for event in events:
        if event.get("type") != "audio_chunk":
            flushed = flush_buffer()
            if flushed is not None:
                yield flushed
            yield event
            continue

        waveform = np.asarray(event["waveform"], dtype=np.float32, order="C")
        if waveform.size == 0:
            continue

        buffered_chunks.append(waveform)
        buffered_samples += len(waveform)
        buffered_event = dict(event)
        sample_rate = max(1, int(event["sample_rate"]))
        buffered_duration_s = buffered_samples / sample_rate
        if buffered_duration_s >= min_chunk_duration_s or event.get("is_final_chunk"):
            flushed = flush_buffer()
            if flushed is not None:
                yield flushed

    flushed = flush_buffer()
    if flushed is not None:
        yield flushed


def ensure_voice_reference(translator: SimultaneousTranslator, ref_audio: str, ref_text: str):
    signature = f"{file_sha1(ref_audio)}::{ref_text}"
    with VOICE_REF_LOCK:
        if VOICE_REF_STATE["signature"] == signature:
            return
        translator.tts.create_voice_reference(ref_audio, ref_text)
        VOICE_REF_STATE["signature"] = signature


def run_startup_warmup(args):
    WARMUP_STATE["status"] = "warming"
    WARMUP_STATE["message"] = "loading_models"
    WARMUP_STATE["started_at"] = time.time()
    WARMUP_STATE["completed_at"] = None
    WARMUP_STATE["error"] = None
    try:
        translator = get_translator(args)
        if args.ref_audio:
            WARMUP_STATE["message"] = "precomputing_voice_clone"
            ensure_voice_reference(translator, args.ref_audio, args.ref_text or "")
        if args.eager_warmup:
            WARMUP_STATE["message"] = "warming_translation"
            translator.mt.translate("你好")
            if args.ref_audio and getattr(translator.tts, "device", "cpu") != "cpu":
                WARMUP_STATE["message"] = "warming_tts"
                waveform, sample_rate = translator.tts.synthesize("Hello.", args.ref_audio, args.ref_text or "")
                del waveform, sample_rate
        WARMUP_STATE["status"] = "ready"
        WARMUP_STATE["message"] = "ready"
        WARMUP_STATE["completed_at"] = time.time()
    except Exception as exc:
        append_log("Warmup exception:\n" + traceback.format_exc())
        WARMUP_STATE["status"] = "error"
        WARMUP_STATE["message"] = "warmup_failed"
        WARMUP_STATE["error"] = str(exc)
        WARMUP_STATE["completed_at"] = time.time()


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "SimulDemo/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = html_page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/config":
            payload = {
                "sample_audio_url": "/assets/test_zh.wav" if (ROOT / "test_zh.wav").exists() else "",
                "sample_ref_audio_url": "/assets/test_ref.wav" if (ROOT / "test_ref.wav").exists() else "",
                "default_ref_text": self.server.args.ref_text or "",
                "mt_backend": self.server.args.mt_backend,
                "voice_cards": TTSModule.get_reference_voice_cards(),
                "streaming": True,
                "warmup": WARMUP_STATE,
            }
            self._send_json(payload)
            return

        if parsed.path == "/api/warmup":
            self._send_json(WARMUP_STATE)
            return

        if parsed.path == "/api/logs":
            since = int((parse_qs(parsed.query).get("since") or ["0"])[0])
            with LOG_LOCK:
                items = [item for item in LOG_BUFFER if item["seq"] > since]
                latest_seq = LOG_SEQ
            self._send_json({"logs": items, "latest_seq": latest_seq})
            return

        if parsed.path.startswith("/assets/"):
            relative = parsed.path[len("/assets/") :]
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
        if parsed.path == "/api/translate":
            self._handle_translate()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt, *args):
        message = fmt % args
        if "/api/logs" in message:
            return
        print(f"[web] {self.address_string()} - {message}")

    def _send_json(self, payload: Dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_event(self, event: Dict):
        line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            self.wfile.write(line)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnectedError("stream client disconnected") from exc

    def _handle_translate(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )

        audio_item = form["audio"] if "audio" in form else None
        if audio_item is None or not getattr(audio_item, "file", None):
            self._send_json({"error": "audio is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        voice_mode = form.getfirst("voice_mode", "clone").strip().lower() or "clone"
        selected_voice_card = form.getfirst("voice_card", "").strip().lower()
        ref_audio_path = self.server.args.ref_audio
        ref_text = form.getfirst("ref_text", self.server.args.ref_text or "")
        uploaded_ref = form["ref_audio"] if "ref_audio" in form and getattr(form["ref_audio"], "filename", "") else None

        temp_files = []
        audio_path = save_uploaded_file(audio_item, suffix=self._infer_suffix(audio_item.filename, ".wav"))
        temp_files.append(audio_path)
        normalized_audio_path = convert_audio_with_ffmpeg(audio_path, sample_rate=16000, channels=1)
        temp_files.append(normalized_audio_path)

        if uploaded_ref is not None:
            ref_audio_path = save_uploaded_file(uploaded_ref, suffix=self._infer_suffix(uploaded_ref.filename, ".wav"))
            temp_files.append(ref_audio_path)
            normalized_ref_audio_path = convert_audio_with_ffmpeg(ref_audio_path, sample_rate=24000, channels=1)
            temp_files.append(normalized_ref_audio_path)
            ref_audio_path = normalized_ref_audio_path
        elif ref_audio_path:
            normalized_ref_audio_path = convert_audio_with_ffmpeg(ref_audio_path, sample_rate=24000, channels=1)
            temp_files.append(normalized_ref_audio_path)
            ref_audio_path = normalized_ref_audio_path

        if not ref_audio_path:
            self._send_json({"error": "reference audio is required"}, status=HTTPStatus.BAD_REQUEST)
            self._cleanup_temp_files(temp_files)
            return

        if voice_mode == "voice_card":
            if not selected_voice_card:
                self._send_json({"error": "voice_card is required when voice_mode=voice_card"}, status=HTTPStatus.BAD_REQUEST)
                self._cleanup_temp_files(temp_files)
                return
            try:
                TTSModule.resolve_reference_voice_card(selected_voice_card)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                self._cleanup_temp_files(temp_files)
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        translator = get_translator(self.server.args)
        try:
            ensure_voice_reference(translator, ref_audio_path, ref_text)
            if voice_mode == "voice_card":
                self._stream_event({"type": "status", "message": f"selected_voice_card:{selected_voice_card}"})
            total_start = time.time()
            wave_chunks = []
            sample_rate: Optional[int] = None
            segments = []
            first_audio_ms = None

            self._stream_event({"type": "status", "message": "normalizing_audio"})
            self._stream_event({"type": "status", "message": "models_ready"})
            event_stream = translator.iter_audio_streaming_events(
                normalized_audio_path,
                ref_audio_path,
                ref_text,
                voice_mode=voice_mode,
                voice_card=selected_voice_card or None,
            )
            for event in iter_web_audio_events(event_stream):
                if event["type"] == "audio_chunk":
                    if first_audio_ms is None:
                        first_audio_ms = event["first_audio_ms"]
                    self._stream_event(
                        {
                            "type": "audio_chunk",
                            "segment_index": event["segment_index"],
                            "chunk_index": event["chunk_index"],
                            "sample_rate": event["sample_rate"],
                            "num_samples": int(len(event["waveform"])),
                            "first_audio_ms": event["first_audio_ms"],
                            "tts_first_chunk_ms": event["tts_first_chunk_ms"],
                            "is_final_chunk": event["is_final_chunk"],
                            "audio_pcm_base64": encode_pcm_base64(event["waveform"]),
                        }
                    )
                    continue

                if event["type"] != "segment":
                    continue

                sample_rate = event["sample_rate"]
                waveform = event["waveform"]
                segment = event["segment"]
                if first_audio_ms is None:
                    first_audio_ms = event["first_audio_ms"]
                wave_chunks.append(waveform)
                segment_payload = segment.to_dict()
                segment_payload["index"] = event["segment_index"]
                segments.append(segment_payload)
                self._stream_event({"type": "segment", "segment": segment_payload})

            if not wave_chunks or sample_rate is None:
                self._stream_event({"type": "error", "message": "No speech detected"})
                return

            merged = np.concatenate(wave_chunks)
            run_id = uuid.uuid4().hex
            output_path = RUNS_DIR / f"{run_id}.wav"
            write_wav(output_path, merged, sample_rate)

            self._stream_event(
                {
                    "type": "done",
                    "audio_url": f"/{output_path.relative_to(ROOT)}",
                    "timings": {
                        "first_audio_ms": first_audio_ms,
                        "total_ms": (time.time() - total_start) * 1000,
                        "segment_count": len(segments),
                    },
                    "segments": segments,
                }
            )
        except ClientDisconnectedError:
            append_log("Streaming client disconnected; stopping response cleanly.")
        except Exception as exc:
            append_log("Translate exception:\n" + traceback.format_exc())
            try:
                self._stream_event({"type": "error", "message": str(exc)})
            except ClientDisconnectedError:
                append_log("Streaming client disconnected before error payload could be delivered.")
        finally:
            self._cleanup_temp_files(temp_files)

    @staticmethod
    def _infer_suffix(filename: str, fallback: str) -> str:
        if not filename:
            return fallback
        suffix = Path(filename).suffix.lower()
        return suffix or fallback

    @staticmethod
    def _cleanup_temp_files(paths):
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass


class DemoServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, args):
        super().__init__(server_address, handler_class)
        self.args = args


def build_parser():
    parser = argparse.ArgumentParser(description="Web demo for simultaneous translation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ref-audio", help="Default reference audio")
    parser.add_argument("--ref-text", default="", help="Default reference transcript")
    parser.add_argument("--asr-model", default="mlx-community/Qwen3-ASR-0.6B-bf16")
    parser.add_argument(
        "--tts-model",
        default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        help="mlx-audio TTS model (4-bit is the low-latency default)",
    )
    parser.add_argument("--mt-backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--mt-model")
    parser.add_argument("--mt-base-url")
    parser.add_argument("--no-startup-warmup", action="store_true", help="Skip background warmup on server start")
    parser.add_argument("--eager-warmup", action="store_true", help="Run a tiny MT/TTS inference during startup warmup")
    return parser


def choose_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            sock.bind((host, 0))
            return sock.getsockname()[1]


def main():
    install_log_capture()
    args = build_parser().parse_args()
    port = choose_port(args.host, args.port)
    server = DemoServer((args.host, port), DemoHandler, args)
    if not args.no_startup_warmup:
        thread = threading.Thread(target=run_startup_warmup, args=(args,), daemon=True)
        thread.start()
    print(f"Web demo running at http://{args.host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

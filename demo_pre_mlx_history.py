"""
Low-latency speech translation demo.

Goal:
    Chinese speech -> English speech with cloned voice.

This version is built for latency-sensitive experimentation:
    - ASR emits short segments instead of waiting for the whole sentence
    - MT backend can use `ollama` or an OpenAI-compatible vLLM server
    - live mode uses WebRTC VAD and short chunk flushing
    - voice clone prompt is precomputed once and reused

Examples:
    python demo.py test_zh.wav --ref-audio test_ref.wav --ref-text "参考音频文本"
    python demo.py test_zh.wav --streaming --mt-backend openai --mt-base-url http://127.0.0.1:8000/v1
    python demo.py --live --ref-audio test_ref.wav --ref-text "参考音频文本"
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error, request

import numpy as np


PUNCTUATION_RE = re.compile(r"(?<=[,，.。!！?？;；:：])")
ENGLISH_PUNCTUATION_RE = re.compile(r"(?<=[,.!?;:])\s+")
ENGLISH_CLAUSE_RE = re.compile(r"\s+(?=(?:and|but|so|then|because|which|that|while|if|when|after|before)\b)", re.IGNORECASE)


def split_for_simul(text: str, max_chars: int = 18) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    raw_parts = [part.strip() for part in PUNCTUATION_RE.split(text) if part.strip()]
    if not raw_parts:
        raw_parts = [text]

    chunks: List[str] = []
    current = ""
    for part in raw_parts:
        if not current:
            current = part
            continue
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)

    return chunks


def split_english_for_tts(text: str, max_words: int = 10, max_chars: int = 56) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    raw_parts = [part.strip() for part in ENGLISH_PUNCTUATION_RE.split(text) if part.strip()]
    if not raw_parts:
        raw_parts = [text]

    refined_parts: List[str] = []
    for part in raw_parts:
        if len(part) <= max_chars and len(part.split()) <= max_words:
            refined_parts.append(part)
            continue

        clauses = [item.strip() for item in ENGLISH_CLAUSE_RE.split(part) if item.strip()]
        if len(clauses) == 1:
            clauses = [part]

        for clause in clauses:
            words = clause.split()
            if len(words) <= max_words and len(clause) <= max_chars:
                refined_parts.append(clause)
                continue

            for index in range(0, len(words), max_words):
                refined_parts.append(" ".join(words[index : index + max_words]))

    return [part for part in refined_parts if part]


def play_audio(path: str):
    import shutil
    import subprocess
    import sys

    if sys.platform == "darwin" and shutil.which("afplay"):
        subprocess.run(["afplay", path], check=False)
    elif shutil.which("ffplay"):
        subprocess.run(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", path],
            check=False,
        )
    elif shutil.which("aplay"):
        subprocess.run(["aplay", path], check=False)


@dataclass
class SegmentResult:
    chinese: str
    english: str
    asr_ready_ms: float
    mt_ms: float
    tts_ms: float
    audio_duration_s: float
    tts_first_chunk_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "chinese": self.chinese,
            "english": self.english,
            "asr_ready_ms": self.asr_ready_ms,
            "mt_ms": self.mt_ms,
            "tts_ms": self.tts_ms,
            "audio_duration_s": self.audio_duration_s,
            "tts_first_chunk_ms": self.tts_first_chunk_ms,
        }


@dataclass
class TranslationUnit:
    chinese: str
    english: str
    asr_ready_ms: float
    mt_ms: float


class SentenceAccumulator:
    """Accumulate incremental translated text and flush sentence/clause-sized units."""

    def __init__(self, max_pending_words: int = 6, max_pending_chars: int = 40):
        self.max_pending_words = max_pending_words
        self.max_pending_chars = max_pending_chars
        self.reset()

    def reset(self):
        self._pending_chinese: List[str] = []
        self._pending_english: List[str] = []
        self._asr_ready_ms: Optional[float] = None
        self._mt_ms_total = 0.0

    def _combined_chinese(self) -> str:
        return "".join(part for part in self._pending_chinese if part).strip()

    def _combined_english(self) -> str:
        return re.sub(r"\s+", " ", " ".join(part for part in self._pending_english if part).strip())

    def _should_flush(self, english_text: str, force: bool) -> bool:
        if force:
            return True
        if not english_text:
            return False
        if re.search(r"[.!?]\s*$", english_text):
            return True
        if len(english_text) >= self.max_pending_chars:
            return True
        if len(english_text.split()) >= self.max_pending_words:
            return True
        return False

    def _emit_units(self) -> List[TranslationUnit]:
        english_text = self._combined_english()
        if not english_text:
            self.reset()
            return []

        english_units = split_english_for_tts(english_text)
        if not english_units:
            self.reset()
            return []

        chinese_text = self._combined_chinese()
        asr_ready_ms = self._asr_ready_ms or 0.0
        mt_ms_total = self._mt_ms_total
        units = [
            TranslationUnit(
                chinese=chinese_text if index == 0 else "",
                english=english_unit,
                asr_ready_ms=asr_ready_ms,
                mt_ms=mt_ms_total if index == 0 else 0.0,
            )
            for index, english_unit in enumerate(english_units)
        ]
        self.reset()
        return units

    def feed(
        self,
        chinese_text: str,
        english_text: str,
        asr_ready_ms: float,
        mt_ms: float,
        *,
        force: bool = False,
    ) -> List[TranslationUnit]:
        english_text = re.sub(r"\s+", " ", (english_text or "").strip())
        chinese_text = (chinese_text or "").strip()
        if not english_text and not force:
            return []

        if self._asr_ready_ms is None:
            self._asr_ready_ms = asr_ready_ms
        self._mt_ms_total += mt_ms
        if chinese_text:
            self._pending_chinese.append(chinese_text)
        if english_text:
            self._pending_english.append(english_text)

        combined_english = self._combined_english()
        if not self._should_flush(combined_english, force):
            return []
        return self._emit_units()

    def flush(self) -> List[TranslationUnit]:
        return self.feed("", "", self._asr_ready_ms or 0.0, 0.0, force=True)


class TTSQueueWorker:
    """Single-threaded TTS worker that streams synthesized audio from queued text units."""

    _INPUT_DONE = object()
    _OUTPUT_DONE = object()

    def __init__(
        self,
        tts_module,
        ref_audio_path: Optional[str],
        ref_text: Optional[str],
        total_start: float,
    ):
        self.tts = tts_module
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.total_start = total_start
        self._input_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._first_audio_ms: Optional[float] = None

    def start(self):
        self._thread.start()

    def enqueue(self, unit: TranslationUnit):
        self._input_queue.put(unit)

    def close(self):
        self._input_queue.put(self._INPUT_DONE)

    def join(self):
        self._thread.join()

    def emit_error(self, exc: Exception):
        self._output_queue.put({"type": "error", "error": exc, "traceback": traceback.format_exc()})

    def iter_events(self):
        while True:
            item = self._output_queue.get()
            if item is self._OUTPUT_DONE:
                break
            yield item

    def _run(self):
        try:
            segment_index = 0
            while True:
                unit = self._input_queue.get()
                if unit is self._INPUT_DONE:
                    break
                segment_index += 1
                tts_t0 = time.time()
                tts_first_chunk_ms: Optional[float] = None
                sample_rate: Optional[int] = None
                segment_wave_chunks: List[np.ndarray] = []

                for chunk_index, (waveform_chunk, sample_rate, is_final_chunk) in enumerate(
                    self.tts.synthesize_stream(unit.english, self.ref_audio_path, self.ref_text),
                    start=1,
                ):
                    now = time.time()
                    if tts_first_chunk_ms is None:
                        tts_first_chunk_ms = (now - tts_t0) * 1000
                    if self._first_audio_ms is None:
                        self._first_audio_ms = (now - self.total_start) * 1000
                    segment_wave_chunks.append(waveform_chunk)
                    self._output_queue.put(
                        {
                            "type": "audio_chunk",
                            "segment_index": segment_index,
                            "chunk_index": chunk_index,
                            "sample_rate": sample_rate,
                            "waveform": waveform_chunk,
                            "is_final_chunk": is_final_chunk,
                            "first_audio_ms": self._first_audio_ms,
                            "tts_first_chunk_ms": tts_first_chunk_ms,
                        }
                    )

                if not segment_wave_chunks or sample_rate is None:
                    continue

                waveform = np.concatenate(segment_wave_chunks).astype(np.float32, copy=False)
                tts_ms = (time.time() - tts_t0) * 1000
                segment = SegmentResult(
                    chinese=unit.chinese,
                    english=unit.english,
                    asr_ready_ms=unit.asr_ready_ms,
                    mt_ms=unit.mt_ms,
                    tts_ms=tts_ms,
                    audio_duration_s=len(waveform) / sample_rate,
                    tts_first_chunk_ms=tts_first_chunk_ms,
                )
                print(
                    f"  [chunk] ASR-ready={unit.asr_ready_ms:.0f}ms | MT={unit.mt_ms:.0f}ms | "
                    f"TTS-first={tts_first_chunk_ms or 0:.0f}ms | TTS-total={tts_ms:.0f}ms | "
                    f"CN={unit.chinese} | EN={unit.english}"
                )
                self._output_queue.put(
                    {
                        "type": "segment",
                        "segment_index": segment_index,
                        "segment": segment,
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                        "first_audio_ms": self._first_audio_ms,
                    }
                )
        except Exception as exc:
            self._output_queue.put({"type": "error", "error": exc, "traceback": traceback.format_exc()})
        finally:
            self._output_queue.put(self._OUTPUT_DONE)


class ASRModule:
    """Chinese ASR using mlx-audio Qwen3-ASR on Apple Silicon."""

    MODEL_ALIASES = {
        "qwen3-asr": "mlx-community/Qwen3-ASR-0.6B-bf16",
        "qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-bf16",
        "mlx-qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-bf16",
    }

    def __init__(self, model_size: str = "mlx-community/Qwen3-ASR-0.6B-bf16", batch_size: int = 12):
        from mlx_audio.stt.utils import load_model

        resolved_model = self.MODEL_ALIASES.get(model_size, model_size)
        self.model_name = resolved_model
        self.sample_rate = 16000
        self.batch_size = batch_size
        print(f"[ASR] Loading mlx-audio ASR ({resolved_model})...")
        self.model = load_model(resolved_model, lazy=False)
        print("[ASR] Ready on Apple MLX.")

    def _normalize_audio(self, filepath: str) -> Tuple[str, bool]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return filepath, False

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="simul-asr-")
        handle.close()
        normalized_path = handle.name
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            filepath,
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            normalized_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            os.unlink(normalized_path)
            raise RuntimeError(f"ASR audio normalization failed: {result.stderr.strip()}")
        return normalized_path, True

    def _transcribe(self, filepath: str):
        normalized_path, cleanup = self._normalize_audio(filepath)
        try:
            return self.model.generate(
                normalized_path,
                language="zh",
                stream=False,
                verbose=False,
            )
        finally:
            if cleanup:
                try:
                    os.unlink(normalized_path)
                except OSError:
                    pass

    def _stream_transcribe(self, filepath: str):
        normalized_path, cleanup = self._normalize_audio(filepath)
        try:
            yield from self.model.generate(
                normalized_path,
                language="zh",
                stream=True,
                verbose=False,
            )
        finally:
            if cleanup:
                try:
                    os.unlink(normalized_path)
                except OSError:
                    pass

    @staticmethod
    def _output_text(result) -> str:
        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        return (getattr(result, "text", "") or "").strip()

    @staticmethod
    def _output_segments(result):
        if isinstance(result, dict):
            return result.get("segments") or []
        return getattr(result, "segments", []) or []

    def transcribe_file(self, filepath: str) -> str:
        result = self._transcribe(filepath)
        return self._output_text(result)

    def transcribe_segments(self, filepath: str) -> Iterable[Tuple[float, float, str]]:
        buffered_text = ""
        segment_start = 0.0
        segment_end = 0.0

        for item in self._stream_transcribe(filepath):
            text = self._output_text(item)
            start = float(getattr(item, "start_time", segment_end) or segment_end)
            end = float(getattr(item, "end_time", start) or start)
            is_final = bool(getattr(item, "is_final", False))

            if text:
                if not buffered_text:
                    segment_start = start
                buffered_text += text
                segment_end = end

            flush_now = False
            stripped = buffered_text.strip()
            if stripped:
                if re.search(r"[，。！？；,.!?;:]$", stripped):
                    flush_now = True
                elif len(stripped) >= 12:
                    flush_now = True

            if flush_now:
                yield segment_start, segment_end, stripped
                buffered_text = ""

            if is_final:
                final_text = buffered_text.strip()
                if final_text:
                    yield segment_start, max(segment_end, end), final_text
                break


class MTModule:
    """Chinese-to-English MT via ollama or OpenAI-compatible vLLM server."""

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: int = 15,
    ):
        self.backend = backend
        self.model = model or ("qwen3:1.7b" if backend == "ollama" else "Qwen/Qwen2.5-1.5B-Instruct")
        self.base_url = (base_url or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout_s = timeout_s

        if backend == "ollama":
            import ollama

            ollama.list()
            print(f"[MT] Using ollama model: {self.model}")
        elif backend == "openai":
            print(f"[MT] Using OpenAI-compatible server: {self.base_url} ({self.model})")
        else:
            raise ValueError("mt backend must be one of: ollama, openai")

    def translate(self, chinese_text: str) -> str:
        if self.backend == "ollama":
            return self._translate_ollama(chinese_text)
        return self._translate_openai(chinese_text)

    def _translate_ollama(self, chinese_text: str) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a simultaneous interpreter.\n"
                        "Translate Chinese into short, natural spoken English.\n"
                        "Output English only.\n"
                        "Do not explain.\n"
                        "Do not think aloud.\n"
                        "Do not include notes or tags."
                    ),
                },
                {"role": "user", "content": f"/no_think\n{chinese_text}"},
            ],
            options={
                "temperature": 0,
                "num_predict": 48,
                "num_ctx": 512,
            },
        )
        text = response["message"]["content"]
        text = re.sub(r"<think\b[^>]*>.*?</think\b[^>]*>", "", text, flags=re.DOTALL)
        text = re.sub(r"(?is)^.*?</think>", "", text).strip()
        text = re.sub(r"(?is)^okay[,.\s].*?(?=\b[A-Z][a-z]|\bYou know\b)", "", text).strip()
        return text.splitlines()[0].strip()

    def _translate_openai(self, chinese_text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a simultaneous interpreter. "
                        "Translate Chinese into short, natural spoken English. "
                        "Output English only. Do not explain or think aloud."
                    ),
                },
                {"role": "user", "content": chinese_text},
            ],
            "temperature": 0,
            "max_tokens": 48,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"openai-compatible MT request failed: {exc}") from exc
        text = body["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think\b[^>]*>.*?</think\b[^>]*>", "", text, flags=re.DOTALL).strip()
        return text.splitlines()[0].strip()


class TTSModule:
    """English TTS with voice cloning via mlx-audio Qwen3-TTS."""

    MODEL_ALIASES = {
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "mlx-qwen3-tts-0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "mlx-qwen3-tts-1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    }

    def __init__(self, model_path: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"):
        from mlx_audio.tts.utils import load

        resolved_model = self.MODEL_ALIASES.get(model_path, model_path)
        self.model_id = resolved_model
        self.device = "mlx"
        self._lock = threading.Lock()
        self._ref_audio_path: Optional[str] = None
        self._ref_text: Optional[str] = None
        self._ref_audio_waveform = None
        self._ref_speaker_embedding = None
        self.use_ref_text = False
        self.streaming_interval_s = 0.3

        print(f"[TTS] Loading mlx-audio model: {resolved_model}")
        self.model = load(resolved_model, lazy=False)
        self.sample_rate = getattr(self.model, "sample_rate", 24000)
        print(f"[TTS] Ready on {self.device} at {self.sample_rate}Hz.")

    def create_voice_reference(self, ref_audio_path: str, ref_text: str):
        from mlx_audio.utils import load_audio

        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text
        self._ref_audio_waveform = load_audio(ref_audio_path, sample_rate=self.sample_rate)
        self._ref_speaker_embedding = None
        if hasattr(self.model, "extract_speaker_embedding"):
            t0 = time.time()
            self._ref_speaker_embedding = self.model.extract_speaker_embedding(self._ref_audio_waveform)
            print(f"[TTS] Cached speaker embedding in {(time.time() - t0) * 1000:.0f}ms.")
        if ref_text and self.use_ref_text:
            print("[TTS] Voice reference registered for mlx-audio (audio + transcript conditioning).")
        else:
            print("[TTS] Voice reference registered for mlx-audio (audio-only speaker conditioning).")

    @contextmanager
    def _generation_session(
        self,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        active_ref_audio = ref_audio_path or self._ref_audio_path
        active_ref_text = ref_text or self._ref_text
        if not active_ref_audio:
            raise ValueError("voice clone reference audio is required")

        use_cached_ref = active_ref_audio == self._ref_audio_path and self._ref_audio_waveform is not None
        ref_audio_input = self._ref_audio_waveform if use_cached_ref else active_ref_audio

        self._lock.acquire()
        original_extract = None
        if use_cached_ref and self._ref_speaker_embedding is not None:
            original_extract = self.model.extract_speaker_embedding

            def _cached_extract_speaker_embedding(audio, sr=24000):
                return self._ref_speaker_embedding

            self.model.extract_speaker_embedding = _cached_extract_speaker_embedding
        try:
            yield ref_audio_input, active_ref_text
        finally:
            if original_extract is not None:
                self.model.extract_speaker_embedding = original_extract
            self._lock.release()

    def synthesize(
        self,
        english_text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        generate_kwargs = {
            "text": english_text,
            "lang_code": "english",
            "stream": False,
        }
        with self._generation_session(ref_audio_path, ref_text) as (ref_audio_input, active_ref_text):
            generate_kwargs["ref_audio"] = ref_audio_input
            if self.use_ref_text and active_ref_text:
                generate_kwargs["ref_text"] = active_ref_text
            results = list(self.model.generate(**generate_kwargs))
        if not results:
            raise RuntimeError("mlx-audio returned no audio")

        chunks = [
            np.asarray(result.audio)
            for result in results
            if getattr(result, "audio", None) is not None
        ]
        if not chunks:
            raise RuntimeError("mlx-audio returned empty audio chunks")

        waveform = np.concatenate(chunks).astype(np.float32, copy=False)
        return waveform, self.sample_rate

    def synthesize_stream(
        self,
        english_text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        yielded = False
        generate_kwargs = {
            "text": english_text,
            "lang_code": "english",
            "stream": True,
            "streaming_interval": self.streaming_interval_s,
        }
        with self._generation_session(ref_audio_path, ref_text) as (ref_audio_input, active_ref_text):
            generate_kwargs["ref_audio"] = ref_audio_input
            if self.use_ref_text and active_ref_text:
                generate_kwargs["ref_text"] = active_ref_text
            for result in self.model.generate(**generate_kwargs):
                audio = getattr(result, "audio", None)
                if audio is None:
                    continue
                waveform = np.asarray(audio).astype(np.float32, copy=False)
                if waveform.size == 0:
                    continue
                yielded = True
                yield waveform, self.sample_rate, bool(getattr(result, "is_final_chunk", False))
        if not yielded:
            raise RuntimeError("mlx-audio returned no streamed audio")


class SimultaneousTranslator:
    def __init__(
        self,
        asr_model_size: str = "mlx-community/Qwen3-ASR-0.6B-bf16",
        mt_backend: str = "ollama",
        mt_model: Optional[str] = None,
        mt_base_url: Optional[str] = None,
        tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    ):
        print("\n" + "=" * 56)
        print("Initializing Simultaneous Translator")
        print("=" * 56)

        self.asr = ASRModule(model_size=asr_model_size)
        self.mt = MTModule(backend=mt_backend, model=mt_model, base_url=mt_base_url)
        self.tts = TTSModule(model_path=tts_model)

        print("=" * 56)
        print("Modules ready.\n")

    def _iter_translation_units(self, audio_path: str, total_start: Optional[float] = None):
        total_start = total_start or time.time()
        accumulator = SentenceAccumulator()

        for _, _, raw_text in self.asr.transcribe_segments(audio_path):
            asr_ready_ms = (time.time() - total_start) * 1000
            for chinese_text in split_for_simul(raw_text):
                mt_t0 = time.time()
                english_text = self.mt.translate(chinese_text)
                mt_ms = (time.time() - mt_t0) * 1000
                if not english_text:
                    continue
                for unit in accumulator.feed(chinese_text, english_text, asr_ready_ms, mt_ms):
                    yield unit

        for unit in accumulator.flush():
            yield unit

    def translate_text(self, chinese_text: str, ref_audio_path: str, ref_text: str) -> Tuple[np.ndarray, int, Dict]:
        total_start = time.time()

        mt_t0 = time.time()
        english_text = self.mt.translate(chinese_text)
        mt_ms = (time.time() - mt_t0) * 1000

        tts_t0 = time.time()
        waveform, sr = self.tts.synthesize(english_text, ref_audio_path, ref_text)
        tts_ms = (time.time() - tts_t0) * 1000
        total_ms = (time.time() - total_start) * 1000

        return waveform, sr, {
            "chinese": chinese_text,
            "english": english_text,
            "mt_ms": mt_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
        }

    def translate_audio(
        self,
        audio_path: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Dict]:
        total_start = time.time()

        asr_t0 = time.time()
        chinese_text = self.asr.transcribe_file(audio_path)
        asr_ms = (time.time() - asr_t0) * 1000
        if not chinese_text:
            return None, None, {}

        waveform, sr, info = self.translate_text(chinese_text, ref_audio_path, ref_text)
        info["asr_ms"] = asr_ms
        info["total_ms"] = (time.time() - total_start) * 1000
        return waveform, sr, info

    def translate_audio_streaming(
        self,
        audio_path: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Dict]:
        total_start = time.time()
        segment_results: List[SegmentResult] = []
        wave_chunks: List[np.ndarray] = []
        sample_rate: Optional[int] = None
        first_audio_ms: Optional[float] = None

        for waveform, sample_rate, segment, first_audio_ms in self.iter_audio_streaming(
            audio_path, ref_audio_path, ref_text
        ):
            wave_chunks.append(waveform)
            segment_results.append(segment)

        if not wave_chunks or sample_rate is None:
            return None, None, {}

        merged = np.concatenate(wave_chunks)
        total_ms = (time.time() - total_start) * 1000
        return merged, sample_rate, {
            "segments": [segment.__dict__ for segment in segment_results],
            "first_audio_ms": first_audio_ms,
            "total_ms": total_ms,
            "segment_count": len(segment_results),
        }

    def iter_audio_streaming(
        self,
        audio_path: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        total_start = time.time()
        first_audio_ms: Optional[float] = None

        for unit in self._iter_translation_units(audio_path, total_start=total_start):
            tts_t0 = time.time()
            waveform, sample_rate = self.tts.synthesize(unit.english, ref_audio_path, ref_text)
            tts_ms = (time.time() - tts_t0) * 1000
            if first_audio_ms is None:
                first_audio_ms = (time.time() - total_start) * 1000

            segment = SegmentResult(
                chinese=unit.chinese,
                english=unit.english,
                asr_ready_ms=unit.asr_ready_ms,
                mt_ms=unit.mt_ms,
                tts_ms=tts_ms,
                audio_duration_s=len(waveform) / sample_rate,
            )
            print(
                f"  [chunk] ASR-ready={unit.asr_ready_ms:.0f}ms | MT={unit.mt_ms:.0f}ms | "
                f"TTS={tts_ms:.0f}ms | CN={unit.chinese} | EN={unit.english}"
            )
            yield waveform, sample_rate, segment, first_audio_ms

    def iter_audio_streaming_events(
        self,
        audio_path: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        total_start = time.time()
        worker = TTSQueueWorker(self.tts, ref_audio_path, ref_text, total_start)
        worker.start()
        producer_error: List[BaseException] = []

        def produce_units():
            try:
                for unit in self._iter_translation_units(audio_path, total_start=total_start):
                    worker.enqueue(unit)
            except BaseException as exc:
                producer_error.append(exc)
                worker.emit_error(exc)
            finally:
                worker.close()

        producer = threading.Thread(target=produce_units, daemon=True)
        producer.start()

        try:
            for event in worker.iter_events():
                if event.get("type") == "error":
                    raise RuntimeError(event.get("traceback") or str(event.get("error")))
                yield event
        finally:
            producer.join()
            worker.join()

        if producer_error:
            raise RuntimeError(str(producer_error[0]))


def write_temp_wav(frames: List[bytes], sample_rate: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        with wave.open(handle.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return handle.name


def run_live_mode(translator: SimultaneousTranslator, args):
    import soundfile as sf
    import pyaudio
    import webrtcvad

    rate = 16000
    frame_ms = 30
    samples_per_frame = int(rate * frame_ms / 1000)
    bytes_per_frame = samples_per_frame * 2
    max_frames = max(1, int(args.max_chunk_ms / frame_ms))
    trailing_silence_frames = max(1, int(args.trailing_silence_ms / frame_ms))

    audio = pyaudio.PyAudio()
    vad = webrtcvad.Vad(args.vad_aggressiveness)
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=samples_per_frame,
    )

    print("\n" + "=" * 56)
    print("Live mode: speak Chinese, hear English clone")
    print("=" * 56)

    frames: List[bytes] = []
    speech_started = False
    silence_count = 0

    try:
        while True:
            data = stream.read(samples_per_frame, exception_on_overflow=False)
            if len(data) < bytes_per_frame:
                continue

            is_speech = vad.is_speech(data, rate)
            if is_speech:
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1

            if speech_started:
                frames.append(data)

            should_flush = speech_started and (
                len(frames) >= max_frames or silence_count >= trailing_silence_frames
            )
            if not should_flush:
                continue

            chunk_path = write_temp_wav(frames, rate)
            frames = []
            speech_started = False
            silence_count = 0

            try:
                if args.streaming:
                    waveform, sr, info = translator.translate_audio_streaming(
                        chunk_path, args.ref_audio, args.ref_text
                    )
                else:
                    waveform, sr, info = translator.translate_audio(
                        chunk_path, args.ref_audio, args.ref_text
                    )
            finally:
                os.unlink(chunk_path)

            if waveform is None:
                continue

            sf.write(args.output, waveform, sr)
            if info.get("first_audio_ms") is not None:
                print(
                    f"  [live] first_audio={info['first_audio_ms']:.0f}ms | "
                    f"total={info['total_ms']:.0f}ms | chunks={info['segment_count']}"
                )
            else:
                print(
                    f"  [live] ASR={info['asr_ms']:.0f}ms | MT={info['mt_ms']:.0f}ms | "
                    f"TTS={info['tts_ms']:.0f}ms | total={info['total_ms']:.0f}ms"
                )
            play_audio(args.output)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def build_parser():
    parser = argparse.ArgumentParser(description="Low-latency simultaneous translation demo")
    parser.add_argument("audio_file", nargs="?", help="Chinese WAV file")
    parser.add_argument("--live", action="store_true", help="Microphone mode")
    parser.add_argument("--streaming", action="store_true", help="Use segmented ASR/MT/TTS path")
    parser.add_argument("--ref-audio", help="Reference WAV for voice cloning")
    parser.add_argument("--ref-text", help="Transcript of the reference audio")
    parser.add_argument("--output", default="output_english.wav", help="Output WAV path")
    parser.add_argument(
        "--asr-model",
        default="mlx-community/Qwen3-ASR-0.6B-bf16",
        help="mlx-audio ASR model",
    )
    parser.add_argument(
        "--mt-backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="Use ollama or an OpenAI-compatible vLLM server",
    )
    parser.add_argument("--mt-model", help="Override MT model name")
    parser.add_argument("--mt-base-url", help="OpenAI-compatible base URL, such as http://127.0.0.1:8000/v1")
    parser.add_argument("--max-chunk-ms", type=int, default=1400, help="Live mode max speech chunk length")
    parser.add_argument("--trailing-silence-ms", type=int, default=360, help="Live mode silence to flush")
    parser.add_argument("--vad-aggressiveness", type=int, choices=[0, 1, 2, 3], default=2)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    translator = SimultaneousTranslator(
        asr_model_size=args.asr_model,
        mt_backend=args.mt_backend,
        mt_model=args.mt_model,
        mt_base_url=args.mt_base_url,
    )

    if args.ref_audio:
        translator.tts.create_voice_reference(args.ref_audio, args.ref_text or "")

    if args.live:
        if not args.ref_audio:
            parser.error("--live requires --ref-audio for voice cloning")
        run_live_mode(translator, args)
        return

    if not args.audio_file:
        parser.error("provide an audio file or use --live")

    if args.streaming:
        waveform, sr, info = translator.translate_audio_streaming(
            args.audio_file, args.ref_audio, args.ref_text
        )
    else:
        waveform, sr, info = translator.translate_audio(
            args.audio_file, args.ref_audio, args.ref_text
        )

    if waveform is None:
        print("No speech detected.")
        return

    import soundfile as sf

    sf.write(args.output, waveform, sr)
    print(f"\nOutput: {args.output}")
    if args.streaming:
        print(
            f"First audio: {info['first_audio_ms']:.0f}ms | "
            f"Total: {info['total_ms']:.0f}ms | Segments: {info['segment_count']}"
        )
        for index, segment in enumerate(info["segments"], start=1):
            print(
                f"  {index}. CN={segment['chinese']} | EN={segment['english']} | "
                f"ASR-ready={segment['asr_ready_ms']:.0f}ms | MT={segment['mt_ms']:.0f}ms | "
                f"TTS={segment['tts_ms']:.0f}ms"
            )
    else:
        print(
            f"CN: {info['chinese']}\nEN: {info['english']}\n"
            f"Latency: ASR={info['asr_ms']:.0f}ms | MT={info['mt_ms']:.0f}ms | "
            f"TTS={info['tts_ms']:.0f}ms | Total={info['total_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()

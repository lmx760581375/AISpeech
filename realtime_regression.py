"""Reproducible end-to-end regression test for the realtime translator.

The test sends randomly sized overlapping audio windows through the HTTP API,
then reports text coverage, stage latencies, queue behavior, and basic output
audio health. On macOS, ``--synthetic-fixture`` generates a known Chinese
speech fixture with ``say`` so the test has no external audio dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf


FIXTURE_REFERENCE_TEXT = "你好，这是我的声音。接下来我会用自然的语气说几句话，让系统学习我的语速、停顿和情绪。"
FIXTURE_SOURCE_TEXT = (
    "大家晚上好，刚才我在路上看到天空突然放晴，心情一下子轻松了很多。"
    "不过今天的工作还是比较忙，我们先把最重要的问题解决，再慢慢讨论那些细节。"
    "如果你已经准备好了，就告诉我一声；要是还需要一点时间，也完全没有关系。"
    "我希望这次翻译听起来像自然的对话，有正常的停顿，也能把每句话完整地说完。"
    "最后我们确认一下计划，明天上午十点见，到时候我会带上最新的结果。"
)


def request_json(url: str, body: bytes | None = None, content_type: str | None = None) -> Dict:
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def make_multipart(fields: Dict[str, str], audio_path: Path) -> Tuple[bytes, str]:
    boundary = f"----AISpeech{uuid.uuid4().hex}"
    parts: List[bytes] = []
    for key, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="ref_audio"; filename="{audio_path.name}"\r\n'.encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def build_synthetic_fixture(directory: Path) -> Tuple[Path, Path, str, str]:
    reference_aiff = directory / "reference.aiff"
    source_aiff = directory / "source.aiff"
    reference_wav = directory / "reference.wav"
    source_wav = directory / "source-16k.wav"
    subprocess.run(["say", "-v", "Tingting", "-r", "190", "-o", str(reference_aiff), FIXTURE_REFERENCE_TEXT], check=True)
    subprocess.run(["say", "-v", "Tingting", "-r", "185", "-o", str(source_aiff), FIXTURE_SOURCE_TEXT], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(reference_aiff), "-ar", "24000", "-ac", "1", str(reference_wav)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source_aiff), "-ar", "16000", "-ac", "1", str(source_wav)], check=True)
    return reference_wav, source_wav, FIXTURE_REFERENCE_TEXT, FIXTURE_SOURCE_TEXT


def percentile(values: Iterable[float], value: int) -> float | None:
    values = list(values)
    return round(float(np.percentile(values, value)), 1) if values else None


def lcs_ratio(expected: str, actual: str) -> float:
    expected = "".join(expected.split())
    actual = "".join(actual.split())
    if not expected:
        return 1.0
    previous = [0] * (len(actual) + 1)
    for left in expected:
        current = [0]
        for index, right in enumerate(actual, start=1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(expected)


def audio_health(audio_bytes: bytes) -> Dict:
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    frame_size = max(1, int(sample_rate * 0.02))
    frame_rms = np.array(
        [np.sqrt(np.mean(waveform[index : index + frame_size] ** 2)) for index in range(0, len(waveform), frame_size)]
    )
    return {
        "duration_s": round(len(waveform) / sample_rate, 3),
        "sample_rate": sample_rate,
        "peak": round(float(np.max(np.abs(waveform))), 5),
        "clipped_fraction": round(float(np.mean(np.abs(waveform) >= 0.995)), 8),
        "near_silence_frame_fraction": round(float(np.mean(frame_rms < 0.003)), 4),
        "rms_p05": round(float(np.percentile(frame_rms, 5)), 5),
        "rms_p50": round(float(np.percentile(frame_rms, 50)), 5),
    }


def run_regression(args, reference_audio: Path, source_audio: Path, reference_text: str, expected_text: str) -> Dict:
    base_url = args.url.rstrip("/")
    form, content_type = make_multipart(
        {"ref_text": reference_text, "window_ms": str(args.window_ms), "hop_ms": str(args.hop_ms)},
        reference_audio,
    )
    session = request_json(f"{base_url}/api/realtime/session", form, content_type)
    session_id = session["session_id"]
    samples, sample_rate = sf.read(source_audio, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration_ms = int(round(len(samples) / sample_rate * 1000))
    randomizer = random.Random(args.seed)
    start_ms = 0
    sequence = 0
    windows = []
    while start_ms < duration_ms:
        window_ms = randomizer.randint(args.window_ms - args.window_jitter_ms, args.window_ms + args.window_jitter_ms)
        hop_ms = randomizer.randint(args.hop_ms - args.hop_jitter_ms, args.hop_ms + args.hop_jitter_ms)
        window_ms = max(600, window_ms)
        hop_ms = max(300, min(hop_ms, window_ms - 1))
        end_ms = min(duration_ms, start_ms + window_ms)
        start = int(start_ms * sample_rate / 1000)
        end = int(end_ms * sample_rate / 1000)
        buffer = io.BytesIO()
        sf.write(buffer, samples[start:end], sample_rate, format="WAV", subtype="PCM_16")
        sequence += 1
        request_json(
            f"{base_url}/api/realtime/chunk?session_id={session_id}&seq={sequence}&start_ms={start_ms}&end_ms={end_ms}",
            buffer.getvalue(),
            "audio/wav",
        )
        windows.append({"start_ms": start_ms, "end_ms": end_ms, "hop_ms": hop_ms})
        if args.pace:
            time.sleep(hop_ms / 1000)
        start_ms += hop_ms

    request_json(f"{base_url}/api/realtime/finalize", json.dumps({"session_id": session_id}).encode(), "application/json")
    deadline = time.time() + args.timeout_s
    payload: Dict = {}
    while time.time() < deadline:
        payload = request_json(f"{base_url}/api/realtime/events?session_id={session_id}")
        if payload.get("done") or payload.get("error"):
            break
        time.sleep(1)
    if not payload.get("done"):
        raise TimeoutError(f"realtime session did not finish: {payload.get('error') or 'timeout'}")
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    events = payload["events"]
    segments = [event["segment"] for event in events if event.get("type") == "segment"]
    partials = [event.get("partial", "") for event in events if event.get("type") == "partial"]
    done_event = next(event for event in events if event.get("type") == "done")
    audio_bytes = b""
    if done_event.get("audio_url"):
        with urllib.request.urlopen(f"{base_url}{done_event['audio_url']}", timeout=30) as response:
            audio_bytes = response.read()
    chinese_output = "".join(segment.get("chinese", "") for segment in segments)
    metrics = {
        "session_id": session_id,
        "source_duration_s": round(duration_ms / 1000, 3),
        "windows_sent": len(windows),
        "segments": len(segments),
        "text_coverage": round(lcs_ratio(expected_text, chinese_output), 4),
        "partials": len(partials),
        "asr_ms": {"p50": percentile((item["asr_ms"] for item in segments if item["asr_ms"]), 50), "p95": percentile((item["asr_ms"] for item in segments if item["asr_ms"]), 95)},
        "mt_ms": {"p50": percentile((item["mt_ms"] for item in segments if item["mt_ms"]), 50), "p95": percentile((item["mt_ms"] for item in segments if item["mt_ms"]), 95)},
        "tts_first_chunk_ms": {"p50": percentile((item["tts_first_chunk_ms"] for item in segments if item["tts_first_chunk_ms"]), 50), "p95": percentile((item["tts_first_chunk_ms"] for item in segments if item["tts_first_chunk_ms"]), 95)},
        "tts_total_ms": {"p50": percentile((item["tts_ms"] for item in segments), 50), "p95": percentile((item["tts_ms"] for item in segments), 95)},
        "lag_ms": {"max": max((item["lag_ms"] for item in segments), default=0), "last": segments[-1]["lag_ms"] if segments else 0},
        "server": done_event.get("timings", {}),
        "english": " ".join(segment.get("english", "") for segment in segments),
        "audio": audio_health(audio_bytes) if audio_bytes else None,
        "windows": windows,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression test for realtime_web_demo.py")
    parser.add_argument("--url", default="http://127.0.0.1:7870")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--source-audio")
    parser.add_argument("--expected-text", default="")
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--window-ms", type=int, default=2200)
    parser.add_argument("--hop-ms", type=int, default=1000)
    parser.add_argument("--window-jitter-ms", type=int, default=200)
    parser.add_argument("--hop-jitter-ms", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--pace", action="store_true", help="Send chunks at their simulated real-time hop interval")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="aispeech-realtime-regression-") as directory:
        if args.synthetic_fixture:
            reference_audio, source_audio, reference_text, expected_text = build_synthetic_fixture(Path(directory))
        else:
            if not (args.ref_audio and args.ref_text and args.source_audio):
                parser.error("provide --synthetic-fixture or --ref-audio, --ref-text, and --source-audio")
            reference_audio = Path(args.ref_audio)
            source_audio = Path(args.source_audio)
            reference_text = args.ref_text
            expected_text = args.expected_text
        metrics = run_regression(args, reference_audio, source_audio, reference_text, expected_text)

    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(report)
    if args.json_out:
        Path(args.json_out).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

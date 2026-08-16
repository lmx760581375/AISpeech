"""Profile steady-state MLX Qwen3-TTS voice cloning without changing the server.

Example:
    conda run -n test python profile_mlx_tts.py \
      --ref-audio test_ref.wav --ref-text "参考音频文本" --profile-out /tmp/qwen3tts.prof
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import mlx.core as mx

from demo import DEFAULT_TTS_MODEL, TTSModule


DEFAULT_TEXT = "This is a short latency benchmark for the simultaneous translation demo."


@dataclass
class RunResult:
    interval_s: float
    first_chunk_ms: float
    total_ms: float
    audio_duration_s: float
    realtime_factor: float
    chunks: int
    peak_memory_gb: float


def run_once(tts: TTSModule, text: str, interval_s: float, options: Dict) -> RunResult:
    tts.streaming_interval_s = interval_s
    mx.reset_peak_memory()
    start = time.perf_counter()
    first_chunk_ms = None
    chunks = 0
    samples = 0
    sample_rate = None

    for waveform, sample_rate, _ in tts.synthesize_stream(text, generation_options=options):
        chunks += 1
        samples += len(waveform)
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - start) * 1000

    if sample_rate is None or first_chunk_ms is None:
        raise RuntimeError("TTS produced no audio")

    total_ms = (time.perf_counter() - start) * 1000
    audio_duration_s = samples / sample_rate
    return RunResult(
        interval_s=interval_s,
        first_chunk_ms=first_chunk_ms,
        total_ms=total_ms,
        audio_duration_s=audio_duration_s,
        realtime_factor=total_ms / 1000 / max(audio_duration_s, 1e-6),
        chunks=chunks,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
    )


def summarize(runs: List[RunResult]) -> Dict:
    return {
        "runs": [asdict(run) for run in runs],
        "median": {
            "first_chunk_ms": statistics.median(run.first_chunk_ms for run in runs),
            "total_ms": statistics.median(run.total_ms for run in runs),
            "realtime_factor": statistics.median(run.realtime_factor for run in runs),
            "peak_memory_gb": max(run.peak_memory_gb for run in runs),
        },
    }


def print_summary(interval_s: float, summary: Dict):
    median = summary["median"]
    print(
        f"interval={interval_s:.2f}s | first={median['first_chunk_ms']:.0f}ms | "
        f"total={median['total_ms']:.0f}ms | RTF={median['realtime_factor']:.3f}x | "
        f"peak={median['peak_memory_gb']:.2f}GB"
    )


def write_profile(profile: cProfile.Profile, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(path))
    text_path = path.with_suffix(".txt")
    with text_path.open("w") as handle:
        pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(80)
    print(f"Python profile: {path} and {text_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile steady-state MLX Qwen3-TTS voice cloning")
    parser.add_argument("--model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--ref-audio", default="test_ref.wav")
    parser.add_argument("--ref-text", default="参考音频文本")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--interval", action="append", type=float, dest="intervals")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per interval after warmup")
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for repeatable comparisons")
    parser.add_argument("--profile-out", help="Optional cProfile output path for the first measured run")
    parser.add_argument("--json-out", help="Optional JSON summary path")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    intervals = args.intervals or [0.16, 0.32, 0.48]
    if any(interval <= 0 for interval in intervals):
        parser.error("--interval must be positive")

    options = {"temperature": args.temperature, "repetition_penalty": 1.5}
    tts = TTSModule(args.model)
    tts.create_voice_reference(args.ref_audio, args.ref_text)

    # Warm the model and cached voice-conditioning path before measuring latency.
    for _ in tts.synthesize_stream("Hello.", generation_options=options):
        pass

    summaries = {}
    profile_pending = Path(args.profile_out) if args.profile_out else None
    for interval_s in intervals:
        runs = []
        for _ in range(args.runs):
            if profile_pending is None:
                runs.append(run_once(tts, args.text, interval_s, options))
                continue
            profile = cProfile.Profile()
            profile.enable()
            try:
                runs.append(run_once(tts, args.text, interval_s, options))
            finally:
                profile.disable()
            write_profile(profile, profile_pending)
            profile_pending = None

        summary = summarize(runs)
        summaries[str(interval_s)] = summary
        print_summary(interval_s, summary)

    payload = {
        "model": args.model,
        "text": args.text,
        "reference_audio": args.ref_audio,
        "reference_text": args.ref_text,
        "generation_options": options,
        "intervals": summaries,
    }
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"JSON summary: {path}")


if __name__ == "__main__":
    main()

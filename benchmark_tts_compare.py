"""
Benchmark current MLX Qwen3-TTS against an optional external ONNX runner.

Usage examples:
    python benchmark_tts_compare.py
    python benchmark_tts_compare.py --text "Hello world."
    python benchmark_tts_compare.py --onnx-runner /path/to/runner.py --onnx-model-dir /path/to/model
    python benchmark_tts_compare.py --json-out benchmark_results.json

Notes:
    - The MLX backend is benchmarked directly in-process.
    - The ONNX backend is intentionally adapter-based. This repo does not yet ship a
      complete Python pipeline for the multi-file ONNX export, so the script supports
      comparing against an external runner once available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from demo import TTSModule


DEFAULT_TEXTS = [
    "Hello.",
    "This is a short latency benchmark for the simultaneous translation demo.",
    "I think we should start from the data we already have, and then iterate quickly while keeping the voice natural and stable.",
]


@dataclass
class BenchmarkResult:
    backend: str
    label: str
    success: bool
    first_chunk_ms: Optional[float] = None
    total_ms: Optional[float] = None
    audio_duration_s: Optional[float] = None
    realtime_factor: Optional[float] = None
    chunks: Optional[int] = None
    sample_rate: Optional[int] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict] = None


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} ms"


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def benchmark_mlx(
    text: str,
    ref_audio: str,
    ref_text: str,
    output_dir: Path,
    model_id: str,
) -> BenchmarkResult:
    label = text if len(text) <= 72 else text[:69] + "..."
    try:
        tts = TTSModule(model_id)
        tts.create_voice_reference(ref_audio, ref_text)
        start = time.time()
        first_chunk_ms = None
        chunks: List[np.ndarray] = []
        sample_rate: Optional[int] = None
        chunk_count = 0

        for chunk_count, (waveform_chunk, sample_rate, _is_final_chunk) in enumerate(
            tts.synthesize_stream(text),
            start=1,
        ):
            if first_chunk_ms is None:
                first_chunk_ms = (time.time() - start) * 1000
            chunks.append(waveform_chunk)

        if not chunks or sample_rate is None:
            raise RuntimeError("MLX backend returned no audio")

        waveform = np.concatenate(chunks).astype(np.float32, copy=False)
        total_ms = (time.time() - start) * 1000
        audio_duration_s = len(waveform) / sample_rate
        realtime_factor = total_ms / 1000 / max(audio_duration_s, 1e-6)

        import soundfile as sf

        output_path = output_dir / "mlx" / f"{slugify(label)}.wav"
        ensure_parent(output_path)
        sf.write(output_path, waveform, sample_rate)

        return BenchmarkResult(
            backend="mlx",
            label=label,
            success=True,
            first_chunk_ms=first_chunk_ms,
            total_ms=total_ms,
            audio_duration_s=audio_duration_s,
            realtime_factor=realtime_factor,
            chunks=chunk_count,
            sample_rate=sample_rate,
            output_path=str(output_path),
            details={"model_id": model_id},
        )
    except Exception as exc:
        return BenchmarkResult(
            backend="mlx",
            label=label,
            success=False,
            error=str(exc),
            details={"model_id": model_id},
        )


def probe_onnx_environment(model_dir: Optional[str]) -> Dict:
    providers = []
    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        providers = ort.get_available_providers()

    files_present: Dict[str, bool] = {}
    if model_dir:
        base = Path(model_dir)
        for filename in [
            "speaker_encoder.onnx",
            "talker_prefill.onnx",
            "talker_decode.onnx",
            "code_predictor.onnx",
            "vocoder.onnx",
        ]:
            files_present[filename] = (base / filename).exists()
        files_present["embeddings_dir"] = (base / "embeddings").exists()
        files_present["tokenizer_dir"] = (base / "tokenizer").exists()

    return {
        "onnxruntime_installed": bool(importlib.util.find_spec("onnxruntime")),
        "available_providers": providers,
        "model_dir": model_dir,
        "files_present": files_present,
    }


def benchmark_external_onnx_runner(
    text: str,
    ref_audio: str,
    ref_text: str,
    output_dir: Path,
    runner: str,
    model_dir: str,
) -> BenchmarkResult:
    """
    Compare against an external runner.

    The runner contract is:
      python runner.py --model-dir <dir> --ref-audio <wav> --ref-text <txt> --text <txt> --json-out <path>

    The JSON file should contain:
      {
        "first_chunk_ms": ...,
        "total_ms": ...,
        "audio_duration_s": ...,
        "realtime_factor": ...,
        "chunks": ...,
        "sample_rate": ...,
        "output_path": "..."
      }
    """

    label = text if len(text) <= 72 else text[:69] + "..."
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        json_out = handle.name
    try:
        cmd = [
            sys.executable,
            runner,
            "--model-dir",
            model_dir,
            "--ref-audio",
            ref_audio,
            "--ref-text",
            ref_text,
            "--text",
            text,
            "--json-out",
            json_out,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"runner failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip() or 'no output'}"
            )
        payload = json.loads(Path(json_out).read_text())
        return BenchmarkResult(
            backend="onnx",
            label=label,
            success=True,
            first_chunk_ms=payload.get("first_chunk_ms"),
            total_ms=payload.get("total_ms"),
            audio_duration_s=payload.get("audio_duration_s"),
            realtime_factor=payload.get("realtime_factor"),
            chunks=payload.get("chunks"),
            sample_rate=payload.get("sample_rate"),
            output_path=payload.get("output_path"),
            details={"runner": runner, "model_dir": model_dir},
        )
    except Exception as exc:
        return BenchmarkResult(
            backend="onnx",
            label=label,
            success=False,
            error=str(exc),
            details={"runner": runner, "model_dir": model_dir},
        )
    finally:
        try:
            os.unlink(json_out)
        except OSError:
            pass


def slugify(text: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    clean = "-".join(filter(None, clean.split("-")))
    return clean[:80] or "sample"


def print_result(result: BenchmarkResult):
    print(f"\n[{result.backend.upper()}] {result.label}")
    if not result.success:
        print(f"  status: FAIL")
        print(f"  error: {result.error}")
        if result.details:
            print(f"  details: {json.dumps(result.details, ensure_ascii=False)}")
        return
    print("  status: OK")
    print(f"  first_chunk: {format_ms(result.first_chunk_ms)}")
    print(f"  total:       {format_ms(result.total_ms)}")
    print(f"  audio_len:   {result.audio_duration_s:.2f} s")
    print(f"  rtf:         {result.realtime_factor:.3f}x")
    print(f"  chunks:      {result.chunks}")
    print(f"  sample_rate: {result.sample_rate}")
    if result.output_path:
        print(f"  output:      {result.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare MLX Qwen3-TTS vs optional ONNX runner.")
    parser.add_argument("--ref-audio", default="test_ref.wav")
    parser.add_argument("--ref-text", default="参考音频文本")
    parser.add_argument("--text", action="append", dest="texts", help="Text to benchmark. Can be repeated.")
    parser.add_argument("--output-dir", default="bench_outputs")
    parser.add_argument("--mlx-model", default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")
    parser.add_argument("--onnx-model-dir", default="")
    parser.add_argument("--onnx-runner", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    texts = args.texts or DEFAULT_TEXTS
    output_dir = Path(args.output_dir)
    results: List[BenchmarkResult] = []

    print("TTS Benchmark")
    print("=" * 72)
    print(f"Reference audio: {args.ref_audio}")
    print(f"MLX model:       {args.mlx_model}")
    if args.onnx_model_dir:
        print(f"ONNX model dir:  {args.onnx_model_dir}")

    onnx_probe = probe_onnx_environment(args.onnx_model_dir or None)
    print(f"ONNX Runtime providers: {onnx_probe['available_providers']}")

    for text in texts:
        mlx_result = benchmark_mlx(text, args.ref_audio, args.ref_text, output_dir, args.mlx_model)
        results.append(mlx_result)
        print_result(mlx_result)

        if args.onnx_runner and args.onnx_model_dir:
            onnx_result = benchmark_external_onnx_runner(
                text=text,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                output_dir=output_dir,
                runner=args.onnx_runner,
                model_dir=args.onnx_model_dir,
            )
        else:
            onnx_result = BenchmarkResult(
                backend="onnx",
                label=text if len(text) <= 72 else text[:69] + "...",
                success=False,
                error="No ONNX runner configured",
                details=onnx_probe,
            )
        results.append(onnx_result)
        print_result(onnx_result)

    if args.json_out:
        path = Path(args.json_out)
        ensure_parent(path)
        path.write_text(
            json.dumps(
                {
                    "results": [asdict(item) for item in results],
                    "onnx_probe": onnx_probe,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"\nSaved JSON summary to {path}")


if __name__ == "__main__":
    main()

"""
Quick test script to verify each module independently.
Run this before the full demo to catch setup issues early.

Usage:
    python test_modules.py
"""

import time
import sys
import subprocess
import tempfile


def test_asr():
    print("\n" + "=" * 60)
    print("TEST 1: mlx-audio Qwen3-ASR")
    print("=" * 60)
    try:
        from mlx_audio.stt.utils import load_model

        print("Loading model...")
        t0 = time.time()
        model = load_model("mlx-community/Qwen3-ASR-0.6B-bf16", lazy=False)
        print(f"  Load time: {(time.time()-t0)*1000:.0f}ms")

        print("Testing with sample audio...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            normalized = handle.name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                "test_zh.wav",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                normalized,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        t0 = time.time()
        result = model.generate(normalized, language="zh", stream=False, verbose=False)
        text = (getattr(result, "text", "") or "").strip()
        t1 = time.time()
        subprocess.run(["rm", "-f", normalized], check=False)
        print(f"  Inference time: {(t1-t0)*1000:.0f}ms")
        print(f"  Result: '{text}'")
        print("✓ ASR module OK")
        return True
    except Exception as e:
        print(f"✗ ASR FAILED: {e}")
        return False


def test_mt():
    print("\n" + "=" * 60)
    print("TEST 2: Qwen MT via ollama")
    print("=" * 60)
    try:
        test_text = "你好，今天天气很好"
        print(f"  Translating: '{test_text}'")

        t0 = time.time()
        result = subprocess.run(
            ["ollama", "run", "qwen3:1.7b", "/no_think\nTranslate to English. Output ONLY the translation:\n" + test_text],
            capture_output=True,
            text=True,
            timeout=15,
        )
        t1 = time.time()

        print(f"  Result: '{result.stdout.strip()}'")
        print(f"  Latency: {(t1-t0)*1000:.0f}ms")
        print("✓ MT module OK")
        return True
    except FileNotFoundError:
        print("✗ ollama not installed. See: https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        print("✗ ollama timeout - is it running? Run: ollama serve")
        return False
    except Exception as e:
        print(f"✗ MT FAILED: {e}")
        return False


def test_tts():
    print("\n" + "=" * 60)
    print("TEST 3: mlx-audio Qwen3-TTS")
    print("=" * 60)
    try:
        from mlx_audio.tts.utils import load
        import soundfile as sf

        model_id = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
        print(f"  Loading model from: {model_id}")
        t0 = time.time()
        model = load(model_id, lazy=False)
        print(f"  Load time: {(time.time()-t0)*1000:.0f}ms")

        ref_audio = "test_ref.wav"
        ref_text = "参考音频文本"

        test_text = "Hello, this is a test of the simultaneous translation system."

        print(f"  Synthesizing: '{test_text}'")
        t0 = time.time()
        results = list(
            model.generate(
                text=test_text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code="english",
                stream=False,
            )
        )
        t1 = time.time()

        # Save test output
        output_path = "./test_tts_output.wav"
        audio = results[0].audio
        sr = getattr(results[0], "sample_rate", getattr(model, "sample_rate", 24000))
        sf.write(output_path, audio, sr)

        duration = len(audio) / sr
        print(f"  Generated: {duration:.1f}s audio")
        print(f"  Latency: {(t1-t0)*1000:.0f}ms")
        print(f"  Saved to: {output_path}")
        print("✓ TTS module OK")
        return True
    except Exception as e:
        print(f"✗ TTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Module Test Suite - AI Simultaneous Translation")
    print("=" * 60)

    results = {}
    results["ASR"] = test_asr()
    results["MT"] = test_mt()
    results["TTS"] = test_tts()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll modules ready! Run the demo:")
        print("  python demo.py --live          # Live microphone mode")
        print("  python demo.py your_audio.wav   # File mode")
    else:
        print("\nSome modules failed. Fix the issues above before running the demo.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

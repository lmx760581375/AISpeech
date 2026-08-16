"""
Download model weights for the simultaneous translation demo.

Models:
1. mlx-audio Qwen3-ASR model: auto-downloads MLX weights on first use
2. mlx-audio Qwen3-TTS model: auto-downloads MLX weights on first use
3. MT model via ollama (`qwen3:1.7b`)
"""

import os
import sys


def check_ollama():
    """Check if ollama is installed and has the translation model."""
    print("=" * 60)
    print("Checking ollama for MT (Qwen3-1.7B)...")
    print("=" * 60)

    try:
        import subprocess

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "qwen3" in result.stdout.lower():
            print("✓ Qwen3 model found in ollama")
        else:
            print("✗ Qwen3 model not found in ollama")
            print("  Run: ollama pull qwen3:1.7b")
    except Exception as e:
        print(f"✗ Failed to check ollama: {e}")
        print("  Install from: https://ollama.ai")
        print("  Then run: ollama pull qwen3:1.7b")


def download_mlx_qwen3_asr():
    """Download mlx-audio Qwen3-ASR weights by loading the model once."""
    print("\n" + "=" * 60)
    print("Downloading mlx-audio Qwen3-ASR model...")
    print("=" * 60)

    try:
        from mlx_audio.stt.utils import load_model

        model_id = "mlx-community/Qwen3-ASR-0.6B-bf16"
        print(f"Loading {model_id} (will download MLX weights on first run)...")
        model = load_model(model_id, lazy=False)
        del model
        print(f"✓ Downloaded {model_id}")
    except Exception as e:
        print(f"✗ Failed to download mlx-audio Qwen3-ASR: {e}")
        print(
            "\n  Manual download alternative:\n"
            "    python -c \"from mlx_audio.stt.utils import load_model; "
            "load_model('mlx-community/Qwen3-ASR-0.6B-bf16', lazy=False)\""
        )


def download_mlx_qwen3_tts():
    """Download mlx-audio Qwen3-TTS weights by loading the model once."""
    print("\n" + "=" * 60)
    print("Downloading mlx-audio Qwen3-TTS model...")
    print("=" * 60)

    model_id = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
    try:
        from mlx_audio.tts.utils import load

        print(f"Loading {model_id} (will download MLX weights on first run)...")
        model = load(model_id, lazy=False)
        del model
        print(f"✓ Downloaded {model_id}")
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        print("\n  Manual download alternative:")
        print(f"    python -c \"from mlx_audio.tts.utils import load; load('{model_id}', lazy=False)\"")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("AI Simultaneous Translation - Model Download")
    print("=" * 60)
    print()

    # Step 1: Check ollama
    check_ollama()

    # Step 2: Download mlx-audio Qwen3-ASR
    download_mlx_qwen3_asr()

    # Step 3: Download mlx-audio Qwen3-TTS
    download_mlx_qwen3_tts()

    print("\n" + "=" * 60)
    print("Download complete! Run the demo with:")
    print("  python demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

# AI Simultaneous Translation Demo

Chinese speech -> English speech with voice cloning.

This repo is now tuned for Apple Silicon end to end:

```text
Mic / WAV
  -> mlx-audio Qwen3-ASR
  -> clause-level MT
  -> mlx-audio Qwen3-TTS voice clone
  -> speaker
```

## What Exists Now

- `demo.py`
  - file mode
  - live microphone mode
  - segmented "streaming-like" path for lower first-audio latency
  - ASR uses `mlx-audio` with `Qwen3-ASR`
  - TTS uses `mlx-audio` with `Qwen3-TTS`
  - MT backend switch: `ollama` or OpenAI-compatible `vLLM`
- `web_demo.py`
  - browser demo page
  - startup warmup status
  - microphone recording for source audio and reference audio
- `download_models.py`
  - checks ollama
  - downloads MLX Qwen3-ASR and MLX Qwen3-TTS weights
- `test_modules.py`
  - smoke tests for ASR / MT / TTS

## Quick Start

```bash
conda create -n speech-trans python=3.11 -y
conda activate speech-trans
pip install -r requirements.txt
python download_models.py
ollama pull qwen3:1.7b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python web_demo.py \
  --ref-audio test_ref.wav --ref-text "参考音频文本" --eager-warmup
```

Then open the local page shown in the terminal.

`download_models.py` is the only step that downloads Hugging Face weights. The web and CLI demos resolve ASR/TTS weights from the local cache and fail fast with a setup message if a model is missing, instead of blocking on a network request. The Ollama Qwen3 integration explicitly disables thinking so its token budget is reserved for the translation.

For the continuous microphone demo, use the realtime entrypoint. `--eager-warmup` moves model startup out of the first session, and the default 2200 ms window / 1000 ms hop favors stable Chinese ASR context:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python realtime_web_demo.py \
  --ref-audio test_ref.wav --ref-text "参考音频文本" --eager-warmup
```

Open `http://127.0.0.1:7870/realtime`.

## Default Models

- ASR: `mlx-community/Qwen3-ASR-0.6B-bf16`
- TTS: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit`
- MT: `qwen3:1.7b` via `ollama`

The default TTS model is the smaller `0.6B` MLX `4bit` port because it is currently the best low-latency fit for local Apple Silicon voice cloning. Use `--tts-model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` when prioritizing a small quality margin over speed.
Voice cloning uses the reference audio and its transcript as Qwen3-TTS ICL conditioning.

## CLI Example

```bash
python demo.py test_zh.wav \
  --streaming \
  --asr-model mlx-community/Qwen3-ASR-0.6B-bf16 \
  --ref-audio test_ref.wav \
  --ref-text "参考音频文本"
```

## vLLM Option For MT

If you want lower MT latency than `ollama`, run an OpenAI-compatible vLLM server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
python demo.py test_zh.wav \
  --streaming \
  --mt-backend openai \
  --mt-model Qwen/Qwen2.5-1.5B-Instruct \
  --mt-base-url http://127.0.0.1:8000/v1 \
  --ref-audio test_ref.wav \
  --ref-text "参考音频文本"
```

## Why This Stack

- `mlx-audio Qwen3-ASR` keeps the ASR path on Apple Silicon MLX and fixes the bad Chinese transcription quality seen with the previous Whisper setup
- `mlx-audio` avoids the `qwen-tts + PyTorch MPS` kernel limitation that caused TTS failures on Apple Silicon
- `Qwen3-TTS 0.6B Base` keeps voice cloning available while cutting local latency compared with the larger 1.7B model

## Reality Check On <2s

`ASR + MT + TTS < 2s` is realistic only if you optimize for first audio chunk, not full-sentence audio completion.

With the current Apple Silicon-focused stack, ASR is no longer the main risk. TTS is still the dominant latency component.

Practical target:

| Stage | Target |
| --- | --- |
| ASR partial text ready | 200-500ms |
| MT for one clause | 100-300ms |
| TTS first audio chunk | 400-1000ms |
| First translated audio heard | 0.9-1.8s |

If you need lower than this consistently, the next thing to optimize is the TTS model and chunking strategy, not ASR.

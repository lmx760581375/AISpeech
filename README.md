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
  --ref-audio test_ref.wav --ref-text "参考音频中实际说出的文本" --eager-warmup
```

Then open the local page shown in the terminal.

`download_models.py` is the only step that downloads Hugging Face weights. The web and CLI demos resolve ASR/TTS weights from the local cache and fail fast with a setup message if a model is missing, instead of blocking on a network request. The Ollama Qwen3 integration explicitly disables thinking so its token budget is reserved for the translation.

For the continuous microphone demo, use the realtime entrypoint. `--eager-warmup` moves model startup out of the first session. The default `burst` mode uses WebRTC VAD to seal a short speech burst after a pause, so ASR partials from the same burst replace each other instead of being concatenated. Use `--asr-segmentation fixed` only to retain the older 2200 ms window / 1000 ms hop behavior:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python realtime_web_demo.py \
  --ref-audio test_ref.wav --ref-text "参考音频中实际说出的文本" --eager-warmup
```

Open `http://127.0.0.1:7870/realtime`.

For lower TTS latency on Apple Silicon, Pocket TTS MLX can be selected for the
realtime worker while leaving Qwen as the default fallback:

```bash
conda run -n test python realtime_web_demo.py \
  --tts-backend pocket-mlx \
  --ref-audio test_ref.wav --ref-text "参考音频中实际说出的文本" --eager-warmup
```

Pocket TTS uses the reference waveform for cloning and does not use
`--ref-text`. Before the first use, accept the terms at
https://huggingface.co/kyutai/pocket-tts and authenticate a token with gated
repository access using `hf auth login --force`. The worker keeps both the
model and the encoded voice state in memory; a changed reference audio is
encoded once before synthesis resumes.

## Realtime Regression Loop

`realtime_regression.py` drives the running realtime API with a fixed-seed,
randomly jittered overlap schedule. It reports Chinese text coverage, per-stage
P50/P95 latency, queue coalescing, output lag, and basic waveform health. On
macOS the synthetic fixture creates a known long Chinese speech sample, so it
can be used for repeatable before/after comparisons:

```bash
conda run -n test python realtime_regression.py \
  --synthetic-fixture --pace --json-out /tmp/realtime-regression.json
```

Use a real reference recording for final listening checks. The automated report
detects content loss and waveform anomalies, but it cannot prove perceived
voice similarity or naturalness.

## Default Models

- ASR: `mlx-community/Qwen3-ASR-0.6B-bf16`
- TTS: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit`
- MT: `qwen3:1.7b` via `ollama`

The default TTS model is the smaller `0.6B` MLX `4bit` port because it is currently the best low-latency fit for local Apple Silicon voice cloning. Use `--tts-model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` when prioritizing a small quality margin over speed.
Voice cloning uses the reference audio and its transcript as Qwen3-TTS ICL conditioning.

## Optional Qwen3.5 MLX MT

`qwen35-mlx` is an optional low-latency backend for conservative ASR-residual translation. It is not the default: the existing Ollama backend remains the supported baseline.

Install the optional runtime without allowing it to upgrade this project's shared web dependencies:

```bash
pip install --no-deps "mlx-vlm==0.6.15"
hf download mlx-community/Qwen3.5-2B-4bit --local-dir models/mt_bench/qwen35-2b-4bit
```

Run the realtime server with the local model path:

```bash
python realtime_web_demo.py \
  --mt-backend qwen35-mlx \
  --mt-model models/mt_bench/qwen35-2b-4bit \
  --tts-backend pocket-mlx \
  --ref-audio test_ref.wav \
  --ref-text "参考音频实际说出的文本"
```

The backend is serialized with ASR and TTS through the existing MLX lock. This prevents Metal contention, so its standalone MT latency will be lower than its end-to-end realtime latency.

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

## MLX TTS Profiling

Profile the warmed voice-clone path before changing the decoder. The command reports median first-audio latency, total generation time, RTF, audio chunk count, and MLX peak memory for several streaming intervals. It can also save a Python call profile and JSON summary for before/after comparisons.

```bash
conda run -n test python profile_mlx_tts.py \
  --ref-audio test_ref.wav \
  --ref-text "参考音频文本" \
  --profile-out /tmp/qwen3tts.prof \
  --json-out /tmp/qwen3tts.json
```

Use `--temperature 0` for repeatable performance comparisons. Lower streaming intervals reduce first-audio latency but increase chunking overhead; the realtime server's default is tuned near the middle of that tradeoff.

`experiment_compiled_code_predictor.py` is a separate MLX experiment for compiling Qwen's fixed 15-codebook prediction chain. It is not used by the web server. With `--temperature 0`, it checks waveform equality against the supported `mlx-audio` path before reporting median latency across several runs:

```bash
conda run -n test python experiment_compiled_code_predictor.py --runs 3
```

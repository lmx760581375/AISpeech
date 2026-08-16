"""Validate a compiled Qwen3-TTS code-predictor step against mlx-audio.

This intentionally remains an experiment. It reproduces the ICL voice-clone
decode loop with only the fixed 15-group code-predictor chain compiled, then
compares its waveform and latency with the supported mlx-audio path.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from demo import DEFAULT_TTS_MODEL, TTSModule


@dataclass
class SynthesisResult:
    waveform: np.ndarray
    sample_rate: int
    first_chunk_ms: float
    total_ms: float
    chunks: int
    token_count: int


class CompiledCodePredictorIcl:
    """ICL decoder that compiles only the fixed 15-codebook inner chain."""

    def __init__(self, tts: TTSModule, temperature: float, top_k: int, top_p: float):
        self.tts = tts
        self.model = tts.model
        self.config = self.model.config.talker_config
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.code_cache = self.model.talker.code_predictor.make_cache()

        def code_predictor_step(code_hidden: mx.array, first_token: mx.array) -> mx.array:
            code_tokens = [first_token]
            code_input = mx.concatenate(
                [code_hidden, self.model.talker.get_input_embeddings()(first_token)], axis=1
            )
            for code_idx in range(self.config.num_code_groups - 1):
                code_logits, _, _ = self.model.talker.code_predictor(
                    code_input,
                    cache=self.code_cache,
                    generation_step=code_idx,
                )
                next_code = self.model._sample_token(
                    code_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                code_tokens.append(next_code)
                if code_idx + 1 < self.config.num_code_groups - 1:
                    code_input = self.model.talker.code_predictor.codec_embedding[code_idx](next_code)
            return mx.concatenate(code_tokens, axis=1)

        self._code_predictor_step = mx.compile(code_predictor_step)

    def _reset_code_cache(self):
        # Match mlx-audio's per-frame code-predictor cache lifecycle.
        for cache in self.code_cache:
            cache.keys = None
            cache.values = None
            cache.offset = 0

    def synthesize(self, text: str, streaming_interval_s: float, max_tokens: int) -> SynthesisResult:
        with self.tts._generation_session() as (ref_audio, ref_text):
            input_embeds, trailing_text_hidden, tts_pad_embed, _ = (
                self.model._prepare_icl_generation_inputs(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    language="english",
                )
            )

            cache = self.model.talker.make_cache()
            generated_codes: List[mx.array] = []
            generated_token_ids: List[int] = []
            trailing_idx = 0
            eos_token_id = self.config.codec_eos_token_id
            suppress_tokens = [
                token_id
                for token_id in range(self.config.vocab_size - 1024, self.config.vocab_size)
                if token_id != eos_token_id
            ]
            streaming_chunk_size = max(1, int(streaming_interval_s * 12.5))
            decoded_tokens = 0
            chunks: List[np.ndarray] = []
            first_chunk_ms = None
            started_at = time.perf_counter()

            self.model.speech_tokenizer.decoder.reset_streaming_state()
            try:
                for _ in range(max_tokens):
                    logits, hidden = self.model.talker(input_embeds, cache=cache)
                    next_token = self.model._sample_token(
                        logits,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        repetition_penalty=1.5,
                        generated_tokens=generated_token_ids or None,
                        suppress_tokens=suppress_tokens,
                        eos_token_id=eos_token_id,
                    )
                    is_eos = next_token[0, 0] == eos_token_id

                    self._reset_code_cache()
                    all_codes = self._code_predictor_step(hidden[:, -1:, :], next_token)

                    if trailing_idx < trailing_text_hidden.shape[1]:
                        text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                        trailing_idx += 1
                    else:
                        text_embed = tts_pad_embed

                    codec_embed = self.model.talker.get_input_embeddings()(next_token)
                    for code_idx in range(self.config.num_code_groups - 1):
                        codec_embed = codec_embed + self.model.talker.code_predictor.codec_embedding[
                            code_idx
                        ](all_codes[:, code_idx + 1 : code_idx + 2])
                    input_embeds = text_embed + codec_embed
                    mx.eval(input_embeds, all_codes, is_eos)

                    if is_eos.item():
                        break

                    generated_token_ids.append(int(next_token[0, 0]))
                    generated_codes.append(all_codes)

                    if len(generated_codes) - decoded_tokens >= streaming_chunk_size:
                        chunk = self._decode_new_codes(generated_codes, decoded_tokens)
                        decoded_tokens = len(generated_codes)
                        chunks.append(chunk)
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started_at) * 1000

                if len(generated_codes) > decoded_tokens:
                    chunk = self._decode_new_codes(generated_codes, decoded_tokens)
                    chunks.append(chunk)
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - started_at) * 1000
            finally:
                self.model.speech_tokenizer.decoder.reset_streaming_state()

        if not chunks or first_chunk_ms is None:
            raise RuntimeError("compiled code-predictor path produced no audio")

        return SynthesisResult(
            waveform=np.concatenate(chunks).astype(np.float32, copy=False),
            sample_rate=self.tts.sample_rate,
            first_chunk_ms=first_chunk_ms,
            total_ms=(time.perf_counter() - started_at) * 1000,
            chunks=len(chunks),
            token_count=len(generated_codes),
        )

    def _decode_new_codes(self, generated_codes: List[mx.array], start: int) -> np.ndarray:
        codes = mx.stack(generated_codes[start:], axis=1)
        codes = mx.transpose(codes, (0, 2, 1))
        waveform = self.model.speech_tokenizer.decoder.streaming_step(codes).squeeze(1)[0]
        mx.eval(waveform)
        return np.asarray(waveform).astype(np.float32, copy=False)


def baseline_synthesize(tts: TTSModule, text: str, options: Dict) -> SynthesisResult:
    started_at = time.perf_counter()
    chunks = []
    first_chunk_ms = None
    sample_rate = None
    for waveform, sample_rate, _ in tts.synthesize_stream(text, generation_options=options):
        chunks.append(waveform)
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - started_at) * 1000
    if not chunks or sample_rate is None or first_chunk_ms is None:
        raise RuntimeError("baseline produced no audio")
    return SynthesisResult(
        waveform=np.concatenate(chunks).astype(np.float32, copy=False),
        sample_rate=sample_rate,
        first_chunk_ms=first_chunk_ms,
        total_ms=(time.perf_counter() - started_at) * 1000,
        chunks=len(chunks),
        token_count=0,
    )


def print_result(label: str, result: SynthesisResult):
    duration_s = len(result.waveform) / result.sample_rate
    rtf = result.total_ms / 1000 / max(duration_s, 1e-6)
    print(
        f"{label}: first={result.first_chunk_ms:.0f}ms | total={result.total_ms:.0f}ms | "
        f"audio={duration_s:.2f}s | RTF={rtf:.3f}x | chunks={result.chunks} | "
        f"tokens={result.token_count}"
    )


def median_result(results: List[SynthesisResult]) -> SynthesisResult:
    if not results:
        raise ValueError("results cannot be empty")
    middle = results[len(results) // 2]
    return SynthesisResult(
        waveform=middle.waveform,
        sample_rate=middle.sample_rate,
        first_chunk_ms=statistics.median(item.first_chunk_ms for item in results),
        total_ms=statistics.median(item.total_ms for item in results),
        chunks=round(statistics.median(item.chunks for item in results)),
        token_count=round(statistics.median(item.token_count for item in results)),
    )


def main():
    parser = argparse.ArgumentParser(description="Experiment with compiled Qwen3-TTS code prediction")
    parser.add_argument("--model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--ref-audio", default="test_ref.wav")
    parser.add_argument("--ref-text", default="参考音频文本")
    parser.add_argument("--text", default="This is a short latency benchmark.")
    parser.add_argument("--interval", type=float, default=0.32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs after warmup")
    args = parser.parse_args()

    if args.interval <= 0 or args.max_tokens < 1 or args.runs < 1:
        parser.error("--interval, --max-tokens, and --runs must be positive")

    options = {"temperature": args.temperature, "top_p": 1.0, "repetition_penalty": 1.5}
    tts = TTSModule(args.model)
    tts.create_voice_reference(args.ref_audio, args.ref_text)
    tts.streaming_interval_s = args.interval

    # Warm the supported baseline, then compile and warm the experiment separately.
    baseline_synthesize(tts, "Hello.", options)
    compiled = CompiledCodePredictorIcl(tts, args.temperature, top_k=50, top_p=1.0)
    compiled.synthesize("Hello.", args.interval, args.max_tokens)
    baseline_runs = []
    compiled_runs = []
    for _ in range(args.runs):
        baseline_runs.append(baseline_synthesize(tts, args.text, options))
        compiled_runs.append(compiled.synthesize(args.text, args.interval, args.max_tokens))

    baseline = median_result(baseline_runs)
    optimized = median_result(compiled_runs)

    print_result("baseline", baseline)
    print_result("compiled", optimized)
    print(f"same_sample_rate={baseline.sample_rate == optimized.sample_rate}")
    print(f"same_shape={baseline.waveform.shape == optimized.waveform.shape}")
    if baseline.waveform.shape == optimized.waveform.shape:
        print(f"max_abs_diff={np.max(np.abs(baseline.waveform - optimized.waveform)):.8f}")


if __name__ == "__main__":
    main()

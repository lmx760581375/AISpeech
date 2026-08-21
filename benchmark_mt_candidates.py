"""Offline latency and output sanity benchmark for realtime MT candidates.

This intentionally does not import the realtime server or write to its state.
It measures warmed inference only, because model-download and model-load time do
not affect a continuously running realtime session.

Examples:
  conda run -n test python benchmark_mt_candidates.py --candidate ollama
  conda run -n test python benchmark_mt_candidates.py --candidate qwen35 \\
    --qwen35-model models/mt_bench/qwen35-2b-4bit
  conda run -n test python benchmark_mt_candidates.py --candidate translategemma \\
    --translategemma-model /path/to/translategemma-4b-it-4bit
  conda run -n test python benchmark_mt_candidates.py --candidate nllb \\
    --nllb-model /path/to/nllb-200-distilled-600M-ct2-int8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    expected_concepts: Sequence[str]


# Samples mirror the failure modes observed in realtime ASR: repeated starts,
# incomplete clauses, recognition substitutions, and context-dependent tails.
CASES = (
    Case("repeated_start", "我我要测一下我", ("test",)),
    Case("incomplete_tail", "我这个软件，你这一", ("software",)),
    Case("spoken_question", "他不应该是直接把店铺搜到吗", ("shouldn't", "store")),
    Case("recognition_noise", "嗯，他怎么是？点了一个账户搜索链接呀", ("clicked", "account", "search", "link")),
    Case("unfinished_statement", "你听到我说话了没？好的。这个店铺里面是", ("heard", "store")),
    Case("normal_control", "我们先把最重要的问题解决，再慢慢讨论那些细节。", ("important", "problem", "details")),
)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q / 100
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def prompt_messages(source: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a simultaneous interpreter. Correct obvious Chinese ASR repetition or "
                "word-boundary errors only when the intended meaning is clear, then translate into "
                "short natural spoken English. Preserve uncertainty in incomplete source text. "
                "Return only the English translation."
            ),
        },
        {"role": "user", "content": f"<source>{source}</source>"},
    ]


def output_sanity(text: str, expected_concepts: Sequence[str]) -> dict:
    normalized = " ".join(text.lower().split())
    concepts = {concept: concept in normalized for concept in expected_concepts}
    return {
        "empty": not bool(normalized),
        "word_count": len(normalized.split()),
        "has_explanation": any(marker in normalized for marker in ("the source", "translation:", "as an ai")),
        "concept_coverage": round(sum(concepts.values()) / len(concepts), 2) if concepts else 1.0,
        "concepts": concepts,
    }


class OllamaCandidate:
    def __init__(self, model: str):
        import ollama

        self.client = ollama
        self.model = model
        self.name = f"ollama-{model.replace(':', '-')}"

    def translate(self, source: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=prompt_messages(source),
            options={"temperature": 0, "num_predict": 64, "num_ctx": 512},
            think=False,
            keep_alive=-1,
        )
        return response["message"]["content"].strip()


class TranslateGemmaCandidate:
    name = "translategemma-4b-mlx-4bit"

    def __init__(self, model: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model, lazy=False)
        # The converted tokenizer exposes only ``<eos>`` as EOS, while the
        # official translation template terminates output with this token.
        self.tokenizer.eos_token_ids.update(
            self.tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        )

    def translate(self, source: str) -> str:
        from mlx_lm import generate
        from mlx_lm.generate import make_sampler

        # TranslateGemma deliberately has a structured, translation-only prompt.
        # Do not embed correction instructions in source text: that would make the
        # comparison invalid and could be translated as user content.
        prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "zh-Hans",
                            "target_lang_code": "en",
                            "text": source,
                        }
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=48,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        ).strip()


class Qwen35Candidate:
    name = "qwen3.5-2b-mlx-4bit"

    def __init__(self, model: str):
        from mlx_vlm import load

        self.model, self.processor = load(model, lazy=False)

    def translate(self, source: str) -> str:
        from mlx_vlm import generate

        prompt = self.processor.apply_chat_template(
            prompt_messages(source), tokenize=False, add_generation_prompt=True
        )
        result = generate(
            self.model,
            self.processor,
            prompt,
            max_tokens=64,
            temperature=0.0,
            verbose=False,
        )
        return result.text.strip()


class NLLBCandidate:
    name = "nllb-200-distilled-600m-ct2-int8"

    def __init__(self, model: str):
        import ctranslate2
        from transformers import AutoTokenizer

        self.translator = ctranslate2.Translator(model, device="auto", compute_type="int8")
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.tokenizer.src_lang = "zho_Hans"

    def translate(self, source: str) -> str:
        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer(source)["input_ids"])
        result = self.translator.translate_batch(
            [source_tokens], target_prefix=[["eng_Latn"]], beam_size=1, max_decoding_length=64
        )[0]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(result.hypotheses[0]), skip_special_tokens=True).strip()


def run_candidate(candidate, cases: Sequence[Case], repeats: int) -> dict:
    # Run an unreported warmup. This also makes missing model/config errors fail early.
    candidate.translate(cases[0].source)
    rows = []
    for case in cases:
        timings, outputs = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            output = candidate.translate(case.source)
            timings.append((time.perf_counter() - started) * 1000)
            outputs.append(output)
        selected = outputs[-1]
        rows.append(
            {
                "case": asdict(case),
                "output": selected,
                "latency_ms": {"p50": round(percentile(timings, 50), 1), "p95": round(percentile(timings, 95), 1)},
                "stable_output": len(set(outputs)) == 1,
                "sanity": output_sanity(selected, case.expected_concepts),
            }
        )
    all_p50 = [row["latency_ms"]["p50"] for row in rows]
    return {
        "candidate": candidate.name,
        "repeats": repeats,
        "summary": {
            "latency_ms": {"p50": round(percentile(all_p50, 50), 1), "p95": round(percentile(all_p50, 95), 1)},
            "mean_concept_coverage": round(statistics.mean(row["sanity"]["concept_coverage"] for row in rows), 3),
            "bad_output_count": sum(row["sanity"]["empty"] or row["sanity"]["has_explanation"] for row in rows),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("ollama", "qwen35", "translategemma", "nllb"), action="append", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ollama-model", default="qwen3:1.7b")
    parser.add_argument("--qwen35-model")
    parser.add_argument("--translategemma-model")
    parser.add_argument("--nllb-model")
    parser.add_argument("--json-out", type=Path, default=Path("bench_outputs/mt_candidates.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    factories: dict[str, Callable[[], object]] = {
        "ollama": lambda: OllamaCandidate(args.ollama_model),
        "qwen35": lambda: Qwen35Candidate(args.qwen35_model or _required("--qwen35-model")),
        "translategemma": lambda: TranslateGemmaCandidate(args.translategemma_model or _required("--translategemma-model")),
        "nllb": lambda: NLLBCandidate(args.nllb_model or _required("--nllb-model")),
    }
    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": []}
    for name in args.candidate:
        print(f"[benchmark] loading {name}", flush=True)
        result = run_candidate(factories[name](), CASES, args.repeats)
        results["results"].append(result)
        print(f"[benchmark] {result['candidate']}: P50={result['summary']['latency_ms']['p50']} ms", flush=True)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {args.json_out}")
    return 0


def _required(flag: str) -> str:
    raise ValueError(f"{flag} is required for this candidate")


if __name__ == "__main__":
    raise SystemExit(main())

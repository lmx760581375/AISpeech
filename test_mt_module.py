"""Unit tests for optional MT backends without loading model weights."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from demo import MTModule, build_parser
from realtime_web_demo import build_parser as build_realtime_parser


class Qwen35MLXTests(unittest.TestCase):
    def test_qwen35_default_and_cli_choices(self):
        self.assertEqual(MTModule._default_model_for_backend("qwen35-mlx"), "mlx-community/Qwen3.5-2B-4bit")
        self.assertEqual(build_parser().parse_args(["--mt-backend", "qwen35-mlx"]).mt_backend, "qwen35-mlx")
        self.assertEqual(
            build_realtime_parser().parse_args(["--mt-backend", "qwen35-mlx"]).mt_backend,
            "qwen35-mlx",
        )

    def test_qwen35_uses_chat_template_and_cleans_output(self):
        model = object()
        processor = SimpleNamespace(apply_chat_template=Mock(return_value="rendered-prompt"))
        mlx_vlm = types.ModuleType("mlx_vlm")
        mlx_vlm.load = Mock(return_value=(model, processor))
        mlx_vlm.generate = Mock(return_value=SimpleNamespace(text="<think>internal</think>It is ready."))

        with patch.dict(sys.modules, {"mlx_vlm": mlx_vlm}), patch("demo.resolve_local_model", return_value="local-model"):
            module = MTModule(backend="qwen35-mlx", model="local-model")
            result = module.translate("准备好了", [("上一句", "Previous sentence.")])

        self.assertEqual(result, "It is ready.")
        processor.apply_chat_template.assert_called_once()
        messages = processor.apply_chat_template.call_args.args[0]
        self.assertIn("<context>", messages[1]["content"])
        self.assertIn("never translate, repeat, or quote it", messages[0]["content"])
        mlx_vlm.generate.assert_called_once_with(
            model,
            processor,
            "rendered-prompt",
            max_tokens=48,
            temperature=0.0,
            verbose=False,
        )

    def test_ollama_retries_without_context_when_it_echoes_history(self):
        ollama = types.ModuleType("ollama")
        ollama.list = Mock()
        ollama.chat = Mock(
            side_effect=[
                {"message": {"content": '{"translation":"Old translation."}'}},
                {"message": {"content": '{"translation":"Fresh translation."}'}},
            ]
        )
        with patch.dict(sys.modules, {"ollama": ollama}):
            module = MTModule(backend="ollama")
            result = module.translate("新的短句。", [("上一句。", "Old translation.")])

        self.assertEqual(result, "Fresh translation.")
        self.assertEqual(ollama.chat.call_count, 2)
        self.assertNotIn("Previous committed context", ollama.chat.call_args_list[1].kwargs["messages"][1]["content"])

    def test_ollama_keeps_context_for_non_echo_translation(self):
        ollama = types.ModuleType("ollama")
        ollama.list = Mock()
        ollama.chat = Mock(return_value={"message": {"content": '{"translation":"Fresh translation."}'}})
        with patch.dict(sys.modules, {"ollama": ollama}):
            module = MTModule(backend="ollama")
            result = module.translate("新的短句。", [("上一句。", "Old translation.")])

        self.assertEqual(result, "Fresh translation.")
        ollama.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()

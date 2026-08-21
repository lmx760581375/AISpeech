"""Fast regression tests for realtime commit logic without loading speech models."""

import unittest

import numpy as np

from realtime_web_demo import BurstTranscriptBuffer, RealtimeSentenceAccumulator, RealtimeSession


class RealtimeLogicTests(unittest.TestCase):
    def test_burst_partial_replaces_instead_of_concatenating(self):
        buffer = BurstTranscriptBuffer()
        buffer.update(3, "我想测试")
        buffer.update(3, "我想测试这个实时翻译。")
        self.assertEqual(buffer.seal(3), "我想测试这个实时翻译。")

    def test_short_question_is_emitted(self):
        accumulator = RealtimeSentenceAccumulator()
        self.assertEqual(accumulator.feed("现在开始了吗？", 120), [("现在开始了吗？", 120)])

    def test_forced_filler_is_not_emitted(self):
        accumulator = RealtimeSentenceAccumulator()
        self.assertEqual(accumulator.feed("嗯。", 120, force=True), [])

    def test_trailing_silence_measurement(self):
        sample_rate = 16000
        samples = np.concatenate(
            (
                np.full(int(sample_rate * 1.2), 0.02, dtype=np.float32),
                np.zeros(int(sample_rate * 0.8), dtype=np.float32),
            )
        )
        self.assertEqual(RealtimeSession._trailing_silence_ms(samples, sample_rate), 800.0)


if __name__ == "__main__":
    unittest.main()

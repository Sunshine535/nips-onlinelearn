"""Unit tests for core SPM components (CPU-only, no model downloads)."""

import os
import pickle
import tempfile

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.streaming_memory import (
    ReservoirBuffer,
    SessionBuffer,
    WorkingMemoryLoRA,
    LongTermMemoryLoRA,
)


class TestReservoirBuffer:
    def test_add_within_capacity(self):
        buf = ReservoirBuffer(max_size=10)
        for i in range(5):
            buf.add(torch.randn(1, 8), torch.randn(1, 8))
        assert len(buf) == 5
        assert buf.total_seen == 5

    def test_reservoir_sampling_overflow(self):
        buf = ReservoirBuffer(max_size=5)
        for i in range(100):
            buf.add(torch.randn(1, 4), torch.randn(1, 4))
        assert len(buf) == 5
        assert buf.total_seen == 100

    def test_sample_returns_correct_count(self):
        buf = ReservoirBuffer(max_size=20)
        for i in range(20):
            buf.add(torch.randn(1, 4), torch.randn(1, 4))
        inputs, labels = buf.sample(5)
        assert len(inputs) == 5
        assert len(labels) == 5

    def test_sample_empty_buffer(self):
        buf = ReservoirBuffer(max_size=10)
        inputs, labels = buf.sample(5)
        assert inputs == []
        assert labels == []

    def test_sample_more_than_available(self):
        buf = ReservoirBuffer(max_size=10)
        for i in range(3):
            buf.add(torch.randn(1, 4), torch.randn(1, 4))
        inputs, labels = buf.sample(10)
        assert len(inputs) == 3

    def test_merge_session(self):
        buf = ReservoirBuffer(max_size=50)
        sess = SessionBuffer()
        for i in range(5):
            sess.add(torch.randn(1, 4), torch.randn(1, 4))
        buf.merge_session(sess)
        assert len(buf) == 5

    def test_serialization(self):
        buf = ReservoirBuffer(max_size=10)
        for i in range(7):
            buf.add(torch.randn(1, 4), torch.randn(1, 4))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(buf, f)
            path = f.name
        try:
            with open(path, "rb") as f:
                restored = pickle.load(f)
            assert len(restored) == 7
            assert restored.total_seen == 7
        finally:
            os.unlink(path)


class TestSessionBuffer:
    def test_add_and_get(self):
        buf = SessionBuffer()
        buf.add(torch.randn(1, 8), torch.randn(1, 8))
        buf.add(torch.randn(1, 8), torch.randn(1, 8))
        inputs, labels = buf.get_all()
        assert len(inputs) == 2
        assert len(labels) == 2

    def test_clear(self):
        buf = SessionBuffer()
        for _ in range(5):
            buf.add(torch.randn(1, 4), torch.randn(1, 4))
        assert len(buf) == 5
        buf.clear()
        assert len(buf) == 0

    def test_cpu_storage(self):
        buf = SessionBuffer()
        t = torch.randn(1, 4)
        buf.add(t, t.clone())
        inputs, labels = buf.get_all()
        assert inputs[0]["input_ids"].device == torch.device("cpu")


class TestWorkingMemoryLoRA:
    def test_init_and_config(self):
        cfg = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "learning_rate": 1e-3,
            "online_update_steps": 2,
        }
        wm = WorkingMemoryLoRA(cfg)
        assert wm.lora_config.r == 8
        assert wm.lora_config.lora_alpha == 16
        assert wm.update_count == 0

    def test_needs_consolidation(self):
        cfg = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
            "max_turns_before_consolidation": 3,
        }
        wm = WorkingMemoryLoRA(cfg)
        assert not wm.needs_consolidation()
        wm.update_count = 3
        assert wm.needs_consolidation()

    def test_reset(self):
        cfg = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
        }
        wm = WorkingMemoryLoRA(cfg)
        wm.update_count = 5
        wm.reset()
        assert wm.update_count == 0
        assert wm._optimizer is None


class TestLongTermMemoryLoRA:
    def test_init(self):
        cfg = {
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.01,
            "target_modules": ["q_proj", "v_proj"],
        }
        lt = LongTermMemoryLoRA(cfg)
        assert lt.lora_config.r == 32
        assert lt.fisher == {}
        assert lt.prev_params == {}


class TestFormatTurn:
    def test_import_format_turn(self):
        try:
            from scripts.train_spm import format_turn
        except ImportError:
            pytest.skip("datasets not installed")
        result = format_turn("Hello", "Hi there", persona="I like dogs")
        assert "<|im_start|>system" in result
        assert "I like dogs" in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_format_turn_no_persona(self):
        try:
            from scripts.train_spm import format_turn
        except ImportError:
            pytest.skip("datasets not installed")
        result = format_turn("Hello", "Hi")
        assert "helpful" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

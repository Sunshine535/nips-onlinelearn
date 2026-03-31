import pytest
import torch
import os
import json

def test_checkpoint_roundtrip(tmp_path):
    """Test that saving and loading checkpoint preserves state."""
    state = {
        "epoch": 5,
        "session": 42,
        "model_state": {"weight": torch.randn(10, 10)},
        "optimizer_state": {"lr": 0.001},
        "rng_state": torch.random.get_rng_state(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["epoch"] == 5
    assert loaded["session"] == 42
    assert torch.equal(loaded["model_state"]["weight"], state["model_state"]["weight"])
    assert torch.equal(loaded["rng_state"], state["rng_state"])

def test_persona_session_grouping():
    """Test that sessions maintain consistent persona identity."""
    conversations = [
        {"persona": ["I love dogs", "I am a teacher"], "utterances": [{"text": "Hi"}]},
        {"persona": ["I love dogs", "I am a teacher"], "utterances": [{"text": "Hello"}]},
        {"persona": ["I play guitar", "I am tall"], "utterances": [{"text": "Hey"}]},
    ]
    persona_groups = {}
    for conv in conversations:
        key = tuple(sorted(conv["persona"]))
        persona_groups.setdefault(key, []).append(conv)
    assert len(persona_groups) == 2
    assert len(persona_groups[("I am a teacher", "I love dogs")]) == 2

def test_reservoir_buffer_sampling():
    """Test reservoir buffer maintains max size."""
    from src.streaming_memory import ReservoirBuffer
    buf = ReservoirBuffer(max_size=5)
    for i in range(20):
        buf.add({"text": f"item_{i}"})
    assert len(buf) <= 5

def test_session_buffer_reset():
    """Test session buffer clears properly."""
    from src.streaming_memory import SessionBuffer
    buf = SessionBuffer()
    buf.add({"text": "test"})
    assert len(buf) == 1
    buf.reset()
    assert len(buf) == 0

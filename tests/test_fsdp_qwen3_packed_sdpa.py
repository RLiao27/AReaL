# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine


def test_qwen3_sdpa_packed_samples_match_separate_forward(monkeypatch):
    """Packed samples must not attend to earlier samples in the same batch."""
    torch.manual_seed(1)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.enable_tree_training = False
    engine.model_config = config
    engine.config = SimpleNamespace(
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        pad_to_maximum=False,
        attn_impl="sdpa",
    )
    engine.logger = Mock()
    monkeypatch.setattr(dist, "get_rank", lambda *args: 0)
    inputs = torch.randint(0, config.vocab_size, (2, 16))
    batch = engine._prepare_mb_list(
        {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}
    )
    packed = batch.padded_mbs[0]
    ids = packed["input_ids"]
    with torch.no_grad():
        actual = model(**packed).logits[:, :32]
        expected = torch.cat(
            [
                model(input_ids=ids[:, :16], use_cache=False).logits,
                model(input_ids=ids[:, 16:32], use_cache=False).logits,
            ],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

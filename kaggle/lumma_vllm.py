"""vLLM plugin that serves FrontiersMind's Lumma through the Transformers backend.

vLLM's generic TransformersForCausalLM builds the base LummaModel and its own
lm_head, which it ties to embed_tokens. Lumma's embeddings are factorized
(vocab x embedding_rank), so its causal-LM head first projects the hidden state
down with lm_head_proj (hidden_size -> embedding_rank). The generic wrapper has
no such layer: the checkpoint's lm_head_proj.weight has nowhere to load and the
logits would multiply 1440-wide states by a 512-wide table. This subclass adds
the projection and applies it before the logits.

Loaded in every vLLM process through the vllm.general_plugins entry point that
lumma.py writes next to this file.
"""

import functools


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model("LummaForCausalLM", "lumma_vllm:LummaForCausalLM")


def __getattr__(name):
    # vLLM imports this module in the API server too; build the class only when
    # the model is actually loaded, so plugin loading never pulls in torch layers.
    if name == "LummaForCausalLM":
        return _model_class()
    raise AttributeError(name)


@functools.cache
def _model_class():
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.models.transformers import TransformersForCausalLM
    from vllm.model_executor.models.utils import maybe_prefix

    class LummaForCausalLM(TransformersForCausalLM):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            cfg = self.text_config
            self.lm_head_proj = ReplicatedLinear(
                cfg.hidden_size, cfg.embedding_rank, bias=False,
                prefix=maybe_prefix(prefix, "lm_head_proj"))

        def compute_logits(self, hidden_states):
            hidden_states, _ = self.lm_head_proj(hidden_states)
            return super().compute_logits(hidden_states)

    return LummaForCausalLM

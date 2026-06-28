# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding proposer.

DSpark (DeepSeek-V4-*-DSpark) attaches a hidden-chain (HC) draft module + a
rank-256 Markov head to the V4 backbone and drafts a block of
``dspark_block_size`` tokens per step. Structurally it mirrors DFlash:
parallel drafting (one forward emits the whole block), a noise/mask draft
token (``dspark_noise_token_id``), and context K/V seeded from target hidden
states (DSpark uses the outputs of ``dspark_target_layer_ids``). We therefore
build on the same base machinery as DFlash and specialize the draft model.

WIP: first version reuses the DFlash-style parallel-drafting path and relies on
the target's standard rejection-sampling verify for correctness. The DSpark
confidence head + load-aware dynamic acceptance are deferred (throughput-only).

M3 BLOCKER (next sub-step): subclassing DFlashProposer is the WRONG base for
DSpark's runtime. DFlash's propose path calls model methods DSpark lacks
(``precompute_and_store_context_kv`` — DFlash cross-attention context-KV), so
init crashes at dummy_run/build_model_inputs_first_pass with
``'DsparkMTP' object has no attribute 'precompute_and_store_context_kv'``.

Correct path: base this on the EAGLE3-style proposer (which consumes the target
aux hidden states already wired via DeepseekV4ForCausalLM.set_aux_hidden_state_layers
== dspark_target_layer_ids [40,41,42]). The DSpark propose() must:
  1. take the 3 aux hidden states, concat -> (T, 3*D),
  2. DsparkMTP.model.stages[0].main_proj/main_norm -> seed,
  3. embed [last_token, noise*4], run the 3 HC stages,
  4. last stage: hc_head -> norm -> head logits + sequential rank-256 Markov
     rollout (logits[i] += markov(prev_tok); sample) -> block of
     num_speculative_tokens draft ids,
  5. return [batch, num_speculative_tokens]; let vLLM rejection-verify vs target.
Draft attention for the 3 stages runs through vLLM paged attention (the stages
ARE DeepseekV4DecoderLayers); KV-cache groups for them are already registered.
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        # Bypass DFlashProposer.__init__'s method assertion by temporarily
        # presenting as the shared base init; DSpark shares DFlash's
        # parallel-drafting / context-KV / noise-token machinery.
        spec = vllm_config.speculative_config
        _orig_method = spec.method
        try:
            object.__setattr__(spec, "method", "dflash")
            super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        finally:
            object.__setattr__(spec, "method", _orig_method)
        self.method = "dspark"
        # DSpark drafts a fixed block; the noise token seeds positions 1..N.
        self.dspark_noise_token_id = getattr(
            self.draft_model_config.hf_config, "dspark_noise_token_id", None
        )
        if self.dspark_noise_token_id is not None:
            self.parallel_drafting_token_id = self.dspark_noise_token_id

    @property
    def dflash_config(self):
        # DSpark has no dflash_config block; default to causal=False
        # (parallel block attention), matching DFlash's non-causal draft.
        return {}

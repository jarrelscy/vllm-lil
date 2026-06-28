# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding proposer.

DSpark (DeepSeek-V4-*-DSpark) attaches a hidden-chain (HC) draft module + a
rank-256 Markov head to the V4 backbone and drafts a block of
``dspark_block_size`` tokens per step from the target's hidden states at
``dspark_target_layer_ids`` ([40,41,42]).

Based on SpecDecodeBaseProposer (the EAGLE/parallel-drafting machinery), NOT
DFlashProposer: DFlash's propose path calls cross-attention context-KV methods
(``precompute_and_store_context_kv``) that DSpark lacks. DSpark instead consumes
the target aux hidden states (wired via
DeepseekV4ForCausalLM.set_aux_hidden_state_layers) and runs its own 3 HC stages
through vLLM paged attention, with the target performing rejection-sampling
verify. parallel_drafting=True (set in SpeculativeConfig) → the whole block is
emitted in one draft forward.

CURRENT BLOCKER (M3, after init-gate chain resolved): init now passes proposer
construction, combine_hidden_states, mask_hidden, and weight load (97 params),
then FAILS during profiling/dummy_run when the draft's HC kernels execute. vLLM
reports `Worker failed with error ''` — the real CUDA error is MASKED by
tilelang's teardown stub (`tilelang/lib/libcudart_stub.so: undefined symbol:
cudaDeviceReset`). Hypothesis: the draft forward (DsparkMultiTokenPredictor.
forward running the 3 DeepseekV4DecoderLayer stages via paged attention) feeds
shapes/contexts the HC/sparse-MLA kernels don't expect during dummy_run, hitting
a CUDA fault; tilelang's broken teardown then hides it. NEXT STEPS: (1) set
VLLM_ENABLE_V1_MULTIPROCESSING=0 or run single-proc / catch the worker exception
to surface the real CUDA error pre-teardown; (2) audit DsparkMultiTokenPredictor.
forward dummy_run path vs the reference DSparkBlock (it uses windowed sparse attn
with main_x context, NOT standard paged causal attn — the stages may need a
custom attention path or the dummy_run shapes corrected); (3) consider
--enforce-eager to bypass cudagraph capture during bring-up.

ARCHITECTURAL NOTE (M3/M4): SpecDecodeBaseProposer.__init__ auto-expands
hidden_size *= hc_mult for DeepseekV4 (it assumes the standard V4-MTP input =
the pre-hc_head residual of width hc_mult*D). DSpark's main_proj instead expects
the concat of len(dspark_target_layer_ids) aux layers (width n_layers*D). The
base's hidden_size and combine_hidden_states branch must therefore special-case
"dspark" (n_layers*D), and DsparkMTP.combine_hidden_states must apply
stage0.main_proj/main_norm. Confidence head + Markov rollout refinement = M5.
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DSparkProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # The noise token seeds masked draft slots (positions 1..N of the block).
        # Already picked up by _init_parallel_drafting_params via
        # dspark_noise_token_id, but keep an explicit handle.
        self.dspark_noise_token_id = getattr(
            self.draft_model_config.hf_config, "dspark_noise_token_id", None
        )
        if self.dspark_noise_token_id is not None:
            self.parallel_drafting_token_id = self.dspark_noise_token_id

    def _get_eagle3_use_aux_hidden_state_from_config(self):
        # DSpark consumes the concat of aux hidden states at
        # dspark_target_layer_ids, then projects via stage0.main_proj
        # (combine_hidden_states). So it follows the EAGLE3 aux path.
        return True

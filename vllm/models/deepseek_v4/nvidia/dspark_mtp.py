# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek V4 (DeepSeek-V4-*-DSpark checkpoints).

DSpark replaces the standard single-block V4 MTP with a 3-stage hidden-chain
(HC) draft + a rank-256 Markov head + a confidence head, drafting a block of
``dspark_block_size`` tokens per step. Checkpoint layout (verified):

  mtp.{0,1,2}.*            three draft stages, each a standard V4 decoder block
                           (attn / ffn / attn_norm / ffn_norm / hc_attn_* /
                           hc_ffn_*) — same param structure as DeepseekV4DecoderLayer
  mtp.0.main_proj.{weight,scale}, mtp.0.main_norm.weight
                           project concat of target layers (dspark_target_layer_ids)
  mtp.2.norm.weight, mtp.2.hc_head_{fn,base,scale}
  mtp.2.markov_head.markov_w1.weight [vocab, rank]
  mtp.2.markov_head.markov_w2.weight [vocab, rank]
  mtp.2.confidence_head.proj.weight  [1, hidden + rank]
  embed.weight, head.weight, norm.weight   shared with main model (draft reuses)
  hc_head_{fn,base,scale}  top-level == MAIN model's final hc_head (draft SKIPS)

This module targets a clean weight load + construction. The draft forward
(HC rollout + sequential Markov sampling) is driven by DSparkProposer.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _hc_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


class DsparkStage(nn.Module):
    """One DSpark draft stage: a V4 decoder block (HC attn+ffn) plus the
    stage-specific extras (input projection on stage 0, heads on the last)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        stage_id: int,
        num_stages: int,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.stage_id = stage_id
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size

        # The HC decoder block (attn + MoE ffn + hc_attn_*/hc_ffn_* params).
        # Use a synthetic layer index >= num_hidden_layers so the attention takes
        # the MTP branch (compress_ratio=1, NO Lightning Indexer / compressor) —
        # matching the reference DSpark draft, which asserts compress_ratio==0.
        # With the natural stages.{0,1,2} prefix, extract_layer_index gives
        # 0/1/2 and compress_ratios[2]==4 builds the indexer, whose
        # sparse_attn_indexer profiling hard-aborts in _dummy_run. The integer in
        # this prefix only drives extract_layer_index + the forward-context key;
        # loaded param names come from the module hierarchy (stages.{i}.mtp_block),
        # so the weight loader is unaffected.
        _root = prefix.rsplit(".stages.", 1)[0] if ".stages." in prefix else prefix
        _block_idx = config.num_hidden_layers + stage_id
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            f"{_root}.stages.{_block_idx}.mtp_block",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

        # Stage 0 projects the concatenated target-layer hidden states.
        if stage_id == 0:
            n_target = len(getattr(config, "dspark_target_layer_ids", []) or [1])
            self.main_proj = ReplicatedLinear(
                config.hidden_size * n_target,
                config.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Last stage carries the heads used to emit draft tokens.
        if stage_id == num_stages - 1:
            rank = config.dspark_markov_rank
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # Markov head: low-rank vocab->rank->vocab bias (loaded as-is).
            self.markov_w1 = nn.Parameter(
                torch.empty(config.vocab_size, rank), requires_grad=False
            )
            self.markov_w2 = nn.Parameter(
                torch.empty(config.vocab_size, rank), requires_grad=False
            )
            self.confidence_proj_weight = nn.Parameter(
                torch.empty(1, config.hidden_size + rank, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_fn = _hc_param(self.hc_mult, self.hc_dim)
            self.hc_head_base = _hc_param(self.hc_mult)
            self.hc_head_scale = _hc_param(1)


class DsparkMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_stages = getattr(config, "n_mtp_layers", None) or 3
        # DSpark stages are stored as mtp.{i}; map to global layer indices for
        # uniqueness alongside the main model's num_hidden_layers.
        self.stage_start = config.num_hidden_layers

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        self.stages = nn.ModuleDict(
            {
                str(i): DsparkStage(
                    vllm_config,
                    stage_id=i,
                    num_stages=self.num_stages,
                    topk_indices_buffer=self.topk_indices_buffer,
                    prefix=f"{prefix}.stages.{i}",
                    aux_stream_list=aux_stream_list,
                )
                for i in range(self.num_stages)
            }
        )

        # Shared with the main model (draft reuses embed + head + final norm).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        # DSpark seeds the draft from the concat of the aux target layers
        # (dspark_target_layer_ids). Stage 0's main_proj reduces n_target*D -> D,
        # then main_norm. Mirrors the reference `main_norm(main_proj(main_hidden))`.
        # Rank-robust: the proposer calls this both with (T, n_target*D) during
        # propose() and with a flat 1D mask_hidden (n_target*D,) during
        # load_model. main_norm (RMSNorm / _C.rms_norm) requires a 2D (T, D)
        # tensor, so normalize the rank around the projection.
        stage0 = self.stages["0"]
        x = target_hidden_states
        squeeze_back = x.dim() == 1
        if squeeze_back:
            x = x.unsqueeze(0)
        out = stage0.main_norm(stage0.main_proj(x))
        if squeeze_back:
            out = out.squeeze(0)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # NOTE: structural — the full reference uses main_x as the stages'
        # attention context; here we seed by adding the combined hidden state
        # to the draft-token embeddings, run the 3 HC stages through vLLM paged
        # attention, and let compute_logits + target verify produce coherent
        # output. Markov rollout / confidence head are M5 refinements.
        import traceback as _tb

        try:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            # previous_hidden_states is the combine_hidden_states output (T, D).
            if previous_hidden_states is not None:
                if previous_hidden_states.dim() == inputs_embeds.dim():
                    inputs_embeds = inputs_embeds + previous_hidden_states
            hidden = inputs_embeds
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1).repeat(1, self.config.hc_mult, 1)
            logger.info(
                "DSpark draft forward: input_ids=%s positions=%s prev=%s hidden=%s",
                tuple(input_ids.shape) if input_ids is not None else None,
                tuple(positions.shape),
                tuple(previous_hidden_states.shape)
                if previous_hidden_states is not None
                else None,
                tuple(hidden.shape),
            )
            post_mix = res_mix = residual = None
            for i in range(self.num_stages):
                block = self.stages[str(i)].mtp_block
                hidden, residual, post_mix, res_mix = block(
                    x=hidden,
                    positions=positions,
                    input_ids=input_ids,
                    post_mix=post_mix,
                    res_mix=res_mix,
                    residual=residual,
                )
            hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
            return hidden.flatten(1)
        except Exception:
            # tilelang's broken teardown (cudaDeviceReset stub) masks the real
            # error; log the true traceback before re-raising.
            logger.error("DSpark draft forward FAILED:\n%s", _tb.format_exc())
            raise

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        last = self.stages[str(self.num_stages - 1)]
        hidden_states = hidden_states.view(-1, last.hc_mult, self.config.hidden_size)
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
            last.rms_norm_eps,
            last.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return self.logits_processor(self.head, hidden_states)


class DsparkMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DsparkMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Hidden state used for masked (noise) draft slots in parallel drafting.
        # DSpark seeds masked slots with the noise TOKEN embedding, so a zero
        # hidden-state mask is the correct neutral stand-in. Width = aux concat
        # (len(target_layer_ids)*D); the proposer projects it via
        # combine_hidden_states (stage0 main_proj) -> D.
        n_target = len(getattr(self.config, "dspark_target_layer_ids", []) or [1])
        self.register_buffer(
            "mask_hidden",
            torch.zeros(1, self.config.hidden_size * n_target),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(target_hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    # ---- weight loading -------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_stage = next(iter(self.model.stages.values()))
        if first_stage.mtp_block.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # Flat (non-block) params that load directly via reassignment / copy.
        direct_params = {
            "embed.weight": "model.embed_tokens.weight",
            "head.weight": "model.head.weight",
            "norm.weight": "model.norm.weight",
        }

        def _flex_load(pname: str, w: torch.Tensor) -> bool:
            param = params_dict.get(pname)
            if param is None:
                return False
            if tuple(param.shape) == tuple(w.shape):
                default_weight_loader(param, w)
            else:
                param.data = w.to(param.data.device)
            loaded_params.add(pname)
            return True

        num_stages = self.model.num_stages
        last_stage = num_stages - 1

        for name, loaded_weight in weights:
            # Shared top-level weights (skip main-model-only norm/hc_head).
            if name in direct_params:
                _flex_load(direct_params[name], loaded_weight)
                continue
            if name.startswith("hc_head_") and not name.startswith("mtp."):
                # main model's final hc_head — not used by the draft.
                continue
            if name.startswith("layers.") or name.startswith("model.layers."):
                continue  # main model body

            m = re.match(r"mtp\.(\d+)\.(.*)", name)
            if m is None:
                continue
            stage_i = int(m.group(1))
            rest = m.group(2)
            if stage_i >= num_stages:
                continue
            base = f"model.stages.{stage_i}"

            # Stage-specific extras (heads / projections).
            if rest == "main_norm.weight":
                _flex_load(f"{base}.main_norm.weight", loaded_weight)
                continue
            if rest.startswith("main_proj."):
                sub = rest[len("main_proj.") :]
                pname = f"{base}.main_proj.{sub}"
                if pname.endswith(".scale"):
                    pname = pname.removesuffix(".scale") + ".weight_scale_inv"
                param = params_dict.get(pname)
                if param is not None:
                    wl = getattr(param, "weight_loader", default_weight_loader)
                    wl(param, loaded_weight)
                    loaded_params.add(pname)
                continue
            if stage_i == last_stage and rest == "norm.weight":
                _flex_load(f"{base}.norm.weight", loaded_weight)
                continue
            if rest == "markov_head.markov_w1.weight":
                _flex_load(f"{base}.markov_w1", loaded_weight)
                continue
            if rest == "markov_head.markov_w2.weight":
                _flex_load(f"{base}.markov_w2", loaded_weight)
                continue
            if rest == "confidence_head.proj.weight":
                _flex_load(f"{base}.confidence_proj_weight", loaded_weight)
                continue
            if rest.startswith("hc_head_"):
                _flex_load(f"{base}.{rest}", loaded_weight)
                continue

            # Otherwise it's a decoder-block weight: route into mtp_block and
            # apply the standard V4 stacked / expert / scale mapping.
            bname = f"{base}.mtp_block.{rest}"
            if bname.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(bname)
                    else ".weight_scale_inv"
                )
                bname = bname.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in bname:
                    continue
                if weight_name not in bname:
                    continue
                mapped = bname.replace(weight_name, param_name)
                param = params_dict.get(mapped)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if ".experts." in bname:
                    if (
                        "weight_scale" in bname
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in bname:
                            continue
                        mapped = bname.replace(weight_name, param_name)
                        param = params_dict.get(mapped)
                        if param is None:
                            continue
                        wl = typing.cast(Callable[..., bool], param.weight_loader)
                        if wl(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            loaded_params.add(mapped)
                            break
                    continue
                if "attn_sink" in bname:
                    param = params_dict.get(bname)
                    if param is not None:
                        narrow = loaded_weight[head_rank_start:head_rank_end]
                        param[: narrow.shape[0]].copy_(narrow)
                        loaded_params.add(bname)
                    continue
                if ".shared_experts.w2" in bname:
                    bname = bname.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if bname.endswith(".ffn.gate.bias"):
                    bname = bname.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict.get(bname)
                if param is None:
                    continue
                wl = getattr(param, "weight_loader", default_weight_loader)
                wl(param, loaded_weight)
                loaded_params.add(bname)

        # Require every stage's core block to have loaded something.
        for i in range(num_stages):
            if not any(f"model.stages.{i}.mtp_block" in p for p in loaded_params):
                raise ValueError(
                    f"DSpark draft stage {i} weights missing from checkpoint."
                )
        for stage in self.model.stages.values():
            stage.mtp_block.ffn.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

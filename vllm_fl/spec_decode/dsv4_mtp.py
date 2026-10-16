# SPDX-License-Identifier: Apache-2.0
"""DSv4-only eager MTP proposer and exact greedy verifier for FL Ascend."""
import os
from copy import copy
from dataclasses import fields, replace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.eagle import EagleProposer

from .dsv4_mtp_utils import (
    advance_draft_positions, greedy_rejection_tokens, padded_draft_indices,
    resolve_next_tokens, validate_dsv4_mtp_config,
)


class Dsv4MTPProposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        validate_dsv4_mtp_config(vllm_config)
        super().__init__(vllm_config, device, runner)
        # The installed generic SpecDecodeBaseProposer accepts runner but
        # does not retain it (unlike AscendSpecDecodeBaseProposer).
        self.runner = runner
        self._fast_mtp_metadata = bool(
            (vllm_config.additional_config or {}).get("dsv4_mtp_fast_metadata", False))
        self._host_metadata_active = False
        self._pending_draft_counts = None

    def _get_model(self):
        # Installed vLLM copies the target's eager flag when constructing the
        # draft ModelConfig. Keep this draft-only override explicit.
        self.speculative_config.draft_model_config.enforce_eager = True
        return super()._get_model()

    def _maybe_share_embeddings(self, target_language_model):
        # DSv4 exports mtp.0.emb.tok_emb separately from target embeddings.
        pass

    def _maybe_share_lm_head(self, target_language_model):
        self._share_mtp_indices = False

    def _determine_batch_execution_and_padding(self, num_tokens, use_cudagraphs=True):
        # Keep upstream DP rank coordination even though the draft runs eager.
        return super()._determine_batch_execution_and_padding(num_tokens, False)

    def dummy_run(self, num_tokens, use_cudagraphs=True,
                  is_graph_capturing=False, slot_mappings=None):
        if is_graph_capturing:
            return
        return super().dummy_run(num_tokens, False, False, slot_mappings)

    def propose(self, *args, **kwargs):
        sampling = kwargs.get("sampling_metadata", args[7] if len(args) > 7 else None)
        if sampling is None or not sampling.all_greedy or sampling.max_num_logprobs is not None:
            raise NotImplementedError("DSv4 MTP requires temperature=0 and no logprobs")
        args = list(args)
        self._draft_active_mask = None
        cad = copy(kwargs["common_attn_metadata"] if "common_attn_metadata" in kwargs else args[6])
        cad.seq_lens = cad.seq_lens.clone()
        lengths = cad._seq_lens_cpu
        if lengths is None:
            lengths = cad.seq_lens_cpu
        if lengths is None or cad.positions_cpu is None:
            raise RuntimeError("DSv4 MTP requires rejection-corrected target CPU metadata")
        if lengths.device.type != "cpu" or cad.positions_cpu.device.type != "cpu":
            raise RuntimeError("DSv4 MTP host metadata must already reside on CPU")
        self._host_lengths = lengths[:cad.num_reqs].clone()
        self._host_positions = cad.positions_cpu[:cad.num_actual_tokens].clone()
        self._host_query_starts = cad.query_start_loc_cpu[:cad.num_reqs + 1].clone()
        rejected = kwargs.get("num_rejected_tokens_gpu", args[9] if len(args) > 9 else None)
        self._host_needs_rejection = rejected is not None
        self._host_sampling_positions = None
        self._host_active_mask = None
        self._host_metadata_active = True
        if "common_attn_metadata" in kwargs:
            kwargs["common_attn_metadata"] = cad
        else:
            args[6] = cad
        try:
            return super().propose(*args, **kwargs)
        finally:
            # Dummy/profile forwards must never reuse a previous real batch.
            self._host_metadata_active = False
            self._pending_draft_counts = None

    def prepare_next_token_ids_padded(self, sampled_token_ids, requests,
                                      gpu_input_batch, discard_request_mask):
        batch = gpu_input_batch.num_reqs
        for i in range(batch):
            self.backup_next_token_ids.np[i] = requests[gpu_input_batch.req_ids[i]].get_token_id(
                gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(batch)
        # The runner passes a max_num_reqs-sized mask, while the sampled
        # token tensor contains only active requests. Slice before torch.where
        # to prevent a one-request batch broadcasting to the full capacity.
        return resolve_next_tokens(sampled_token_ids[:batch], discard_request_mask[:batch],
                                   self.backup_next_token_ids.gpu[:batch],
                                   gpu_input_batch.vocab_size)

    def prepare_inputs_padded(self, common_attn_metadata, spec_decode_metadata,
                              valid_sampled_tokens_count):
        indices, rejected = padded_draft_indices(
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count, common_attn_metadata.query_start_loc)
        # Preserve Ascend positions, phase and CPU mirrors, not just base fields.
        self._pending_draft_counts = tuple(spec_decode_metadata.num_draft_tokens)
        return copy(common_attn_metadata), indices, rejected

    def prepare_inputs(self, common_attn_metadata, sampled_token_ids, num_draft_tokens):
        base, indices = super().prepare_inputs(
            common_attn_metadata, sampled_token_ids, num_draft_tokens)
        result = copy(common_attn_metadata)
        for field in fields(base):
            setattr(result, field.name, getattr(base, field.name))
        result.positions = common_attn_metadata.positions[indices]
        result.positions_cpu = result.positions.cpu()
        return result, indices

    def _update_positions_dependent_metadata(self, positions, common_attn_metadata,
                                            batch_size, input_batch_size, block_size):
        if self._host_metadata_active:
            self._advance_host_metadata(common_attn_metadata, batch_size, block_size)
        new_pos, slots, lengths = advance_draft_positions(
            positions, common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens, block_size, self.max_model_len,
            getattr(self, "_draft_active_mask", None))
        self._draft_active_mask = slots >= 0
        self.positions[:batch_size].copy_(new_pos)
        self._slot_mapping_buffer[:input_batch_size].fill_(-1)
        self._slot_mapping_buffer[:batch_size].copy_(slots)
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
        # Do not mutate the target runner's lengths via an aliased tensor.
        common_attn_metadata.seq_lens = lengths
        return self.positions[:batch_size]

    def _advance_host_metadata(self, cad, batch_size, block_size):
        if self._host_sampling_positions is None:
            ends = self._host_query_starts[1:] - 1
            if self._host_needs_rejection:
                # The runner enqueued this count copy before propose(). Wait
                # once per proposal, only when K>1 actually needs a new step.
                # Reuse its event instead of copying positions/lengths per step.
                if self.runner is None or self._pending_draft_counts is None:
                    raise RuntimeError("Missing DSv4 rejection-count owner")
                counts = self.runner._get_valid_sampled_token_count()
                drafts = torch.tensor(self._pending_draft_counts, dtype=torch.int32)
                valid = torch.tensor(counts, dtype=torch.int32)
                if valid.shape != drafts.shape or valid.numel() != batch_size:
                    raise RuntimeError("DSv4 acceptance-count rows do not match the draft batch")
                if torch.any((valid < 0) | (valid > drafts + 1)):
                    raise RuntimeError("DSv4 acceptance count exceeds the previous draft extent")
                rejected = torch.where(drafts > 0, drafts + 1 - valid, 0)
                ends = ends - rejected
                self._host_lengths = self._host_lengths - rejected
            self._host_sampling_positions = self._host_positions[ends.long()]

        # This is the host mirror of the same kernel-block table used by the
        # GPU path, including hybrid block splitting and row compaction.
        if self.runner is None:
            raise RuntimeError("DSv4 MTP requires the runner's CPU block table")
        table = self.runner.input_batch.block_table[self.kv_cache_gid]
        if table.block_size != block_size:
            raise RuntimeError("DSv4 host/GPU kernel block sizes differ")
        host_table = table.get_cpu_tensor()[:batch_size, :cad.block_table_tensor.shape[1]]
        positions, slots, lengths = advance_draft_positions(
            self._host_sampling_positions, host_table, self._host_lengths,
            block_size, self.max_model_len, self._host_active_mask)
        self._host_sampling_positions = positions
        self._host_positions = positions
        self._host_lengths = lengths
        self._host_active_mask = slots >= 0

    def build_per_group_and_layer_attn_metadata(self, common_attn_metadata, draft_index=0):
        from vllm_fl.attention.ascend.dsa_v1 import AscendDSAMetadataBuilder
        cad = common_attn_metadata
        count = cad.num_actual_tokens
        cad.positions = self.positions[:count]
        if self._host_metadata_active:
            cad.positions_cpu = self._host_positions[:count]
            lengths = self._host_lengths
        else:
            # Startup dummy/profile only; no real target host snapshot exists.
            cad.positions_cpu = cad.positions.cpu()
            lengths = cad.seq_lens.cpu()
        queries = cad.query_start_loc_cpu[1:] - cad.query_start_loc_cpu[:-1]
        cad._seq_lens_cpu = cad.seq_lens_cpu = lengths
        cad._num_computed_tokens_cpu = cad.num_computed_tokens_cpu = lengths - queries
        cad.seq_lens_cpu_upper_bound = lengths.clone()
        cad.max_seq_len = int(lengths.max()) if lengths.numel() else 0
        cad.num_input_tokens = count
        cad.graph_pad_size = -1
        groups, layers = [], {}
        for group in self.draft_attn_groups:
            builder = group.get_metadata_builder()
            if not isinstance(builder, AscendDSAMetadataBuilder) or builder.compressor_ratio > 1:
                raise NotImplementedError("DSv4 MTP requires FL DSA SWA-only draft attention")
            fast_decode = (
                getattr(self, "_fast_mtp_metadata", False) and self._host_metadata_active
                and draft_index > 0 and cad.max_query_len == 1
                and cad.num_actual_tokens == cad.num_reqs and cad.num_reqs > 0
                and cad.graph_pad_size == -1)
            if fast_decode:
                metadata = builder.build_for_dsv4_mtp_decode(cad, draft_index)
            else:
                metadata = builder.build(
                    common_prefix_len=0, common_attn_metadata=cad,
                    num_reqs_actual=cad.num_reqs,
                    prefill_ratio_to_sas_metadata={}, decode_ratio_to_sas_metadata={},
                    common_ratio_to_sas_metadata={}, block_size=group.kv_cache_spec.block_size)
            groups.append(metadata)
            layers.update(dict.fromkeys(group.layer_names, metadata))
        return groups, layers


class Dsv4GreedyRejectionSampler(RejectionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flag = os.getenv("MTP_FUSED_GREEDY", "1")
        if flag not in ("0", "1"):
            raise ValueError("MTP_FUSED_GREEDY must be 0 or 1")
        self._fused_greedy = flag == "1"

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        if not sampling_metadata.all_greedy or sampling_metadata.max_num_logprobs is not None:
            raise NotImplementedError("DSv4 MTP requires temperature=0 and no logprobs")
        bonus = self.sampler(
            logits=logits[metadata.bonus_logits_indices],
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
            predict_bonus_token=True).sampled_token_ids
        target = self.apply_logits_processors(
            logits[metadata.target_logits_indices].float(), sampling_metadata, metadata)
        target_argmax = target.argmax(-1)
        if self._fused_greedy and logits.device.type in ("npu", "privateuseone"):
            from vllm.triton_utils import HAS_TRITON
            if not HAS_TRITON:
                raise RuntimeError("MTP_FUSED_GREEDY=1 requires Ascend Triton; use 0 for the previous path")
            from .dsv4_mtp_greedy import fused_greedy_rejection_tokens
            tokens = fused_greedy_rejection_tokens(metadata, target_argmax, bonus)
        else:
            tokens = greedy_rejection_tokens(
                metadata.draft_token_ids, metadata.num_draft_tokens,
                target_argmax, bonus, metadata.max_spec_len)
        return SamplerOutput(sampled_token_ids=tokens, logprobs_tensors=None)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from .circuit_pin_slot_tokenizer import CircuitPinSlotTokenizer, PinSlotSpec


class CircuitPinSlotGCTokenizer(CircuitPinSlotTokenizer):
    def __init__(
        self,
        *,
        gate_count_buckets: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        buckets = [64, 80, 96, 112, 128] if gate_count_buckets is None else [int(x) for x in gate_count_buckets]
        buckets = sorted(set(int(x) for x in buckets if int(x) > 0))
        self.gate_count_buckets = buckets
        self.gc_tokens = {}
        for b in self.gate_count_buckets:
            name = f'GC_{int(b)}'
            self.gc_tokens[int(b)] = int(len(self.special_toks))
            self.special_toks.append(name)
        self.idx_offset = len(self.special_toks)

    def _bucket_for_gate_count(self, gate_count: int) -> int:
        gate_count = int(gate_count)
        if not self.gate_count_buckets:
            return gate_count
        best = self.gate_count_buckets[0]
        best_d = abs(gate_count - int(best))
        for b in self.gate_count_buckets[1:]:
            d = abs(gate_count - int(b))
            if d < best_d:
                best = int(b)
                best_d = d
        return int(best)

    def tokenize(self, data: Data):
        tokens = super().tokenize(data)
        if int(tokens.numel()) <= 0:
            return tokens
        x = data.x.reshape(-1).to(torch.long)
        gate = (x != int(self.net_id)) & (x != int(self.boundary_stub_id))
        gate_count = int(torch.nonzero(gate, as_tuple=False).numel())
        bucket = self._bucket_for_gate_count(gate_count)
        tok = self.gc_tokens.get(int(bucket))
        if tok is None:
            return tokens
        if int(tokens[0].item()) != int(self.sos):
            return tokens
        out = torch.cat([tokens[:1], torch.tensor([int(tok)], dtype=torch.long), tokens[1:]], dim=0)
        return out

    def decode(self, t: torch.Tensor):
        t = t.detach().cpu().reshape(-1).to(torch.long)
        target_bucket = None
        if int(t.numel()) >= 2 and int(t[0].item()) == int(self.sos):
            inv = {int(v): int(k) for k, v in self.gc_tokens.items()}
            b = inv.get(int(t[1].item()))
            if b is not None:
                target_bucket = int(b)
                t = torch.cat([t[:1], t[2:]], dim=0)
        g = super().decode(t)
        if target_bucket is not None:
            setattr(g, 'target_gate_count_bucket', int(target_bucket))
        return g

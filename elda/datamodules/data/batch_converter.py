import torch
from typing import Sequence


class BatchConverter(object):
    """Callable to convert a list of trails to a processed batch.
    """

    def __init__(self, tokenizer, truncation_length: int = None):
        self.tokenizer = tokenizer
        self.truncation_length = truncation_length

    def _protected_prefix_len(self, x: torch.Tensor) -> int:
        if len(x) == 0 or int(x[0].item()) != int(self.tokenizer.sos):
            return 0
        pos = 1
        dataset_count = len(getattr(self.tokenizer, "dataset_names", []))
        if dataset_count > 0 and pos < len(x):
            tok = int(x[pos].item())
            first_dataset = len(self.tokenizer.special_toks)
            if first_dataset <= tok < first_dataset + dataset_count:
                pos += 1
        if getattr(self.tokenizer, "has_schema_tokens", False):
            for schema_pos in range(len(getattr(self.tokenizer, "schema_token_specs", []))):
                if pos >= len(x):
                    break
                tok = int(x[pos].item())
                if tok in set(self.tokenizer.get_schema_token_ids_for_position(schema_pos)):
                    pos += 1
                else:
                    break
        return pos

    def _truncate_preserving_prefix(self, x: torch.Tensor) -> torch.Tensor:
        if self.truncation_length is None or self.truncation_length >= len(x):
            return x
        prefix_len = min(self._protected_prefix_len(x), self.truncation_length)
        if prefix_len <= 0:
            start_idx = torch.randint(0, len(x) - self.truncation_length + 1, (1,)).item()
            return x[start_idx:start_idx + self.truncation_length]

        body_budget = self.truncation_length - prefix_len
        if body_budget <= 0:
            return x[:self.truncation_length]
        body = x[prefix_len:]
        if len(body) <= body_budget:
            return x[:self.truncation_length]
        start_idx = torch.randint(0, len(body) - body_budget + 1, (1,)).item()
        return torch.cat([x[:prefix_len], body[start_idx:start_idx + body_budget]], dim=0)

    def __call__(self, batch: Sequence[torch.Tensor]):
        aux_payload = None
        if len(batch) > 0 and isinstance(batch[0], (tuple, list)):
            sequences = [item[0] for item in batch]
            aux_items = [item[1] for item in batch]
            batch = sequences
            if aux_items and isinstance(aux_items[0], dict):
                aux_payload = {}
                keys = sorted({key for item in aux_items for key in item.keys()})
                for key in keys:
                    values = [float(item.get(key, 0.0)) for item in aux_items]
                    aux_payload[key] = torch.tensor(values, dtype=torch.float32)

        batch_size = len(batch)

        max_len = max([len(b) for b in batch])
        if self.truncation_length is not None:
            max_len = min(max_len, self.truncation_length)

        batched_tensor = torch.full(
            [batch_size, max_len],
            self.tokenizer.pad,
            dtype=batch[0].dtype
        )

        for i, x in enumerate(batch):
            x = self._truncate_preserving_prefix(x)
            batched_tensor[i, :len(x)] = x

        if aux_payload is not None:
            return batched_tensor, aux_payload
        return batched_tensor

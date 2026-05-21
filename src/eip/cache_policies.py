from __future__ import annotations

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicSlidingWindowLayer


class AttentionSinkLayer(CacheLayerMixin):
    """Keep sink tokens from the prefix and a recent-window tail."""

    is_sliding = True

    def __init__(self, window_length: int, num_sink_tokens: int):
        super().__init__()
        if window_length < 1:
            raise ValueError("window_length must be >= 1")
        if num_sink_tokens < 0:
            raise ValueError("num_sink_tokens must be >= 0")
        self.window_length = int(window_length)
        self.num_sink_tokens = int(num_sink_tokens)
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _prune(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_tokens = key_states.shape[-2]
        keep_total = self.num_sink_tokens + self.window_length
        if total_tokens <= keep_total:
            return key_states, value_states

        sink_keys = key_states[:, :, : self.num_sink_tokens, :] if self.num_sink_tokens else key_states[:, :, :0, :]
        sink_values = (
            value_states[:, :, : self.num_sink_tokens, :] if self.num_sink_tokens else value_states[:, :, :0, :]
        )
        recent_keys = key_states[:, :, -self.window_length :, :]
        recent_values = value_states[:, :, -self.window_length :, :]
        pruned_keys = torch.cat([sink_keys, recent_keys], dim=-2)
        pruned_values = torch.cat([sink_values, recent_values], dim=-2)
        return pruned_keys, pruned_values

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        self.keys, self.values = self._prune(full_key_states, full_value_states)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        cached_length = 0 if not self.is_initialized else self.keys.shape[-2]
        return cached_length + query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.num_sink_tokens + self.window_length

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(self.cumulative_length - abs(max_length), 0)
        if self.cumulative_length <= max_length:
            return
        if not self.is_initialized:
            self.cumulative_length = max_length
            return
        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]
        self.cumulative_length = max_length


class AttentionSinkCache(Cache):
    """A per-layer cache that keeps sink tokens plus a recent window."""

    def __init__(self, config, window_length: int, num_sink_tokens: int):
        decoder_config = config.get_text_config(decoder=True)
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * decoder_config.num_hidden_layers
        if hasattr(decoder_config, "num_kv_shared_layers"):
            layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

        layers: list[CacheLayerMixin] = []
        for layer_type in layer_types:
            if layer_type in {"full_attention", "sliding_attention"}:
                layers.append(AttentionSinkLayer(window_length=window_length, num_sink_tokens=num_sink_tokens))
            else:
                raise ValueError(f"Unsupported layer type for AttentionSinkCache: {layer_type}")

        super().__init__(layers=layers, offloading=False, offload_only_non_sliding=False)


class RecencyWindowLayer(CacheLayerMixin):
    """Keep a contiguous recency window for full-attention layers."""

    is_sliding = False

    def __init__(self, window_length: int):
        super().__init__()
        if window_length < 1:
            raise ValueError("window_length must be >= 1")
        self.window_length = int(window_length)
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)

        self.keys = full_key_states[:, :, -self.window_length + 1 :, :]
        self.values = full_value_states[:, :, -self.window_length + 1 :, :]
        return full_key_states, full_value_states

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        is_full = self.cumulative_length >= self.window_length
        kv_offset = max(self.cumulative_length - self.window_length + 1, 0)
        if is_full:
            kv_length = self.window_length - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.window_length


class HybridRecencyWindowCache(Cache):
    """
    Preserve Gemma4's native sliding-window layers while applying a contiguous
    recency window only to full-attention layers.
    """

    def __init__(self, config, window_length: int):
        decoder_config = config.get_text_config(decoder=True)
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * decoder_config.num_hidden_layers
        if hasattr(decoder_config, "num_kv_shared_layers"):
            layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

        layers: list[CacheLayerMixin] = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                layers.append(DynamicSlidingWindowLayer(decoder_config))
            elif layer_type == "full_attention":
                layers.append(RecencyWindowLayer(window_length=window_length))
            else:
                raise ValueError(f"Unsupported layer type for HybridRecencyWindowCache: {layer_type}")

        super().__init__(layers=layers, offloading=False, offload_only_non_sliding=False)

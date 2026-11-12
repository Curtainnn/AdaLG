"""Freeze evaluation states across paired local-only interventions."""
import hashlib

import torch


class LocalReferenceCache:
    def __init__(self):
        self.active = False
        self.entries = []
        self.cursor = 0
        self.capture = True

    def begin(self):
        self.entries = []
        self.cursor = 0
        self.capture = True
        self.active = True

    def rewind(self):
        if not self.active:
            raise RuntimeError('Local reference cache is not active.')
        if self.cursor != len(self.entries):
            raise RuntimeError('Local intervention passes have different numbers of batches.')
        self.cursor = 0
        self.capture = False

    def end(self):
        self.active = False
        self.entries = []
        self.cursor = 0

    @staticmethod
    def fingerprint(value):
        digest = hashlib.sha256()

        def update(item):
            if torch.is_tensor(item):
                digest.update(str((item.dtype, tuple(item.shape))).encode())
                raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                digest.update(raw.numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item):
                    digest.update(str(key).encode())
                    update(item[key])
            elif isinstance(item, (list, tuple)):
                for child in item:
                    update(child)
            else:
                digest.update(repr(item).encode())

        update(value)
        return digest.hexdigest()

    @staticmethod
    def _map(value, tensor_fn):
        if torch.is_tensor(value):
            return tensor_fn(value)
        if isinstance(value, dict):
            return {key: LocalReferenceCache._map(item, tensor_fn) for key, item in value.items()}
        if isinstance(value, list):
            return [LocalReferenceCache._map(item, tensor_fn) for item in value]
        if isinstance(value, tuple):
            return tuple(LocalReferenceCache._map(item, tensor_fn) for item in value)
        return value

    def lookup(self, inputs, device):
        if not self.active:
            return None, None
        if torch.is_grad_enabled():
            raise RuntimeError('Local reference caching requires no_grad evaluation.')
        index = self.cursor
        self.cursor += 1
        signature = self.fingerprint(inputs)
        if self.capture:
            return (index, signature), None
        if index >= len(self.entries):
            raise RuntimeError('Local intervention pass contains an extra batch.')
        saved_signature, state = self.entries[index]
        if signature != saved_signature:
            raise RuntimeError('Inputs or query order changed during a local-only intervention.')
        return None, self._map(state, lambda tensor: tensor.to(device))

    def store(self, ticket, state):
        if ticket is None:
            return
        index, signature = ticket
        if index != len(self.entries):
            raise RuntimeError('Local reference batches must be captured in order.')
        # Keep references off GPU, with no graph or alias to mutable live inputs.
        frozen = self._map(state, lambda tensor: tensor.detach().cpu().clone())
        self.entries.append((signature, frozen))

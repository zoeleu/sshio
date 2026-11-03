"""Fixed-size opponent history with one permanent initial anchor."""
import copy
import torch
from torch.distributions import Categorical


def role_masks(count, agents, device):
    index = torch.arange(count, device=device)
    slots = index % agents + index // agents if agents >= 3 else index
    heuristic = slots % 8 == 0
    historical = slots % 8 == 1 if agents >= 3 else slots % 8 == 3
    return heuristic, historical, ~(heuristic | historical)


class League:
    def __init__(self, model, capacity=8, states=None):
        self.capacity = capacity
        self.models = []
        for state in states or [model.state_dict()]:
            frozen = copy.deepcopy(model).eval()
            frozen.load_state_dict(state)
            frozen.requires_grad_(False)
            self.models.append(frozen)
        self._trim()

    def _trim(self):
        # Retain the initial anchor and the most recent opponents.
        if len(self.models) > self.capacity:
            self.models = [self.models[0], *self.models[-(self.capacity - 1):]]

    def add(self, model):
        frozen = copy.deepcopy(model).eval().requires_grad_(False)
        self.models.append(frozen)
        self._trim()

    def states(self):
        return [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in self.models]

    @torch.no_grad()
    def play(self, obs, actions, historical, agents, round_index):
        # Build fixed-size index tensors once per league round. Boolean CUDA
        # indexing/any() here used to synchronize the host on every action.
        key = (len(obs), agents, round_index, len(self.models), str(obs.device))
        if not hasattr(self, "_groups") or self._groups[0] != key:
            enabled = historical.detach().cpu().tolist()
            groups = [[] for _ in self.models]
            for index, selected in enumerate(enabled):
                if selected:
                    assignment = (index // agents + index % agents // 8 + round_index) % len(self.models)
                    groups[assignment].append(index)
            self._groups = key, [(i, torch.tensor(indices, device=obs.device, dtype=torch.long))
                                  for i, indices in enumerate(groups) if indices]
        for i, indices in self._groups[1]:
            logits, _ = self.models[i](obs.index_select(0, indices))
            chosen = Categorical(logits=logits, validate_args=False).sample()
            actions.index_copy_(0, indices, chosen)

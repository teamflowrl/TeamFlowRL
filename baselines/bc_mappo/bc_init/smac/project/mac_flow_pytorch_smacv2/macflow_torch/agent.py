from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
from torch.nn import functional as F

from .models import MACFlowConfig, MACFlowTorch, one_hot


class MACFlowLearner:
    def __init__(self, cfg: MACFlowConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = MACFlowTorch(cfg).to(device)
        self.model.requires_grad_(False)
        self.model.encoder.requires_grad_(True)
        self.model.actor_onestep_flow.requires_grad_(True)
        self.optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = self.model.encoder(batch['obs'])[:, :-1]
        actions = batch['actions'][:, :-1]
        legal = batch['legals'][:, :-1].bool()
        assert legal.any(-1).all() and legal.gather(-1, actions.unsqueeze(-1)).all()
        zeros = torch.zeros((*obs.shape[:-1], self.cfg.action_dim), device=obs.device)
        logits = self.model.actor_onestep_flow(obs, zeros).masked_fill(~legal, -1e9)
        # A forced single legal action contributes no policy choice or gradient.
        active = legal.sum(-1) > 1
        assert active.any()
        per_action = F.cross_entropy(logits.reshape(-1, self.cfg.action_dim),
                                     actions.reshape(-1), reduction='none').reshape_as(actions)
        loss = (per_action * active).sum() / active.sum()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 10.)
        self.optim.step()
        accuracy = ((logits.argmax(-1) == actions) * active).sum() / active.sum()
        return {'loss': float(loss.detach()), 'bc_loss': float(loss.detach()),
                'bc_acc': float(accuracy.detach()), 'active_samples': float(active.sum())}

    def save(self, path: str, extra: Dict | None = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "config": asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> Dict:
        payload = torch.load(path, map_location=map_location)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optim.load_state_dict(payload["optimizer"])
        return payload

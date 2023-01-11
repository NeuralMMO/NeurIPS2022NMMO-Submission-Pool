from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.mask import MaskedPolicy


class ActionHead(nn.Module):
    name2dim = {"move": 5, "attack_target": 16}

    def __init__(self, input_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(input_dim, output_dim)
            for name, output_dim in self.name2dim.items()
        })

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = {name: self.heads[name](x) for name in self.name2dim}
        return out


class NMMONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_map_cnn = nn.Sequential(
            nn.Conv2d(24, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.local_map_fc = nn.Linear(32 * 4 * 4, 64)

        self.self_entity_fc1 = nn.Linear(26, 32)
        self.self_entity_fc2 = nn.Linear(32, 32)

        self.other_entity_fc1 = nn.Linear(26, 32)
        self.other_entity_fc2 = nn.Linear(15 * 32, 32)

        self.fc = nn.Linear(64 + 32 + 32, 64)
        self.action_head = ActionHead(64)
        self.value_head = nn.Linear(64, 1)

    def local_map_embedding(self, input_dict):
        terrain = input_dict["terrain"]
        death_fog_damage = input_dict["death_fog_damage"]
        reachable = input_dict["reachable"]
        population = input_dict["entity_population"]

        T, B, *_ = terrain.shape

        terrain = F.one_hot(terrain, num_classes=16).permute(0, 1, 4, 2, 3)
        population = F.one_hot(population,
                               num_classes=6).permute(0, 1, 4, 2, 3)
        death_fog_damage = death_fog_damage.unsqueeze(dim=2)
        reachable = reachable.unsqueeze(dim=2)
        local_map = torch.cat(
            [terrain, reachable, population, death_fog_damage], dim=2)

        local_map = torch.flatten(local_map, 0, 1).to(torch.float32)
        local_map_emb = self.local_map_cnn(local_map)
        local_map_emb = local_map_emb.view(T * B, -1).view(T, B, -1)
        local_map_emb = F.relu(self.local_map_fc(local_map_emb))

        return local_map_emb

    def entity_embedding(self, input_dict):
        self_entity = input_dict["self_entity"]
        other_entity = input_dict["other_entity"]

        T, B, *_ = self_entity.shape

        self_entity_emb = F.relu(self.self_entity_fc1(self_entity))
        self_entity_emb = self_entity_emb.view(T, B, -1)
        self_entity_emb = F.relu(self.self_entity_fc2(self_entity_emb))

        other_entity_emb = F.relu(self.other_entity_fc1(other_entity))
        other_entity_emb = other_entity_emb.view(T, B, -1)
        other_entity_emb = F.relu(self.other_entity_fc2(other_entity_emb))

        return self_entity_emb, other_entity_emb

    def forward(
        self,
        input_dict: Dict,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        T, B, *_ = input_dict["terrain"].shape
        local_map_emb = self.local_map_embedding(input_dict)
        self_entity_emb, other_entity_emb = self.entity_embedding(input_dict)

        x = torch.cat([local_map_emb, self_entity_emb, other_entity_emb],
                      dim=-1)
        x = F.relu(self.fc(x))

        logits = self.action_head(x)
        value = self.value_head(x).view(T, B)

        output = {"value": value}
        for key, val in logits.items():
            if not training:
                dist = MaskedPolicy(val, input_dict[f"va_{key}"])
                action = dist.sample()
                logprob = dist.log_prob(action)
                output[key] = action
                output[f"{key}_logp"] = logprob
            else:
                output[f"{key}_logits"] = val
        return output

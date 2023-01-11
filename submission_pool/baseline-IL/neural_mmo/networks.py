from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    name2dim = {
        "attack_style": 4,      # "not attack" + 3 ( 0 indicates not attack)
        "attack_target": 101,   # "not attack" + 100 (0 indicates not attack)
        "direction": 5,         # "not move" + 4  (0 indicates not move)
        "buy_item": 171,        # "not buy" + 170 (0 indicates not buy)
        "sell_item": 13,       # "not sell" + 13 (0 indicates not sell)
        "sell_price": 101,        # "not sell" + 100  (0 indicates not sell)
        "use_item": 13,        # "not use" + 13 (0 indicates not use)
    }

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
        self.other_entity_fc = nn.Linear(15*26, 128)
        self.market_fc = nn.Linear(170*16, 256)
        
        # step:1, localmap:225*4+64
        # self-entity:26, other-entity:128, 
        # items:12*16, market:256
        self.fc = nn.Linear(1 + 225*4 + 64 + 26 + 128 + 12*16 + 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.action_head = ActionHead(256)

    def local_map_embedding(self, input_dict):
        terrain = input_dict["terrain"]
        death_fog_damage = input_dict["death_fog_damage"]
        reachable = input_dict["reachable"]
        population = input_dict["entity_population"]

        T, B, *_ = terrain.shape
        terrain_fc_emb = terrain.view(T, B, -1).float()
        fog_fc_emb = death_fog_damage.view(T, B, -1)
        reachable_fc_emb = reachable.view(T, B, -1).float()
        population_fc_emb = population.view(T, B, -1).float()
        local_map_blend_emb = torch.cat([
                                terrain_fc_emb,
                                fog_fc_emb,
                                reachable_fc_emb,
                                population_fc_emb], dim=-1)
         
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
        local_map_emb = torch.cat([local_map_emb, local_map_blend_emb], dim=-1)

        return local_map_emb

    def entity_embedding(self, input_dict):
        self_entity = input_dict["self_entity"]
        other_entity = input_dict["other_entity"]

        T, B, *_ = self_entity.shape
        self_entity_emb = self_entity.view(T, B, -1)
        other_entity_emb = F.relu(self.other_entity_fc(other_entity.view(T, B, -1)))

        return self_entity_emb, other_entity_emb

    def items_embedding(self, input_dict):
        items = input_dict["items"]
        T, B, *_ = items.shape
        items_embed = items.view(T, B, -1)
        
        return items_embed
        
    def market_embedding(self, input_dict):
        market = input_dict["market"]
        T, B, *_ = market.shape
        market_embed = F.relu(self.market_fc(market.view(T, B, -1)))

        return market_embed

    def forward(
        self,
        input_dict: Dict,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        T, B, *_ = input_dict["terrain"].shape
        step_emb = input_dict["step"]
        local_map_emb = self.local_map_embedding(input_dict)
        self_entity_emb, other_entity_emb = self.entity_embedding(input_dict)
        inventory_emb = self.items_embedding(input_dict)
        market_emb = self.market_embedding(input_dict)
        x = torch.cat([
                    step_emb,
                    local_map_emb, 
                    self_entity_emb, 
                    other_entity_emb, 
                    inventory_emb, 
                    market_emb
                    ], 
                    dim=-1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)

        output = {}
        for key, val in logits.items():
            output[f"{key}_logits"] = val
        
        return output

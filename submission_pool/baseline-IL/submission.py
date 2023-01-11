from pathlib import Path
from typing import Dict, List, Tuple, Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import tree

import nmmo
from nmmo import config
from neurips2022nmmo import Team
from neurips2022nmmo.scripted.scripted_team import RangeTeam
from neural_mmo import FeatureParser, NMMONet


def batch(
    obs: Dict[str, np.ndarray],
    filter_keys: List[str],
) -> Tuple[Dict[str, Tensor], List[int]]:
    """Transform agent-wise env_output to batch format."""
    filter_keys = list(filter_keys)
    obs_batch = {key: [] for key in filter_keys}
    agent_ids = []
    for agent_id, out in obs.items():
        agent_ids.append(agent_id)
        for key, val in out.items():
            if key in filter_keys:
                obs_batch[key].append(val)
    for key, val in obs_batch.items():
        obs_batch[key] = torch.cat(val, dim=1)

    return obs_batch, agent_ids


def unbatch(agent_output: Dict[str, Tensor], agent_ids: List[int]):
    """Transform agent_output to agent-wise format."""
    unbatched_agent_output = {key: {} for key in agent_ids}
    for key, val in agent_output.items():
        for i, agent_id in enumerate(agent_ids):
            unbatched_agent_output[agent_id][
                key] = val[:, i]  # shape: [1, B, ...]
    return unbatched_agent_output


class ImitationModel(Team):
    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self.model: nn.Module = NMMONet()
        if checkpoint_path is not None:
            print(f"load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
        self.feature_parser = FeatureParser()
        self.reset()

    def reset(self):
        self.range_team = RangeTeam("ScriptedTeam", self.env_config)
        self.step = 0

    @staticmethod
    def convert_to_action_format(outputs: Dict[int, Dict[str, Any]]):
        actions = {
            pid: {} for pid in outputs
        }
        
        for pid, out in outputs.items():
            direction = out["direction"].item()
            attack_style = out["attack_style"].item()
            attack_target = out["attack_target"].item()
            # buy_item = out[pid]["buy_item"].item()
            # sell_item = out[pid]["sell_item"].item()
            # sell_price = out[pid]["sell_price"].item()
            # use_item = out[pid]["use_item"].item()

            if direction != 0:
                actions[pid].update({
                    nmmo.action.Move: {
                        nmmo.action.Direction: direction - 1,
                    }
                })
            if attack_style!=0 and attack_target !=0:
                actions[pid].update({
                    nmmo.action.Attack: {
                        nmmo.action.Style: attack_style-1,
                        nmmo.action.Target: attack_target-1
                    }
                })
            # if buy_item != 0:
            #     actions[pid].update({
            #         nmmo.action.Buy: {
            #             nmmo.action.Item: buy_item - 1,
            #         }
            #     })
            # if sell_item!=0 and sell_price!=0:
            #     actions[pid].update({
            #         nmmo.action.Sell: {
            #             nmmo.action.Item: sell_item - 1,
            #             nmmo.action.Price: sell_price - 1
            #         }
            #     })
            # if use_item != 0:
            #     actions[pid].update({
            #         nmmo.action.Use: {
            #             nmmo.action.Item: use_item - 1,
            #         }
            #     })
                
        return actions

    def compute_nn_actions(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        feature = self.feature_parser.parse(observations, self.step)
        feature = tree.map_structure(
            lambda x: torch.from_numpy(x).view(1, 1, *x.shape), 
            feature
        )
        feature_batch, agent_ids = batch(feature, self.feature_parser.spec.keys())
        
        # Inference
        output = self.model(feature_batch, training=False)
        logits_keys = list(output.keys())
        for key in logits_keys:
            argmax_key = key[:-7]
            output[argmax_key] = torch.argmax(output[key], dim=-1)
        output = unbatch(output, agent_ids)
        
        actions = ImitationModel.convert_to_action_format(output)
        
        return actions

    def act(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        self.step += 1
        if "stat" in observations:
            stat = observations.pop("stat")
        
        # rule-based actions
        scripted_actions = self.range_team.act(deepcopy(observations))
        
        # neural-based actions
        nn_actions = self.compute_nn_actions(observations)
        
        # merge actions above
        merged_actions = deepcopy(scripted_actions)
        for pid in nn_actions:
            merged_actions[pid].update(nn_actions[pid])
        
        return merged_actions


class Submission:
    team_klass = ImitationModel
    init_params = {
        "checkpoint_path":
        Path(__file__).parent / "checkpoints" / "model.pth"
    }

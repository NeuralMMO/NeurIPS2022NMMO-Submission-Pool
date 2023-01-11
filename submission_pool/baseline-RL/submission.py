from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import tree
from neurips2022nmmo import Team
from nmmo import config

from monobeast import batch, unbatch
from neural_mmo import FeatureParser, MyMeleeTeam, NMMONet, TrainEnv


class MonobeastBaseline(Team):
    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self.model: nn.Module = NMMONet()
        if checkpoint_path is not None:
            print(f"load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_parser = FeatureParser()
        self.reset()

    def reset(self):
        self.my_script = {0: MyMeleeTeam("MyMelee", self.env_config)}
        self.step = 0

    def compute_actions(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        feature = self.feature_parser.parse(observations, self.step)
        feature = tree.map_structure(
            lambda x: torch.from_numpy(x).view(1, 1, *x.shape), feature)
        feature_batch, ids = batch(feature, self.feature_parser.spec.keys())
        output = self.model(feature_batch, training=False)
        output = unbatch(output, ids)

        actions = {}
        for i, out in output.items():
            actions[i] = {
                "move": out["move"].item(),
                "attack_target": out["attack_target"].item()
            }

        actions = TrainEnv.transform_action({0: actions}, {0: observations},
                                            self.my_script)
        return actions[0]

    def act(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        self.step += 1
        if "stat" in observations:
            stat = observations.pop("stat")
        actions = self.compute_actions(observations)
        return actions


class Submission:
    team_klass = MonobeastBaseline
    init_params = {
        "checkpoint_path":
        Path(__file__).parent / "checkpoints" / "model_2757376.pt"
    }

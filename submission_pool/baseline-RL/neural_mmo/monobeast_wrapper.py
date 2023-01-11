"""The environment class for MonoBeast."""
from typing import Dict

import torch
import tree
from numpy import ndarray

from neural_mmo.train_wrapper import TrainEnv


def to_tensor(x):
    if isinstance(x, ndarray):
        return torch.from_numpy(x).view(1, 1, *x.shape)
    elif isinstance(x, (int, float)):
        return torch.tensor(x).view(1, 1)
    else:
        raise RuntimeError


class MonobeastEnv:
    def __init__(self, env: TrainEnv):
        if not isinstance(env, TrainEnv):
            raise RuntimeError(f"env is not instance of TrainEnv.")
        self.env = env

    def initial(self):
        obs = self.env.reset()
        obs = tree.map_structure(to_tensor, obs)
        self._info = self.reset_info()
        return obs

    def step(self, actions: Dict[int, int]):
        obs, reward, done, info = self.env.step(actions)
        for agent_id in done.keys():
            if info[agent_id]["mask"]:
                self._info[agent_id]["episode_step"] += 1
            self._info[agent_id]["episode_return"] += reward[agent_id]
            self._info[agent_id].update(info[agent_id])
        info_ = self._info
        if all(done.values()):
            obs = self.env.reset()
            self._info = self.reset_info()

        obs = tree.map_structure(to_tensor, obs)
        reward = tree.map_structure(to_tensor, reward)
        done = tree.map_structure(to_tensor, done)
        return obs, reward, done, info_

    def close(self):
        self.env.close()

    def reset_info(self):
        info = {
            agent_id: {
                "episode_step": 0,
                "episode_return": 0,
            }
            for agent_id in self.env.agents
        }
        return info

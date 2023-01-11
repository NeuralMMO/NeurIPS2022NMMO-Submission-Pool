import logging
from copy import deepcopy
from typing import Any, Dict, Tuple

import nmmo
import numpy as np
from gym import Wrapper, spaces
from neurips2022nmmo import CompetitionConfig, TeamBasedEnv, scripted
from neurips2022nmmo.scripted import Melee
from neurips2022nmmo.scripted.baselines import Combat
from neurips2022nmmo.scripted.scripted_team import ScriptedTeam
from numpy import ndarray

from neural_mmo import FeatureParser, RewardParser


class TrainEnv(Wrapper):
    max_step = 1024
    num_team_member = 8
    opponent_pool = [scripted.MixtureTeam, scripted.CombatTeam]

    def __init__(
        self,
        env: TeamBasedEnv,
        num_selfplay_team: int = 1,
        reward_setting: str = "phase1",
    ):
        logging.info(
            f"num_selfplay_team: {num_selfplay_team}, reward_setting: {reward_setting}"
        )
        super().__init__(env)
        self.num_selfplay_team = num_selfplay_team
        self.feature_parser = FeatureParser()
        self.reward_parser = RewardParser(reward_setting)
        self._setup()

    def _setup(self):
        observation_space, action_space, dummy_feature = {}, {}, {}
        for key, val in self.feature_parser.spec.items():
            observation_space[key] = val
            dummy_feature[key] = np.zeros(shape=val.shape, dtype=val.dtype)
            if key.startswith("va_"):
                key_ = key.replace("va_", "")
                action_space[key_] = spaces.Discrete(val.shape[0])
        self.observation_space = spaces.Dict(observation_space)
        self.action_space = spaces.Dict(action_space)
        self._dummy_feature = dummy_feature

    def reset(self) -> Dict[int, Dict[str, ndarray]]:
        self._step = 0
        self.reward_parser.reset()
        self.reset_arena(self.config)

        raw_obs = super().reset()
        obs = self._flatten(self._get(raw_obs))
        obs = self.feature_parser.parse(obs, self._step)
        metrics = self._flatten(self._get(self.metrices_by_team()))

        self.agents = list(obs.keys())
        self._prev_raw_obs = raw_obs
        self._prev_metrics = metrics
        return obs

    def step(self, actions: Dict[int, Dict[str, int]]):
        self._step += 1

        # preprocess action
        decisions = self.get_opponent_decision(self._prev_raw_obs)
        actions = self.transform_action(
            self._unflatten(actions),
            observations=self._prev_raw_obs,
            my_script=self.myself,
        )
        decisions.update(actions)

        # step
        raw_obs, _, raw_done, _ = super().step(decisions)

        obs = self._flatten(self._get(raw_obs))
        obs = self.feature_parser.parse(obs, self._step)
        done = self._flatten(self._get(raw_done))
        metrics = self._flatten(self._get(self.metrices_by_team()))
        reward = self.reward_parser.parse(self._prev_metrics, metrics, obs,
                                          self._step, done)

        self._prev_raw_obs = raw_obs
        self._prev_metrics = metrics

        # padding
        info = {uid: {} for uid in self.agents}
        for uid in self.agents:
            if uid not in obs:
                obs[uid] = self._dummy_feature
            if uid not in done:
                reward[uid] = 0
                done[uid] = True
                info[uid]["mask"] = False
            else:
                info[uid]["mask"] = True

        if self._step >= self.max_step:
            done = {uid: True for uid in self.agents}

        return obs, reward, done, info

    def _get(
        self,
        xs: Dict[int, Dict[int, Any]],
    ) -> Dict[int, Dict[int, Dict]]:
        ret = {}
        for tid in range(self.num_selfplay_team):
            if tid in xs:
                ret[tid] = xs[tid]
        return ret

    def _flatten(self, xs: Dict[int, Dict[int, Any]]) -> Dict[int, Any]:
        ret = {}
        for tid in xs:
            for pid in xs[tid]:
                if pid == "stat": continue
                uid = self._tidpid2uid(tid, pid)
                ret[uid] = xs[tid][pid]
        return ret

    def _unflatten(self, xs: Dict[int, Any]) -> Dict[int, Dict[int, Any]]:
        ret = {}
        for uid in xs:
            tid, pid = self._uid2tidpid(uid)
            if tid not in ret:
                ret[tid] = {}
            ret[tid][pid] = xs[uid]
        return ret

    @classmethod
    def _tidpid2uid(cls, tid: int, pid: int) -> int:
        return tid * cls.num_team_member + pid

    @classmethod
    def _uid2tidpid(cls, uid: int) -> Tuple[int, int]:
        tid = uid // cls.num_team_member
        pid = uid % cls.num_team_member
        return tid, pid

    def reset_arena(self, config: CompetitionConfig):
        # myself
        if getattr(self, "myself", None):
            [s.reset() for s in self.myself.values()]
        else:
            self.myself = {
                i: MyMeleeTeam(f"MyMelee-{i}", config)
                for i in range(self.num_selfplay_team)
            }
        # opponent
        if getattr(self, "_opponent", None):
            [o.reset() for o in self._opponent.values()]
        else:
            self._opponent = {}
            for i in range(self.num_selfplay_team, len(config.PLAYERS)):
                klass = np.random.choice(self.opponent_pool)
                self._opponent[i] = klass(f"{klass.__name__}-{i}", config)

    def get_opponent_decision(
        self,
        observations: Dict[int, Dict[int, Dict]],
    ) -> Dict[int, Dict[int, int]]:
        decisions = {}
        for tid, obs in observations.items():
            if tid in self._opponent:
                decisions[tid] = self._opponent[tid].act(obs)
        return decisions

    @staticmethod
    def transform_action(actions: Dict[int, Dict[int, Any]],
                         observations: Dict[int, Dict[int, Any]],
                         my_script=None):
        # remove actions of dead agents
        original_actions = deepcopy(actions)
        for tid in original_actions:
            if my_script is not None:
                assert tid in my_script
            if tid not in observations:
                actions.pop(tid)
                continue
            for pid in original_actions[tid]:
                if pid not in observations[tid]:
                    actions[tid].pop(pid)

        if my_script is not None:
            decisions = {
                tid: my_script[tid].act(observations[tid])
                for tid in actions
            }
        else:
            decisions = {
                tid: {pid: {}
                      for pid in actions[pid]}
                for tid in actions
            }

        for tid in actions:
            for pid in actions[tid]:
                move = actions[tid][pid]["move"]
                attack_target = actions[tid][pid]["attack_target"]
                decisions[tid][pid].update({
                    nmmo.action.Attack: {
                        nmmo.action.Style: 0,
                        nmmo.action.Target: attack_target
                    }
                })
                if move != 0:
                    decisions[tid][pid].update({
                        nmmo.action.Move: {
                            nmmo.action.Direction: move - 1,
                        }
                    })
        return decisions


class MyMelee(Melee):
    name = 'MyMelee'

    def __call__(self, obs):
        super(Combat, self).__call__(obs)
        self.use()
        self.exchange()
        assert nmmo.action.Move not in self.actions
        assert nmmo.action.Attack not in self.actions
        return self.actions


class MyMeleeTeam(ScriptedTeam):
    agent_klass = [MyMelee]


class TrainConfig(CompetitionConfig):
    MAP_N = 400

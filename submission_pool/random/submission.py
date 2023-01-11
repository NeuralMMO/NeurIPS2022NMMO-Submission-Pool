import random
from nmmo.io import action

from neurips2022nmmo import Team


class RandomTeam(Team):

    def act(self, observations):
        actions = {}
        if "stat" in observations:
            observations.pop("stat")
        for player_idx, obs in observations.items():
            actions[player_idx] = {
                action.Move: {
                    action.Direction:
                    random.randint(0,
                                   len(action.Direction.edges) - 1)
                }
            }
        return actions


class Submission:
    team_klass = RandomTeam
    init_params = {}

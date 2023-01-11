from pathlib import Path

from neurips2022nmmo import CompetitionConfig, RollOut, scripted

from submission import MonobeastBaseline


def rollout():
    config = CompetitionConfig()
    config.RENDER = False
    config.SAVE_REPLAY = "eval"
    checkpoint_path = Path(
        __file__).parent / "checkpoints" / "model_2757376.pt"
    my_team = MonobeastBaseline(team_id=f"my-team",
                                env_config=config,
                                checkpoint_path=checkpoint_path)
    all_teams = [scripted.CombatTeam(f"C-{i}", config) for i in range(5)]
    all_teams.extend(
        [scripted.MixtureTeam(f"M-{i}", config) for i in range(10)])
    all_teams.append(my_team)
    ro = RollOut(config, all_teams, parallel=False)
    ro.run(n_episode=1, render=config.RENDER)


if __name__ == "__main__":
    rollout()

import argparse

import pandas as pd
from matplotlib import pyplot as plt

STATS = [
    "total_loss",
    "mean_episode_return",
    "mean_episode_step",
    "max_episode_step",
    "min_episode_step",
    "valid_data_frac",
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "policy_clip_frac",
    "advantage",
    "grad_norm",
]
N_COLUMN = 3


def plot(log_file, window=50):
    log_df = pd.read_csv(log_file, sep=",")
    for key in STATS + ["step"]:
        assert key in log_df.columns
    fig, axes = plt.subplots(len(STATS) // N_COLUMN, N_COLUMN)
    fig.suptitle("Learning curves on NEURIPS2022-NMMO", fontsize=20)
    fig.set_size_inches(18, 15)
    for i, key in enumerate(STATS):
        df = log_df[["step", key]].dropna()
        df = df.rolling(window).mean()
        ax = axes[i // N_COLUMN, i % N_COLUMN]
        ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='x')
        ax.set_title(key)
        ax.plot(df["step"], df[key])
    plt.savefig("plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        type=str,
        default="results/nmmo/logs.csv",
    )
    args = parser.parse_args()
    plot(args.logfile)

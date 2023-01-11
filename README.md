# Introduction
This repository contains some top solutions for the NeurIPS2022 Neural MMO challenge and supports selecting several submissions to start a match.

NeurIPS2022 Neural MMO challenge: https://www.aicrowd.com/challenges/neurips-2022-the-neural-mmo-challenge

# Install
Considering that the dependencies required by different submissions may conflict, we use containers to run them independently. 
Therefore, make sure you have installed [docker](https://www.docker.com/).

```bash
git clone https://github.com/bowuaaa/NeurIPS2022NMMO-Submission-Pool.git
conda create -n neurips2022-nmmo python==3.9
conda activate neurips2022-nmmo
cd ./NeurIPS2022NMMO-Submission-Pool
pip install -r requirements.txt
```

# Run
You can specify the submission to participate in the match by modifying `config.py`. For example,  if you want the 16 teams in a match to be: 8 baseline-RL + 8 random, then you can modify `config.py` as follow:
```python
participants = [
    "baseline-RL",
    "baseline-RL",
    "baseline-RL",
    "baseline-RL",
    "baseline-RL",
    "baseline-RL",
    "baseline-RL",
    "baseline-RL",
    "random",
    "random",
    "random",
    "random",
    "random",
    "random",
    "random",
    "random",
]
```

Then
```bash
python evaluate.py run
```

After a few minutes, you will see the final results.

# Contributing
Please see [`CONTRIBUTING.md`](CONTRIBUTING.md).


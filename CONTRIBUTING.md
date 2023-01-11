# Install
Considering that the dependencies required by different submissions may conflict, we use containers to run them independently. Make sure you have installed [docker](https://www.docker.com/).

```bash
git clone https://github.com/bowuaaa/NeurIPS2022NMMO-Submission-Pool.git
conda create -n neurips2022-nmmo python==3.9
conda activate neurips2022-nmmo
cd ./NeurIPS2022NMMO-Submission-Pool
pip install -r requirements.txt
```

# Add your submission into the pool
1. Make your own submission folder, e.g. `submission_pool/<your_team_name>/`
2. Put all the files under the `my-submission/` folder of **your starterkit repository** into the `submission_pool/<your_team_name>/`
3. If you have other third-party dependencies, you need to put your `requirements.txt` in the `submission_pool/<your_team_name>/`

A qualified submission structure is as follows:
```
- submission_pool/
    - your_team_name/           # Directory containing your submission.
        | - other_files         # All others files needed by your submission.
        | - submission.py       # Entrypoint of your submission.
        | - requirements.txt    # (Optinal) Python requirements for your submission.
```

# Test your submission
```bash
python evaluate.py test <your_team_name>
```
If all goes well, you will see "TEST PASS".

# Make a PR
Make a pull request, and we will merge your submission as soon as possible.

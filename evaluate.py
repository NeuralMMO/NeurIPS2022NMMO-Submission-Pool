#!/bin/sh
''''exec python -u "$0" ${1+"$@"} # '''

import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from subprocess import CalledProcessError
from typing import List

import termcolor
from neurips2022nmmo import submission as subm

IMAGE = "neurips2022nmmo/submission-runtime"

log = Path(__file__).parent / "debug.log"
log.unlink(missing_ok=True)


def ok(msg: str):
    print(termcolor.colored(msg, "green", attrs=['bold']))


def warn(msg: str):
    print(termcolor.colored(msg, "yellow", attrs=['bold']))


def err(msg: str):
    print(termcolor.colored(msg, "red", attrs=['bold']))


def run_team_server(submission_path: str, port: int):
    subm.check(submission_path)
    team_klass, init_params = subm.parse_submission(submission_path)
    print(f"Start TeamServer for {team_klass.__name__}")
    from neurips2022nmmo import TeamServer
    server = TeamServer("0.0.0.0", port, team_klass, init_params)
    server.run()


def run_submission_in_docker(submission_path: str, port: int,
                             container_name: str):

    full_submission_path = f"submission_pool/{submission_path}"

    need_root = True if os.system("docker ps 1>/dev/null 2>&1") else False

    def _shell(command, capture_output=False, print_command=True):
        log_fp = open("debug.log", "a")

        if command.startswith("docker") and need_root:
            command = "sudo " + command
        if print_command:
            print(command, file=log_fp)

        stdout, stderr = log_fp, log_fp
        if capture_output:
            stdout, stderr = None, None
        r = subprocess.run(command,
                           shell=True,
                           capture_output=capture_output,
                           stdout=stdout,
                           stderr=stderr)
        log_fp.close()
        if not capture_output: return r.returncode
        if r.returncode != 0:
            # grep return 1 when no lines matching
            if r.returncode == 1 and "grep" in command:
                pass
            else:
                raise CalledProcessError(r.returncode, r.args, r.stdout,
                                         r.stderr)

        return r.stdout.decode().strip()

    if _shell(
            f"docker build --network=host --build-arg submission={full_submission_path} -t {IMAGE}:{submission_path} -f Dockerfile ."
    ):
        err("Build failed.")
        sys.exit(10)
    if _shell(f'docker ps -a | grep -w "{container_name}"',
              capture_output=True) != "":
        _shell(f"docker stop {container_name}")
        _shell(f"docker rm {container_name}")
    command = f"python evaluate.py run_team_server --submission={full_submission_path} --port={port}"
    container_id = _shell(
        f"docker run -it -d --rm --name {container_name} -p {port}:{port} {IMAGE}:{submission_path} {command}",
        capture_output=True)

    threading.Thread(target=_shell,
                     args=(f"docker logs -f {container_id}", False),
                     daemon=True).start()

    def _check_container_alive():
        while 1:
            ret = _shell(
                f"docker inspect {container_id} --format='{{{{.State.ExitCode}}}}'",
                capture_output=True,
                print_command=False)
            if ret != "0":
                err(f"Container {container_id} exit unexpectedly")
                os._exit(1)
            time.sleep(1)

    threading.Thread(target=_check_container_alive, daemon=True).start()

    return container_id


def rollout(submissions: List[str]):
    from neurips2022nmmo import CompetitionConfig

    class Config(CompetitionConfig):
        SAVE_REPLAY = "-".join(map(lambda x: x.replace("-", ""), submissions))

    teams = []
    for i, submission_path in enumerate(submissions):
        port = 5000 + i
        team_name = f"{submission_path}-{i}"
        ok(f"Preparing team [{team_name}] ...")
        container_id = run_submission_in_docker(submission_path, port,
                                                team_name)
        ok(f"Team [{team_name}] is running in container {container_id}")

        from neurips2022nmmo import ProxyTeam
        team = ProxyTeam(team_name, Config(), "127.0.0.1", port=port)
        teams.append(team)

    try:
        from neurips2022nmmo import RollOut
        ro = RollOut(Config(), teams, True)
        ro.run()
    except:
        raise
    finally:
        [team.stop() for team in teams]


def test(submission_path: str):
    from neurips2022nmmo import CompetitionConfig

    class Config(CompetitionConfig):
        MAP_N = 1

    port = 5000
    ok(f"Preparing submission [{submission_path}] ...")
    container_id = run_submission_in_docker(submission_path, port,
                                            f"{submission_path}-test")
    ok(f"Submission [{submission_path}] is running in container {container_id}"
       )

    from neurips2022nmmo import ProxyTeam
    team = ProxyTeam(submission_path, Config(), "127.0.0.1", port=port)
    try:
        from neurips2022nmmo import RollOut, scripted
        ro = RollOut(
            Config(),
            [
                scripted.RandomTeam(f"random-{i}", Config())
                for i in range(len(CompetitionConfig.PLAYERS) - 1)
            ] + [team],
            True,
        )
        ro.run()
    except:
        raise
    finally:
        team.stop()


class Toolkit:

    def run(self):
        r"""
        Start a match using the submissions specified in config.py
        """

        from config import participants
        assert len(participants) == 16, f"A Match requires 16 teams"
        ok(f"Starting match: {participants}")

        from art import text2art
        try:
            rollout(participants)
        except:
            traceback.print_exc()
            err(text2art("FAIL", "sub-zero"))
            sys.exit(1)
        else:
            ok(text2art("FINISH", "sub-zero"))

    def test(self, submission_path: str):
        r"""
        Test submission
        """

        ok(f"Testing submission: {submission_path}")
        from art import text2art
        try:
            test(submission_path)
        except:
            traceback.print_exc()
            err(text2art("TEST FAIL", "sub-zero"))
            sys.exit(1)
        else:
            ok(text2art("TEST PASS", "sub-zero"))

    def run_team_server(self, submission: str, port: int):
        run_team_server(submission, port)


if __name__ == "__main__":
    import fire
    fire.Fire(Toolkit)

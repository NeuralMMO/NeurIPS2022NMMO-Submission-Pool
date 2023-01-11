import argparse
import logging
import pprint
import threading
import time
import timeit
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
from gym import spaces
from neurips2022nmmo import TeamBasedEnv
from torch import Tensor
from torch import multiprocessing as mp
from torch import nn

from core import file_writer, loss, prof, advantage
from neural_mmo import MonobeastEnv as Environment
from neural_mmo import NMMONet, TrainConfig, TrainEnv

to_torch_dtype = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")
parser.add_argument("--env", type=str, default="Neurips2022-NMMO",
                    help="Gym environment.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/monobeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--checkpoint_interval", default=600, type=int, metavar="T",
                    help="Checkpoint interval (default: 10min).")
parser.add_argument("--restart_actor_interval", default=18000, type=int, metavar="T",
                    help="Restart actor interval (default: 5h).")
parser.add_argument("--checkpoint_path", default=None, type=str,
                    help="Load previous checkpoint to continue training")
parser.add_argument("--num_selfplay_team", default=1, type=int, metavar="T",
                    help="Number of self-play team (default: 1).")
parser.add_argument("--data_reuse", default=4, type=int, metavar="T",
                    help="Data reuse(default: 4).")
parser.add_argument("--reward_setting", default="phase1", type=str,
                    help="Reward setting.")

# Loss settings.
parser.add_argument("--upgo_coef", default=0.5,
                    type=float, help="Upgo coefficient.")
parser.add_argument("--entropy_coef", default=0.0006,
                    type=float, help="Entropy coefficient.")
parser.add_argument("--value_coef", default=0.5,
                    type=float, help="value coefficient.")
parser.add_argument("--clip_ratio", default=0.2,
                    type=float, help="Clip ratio.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"),
    level=logging.INFO,
)

Buffers = Dict[str, List[torch.Tensor]]
Net = NMMONet


def create_env(flags):
    env = TeamBasedEnv(config=TrainConfig())
    return TrainEnv(
        env,
        num_selfplay_team=flags.num_selfplay_team,
        reward_setting=flags.reward_setting,
    )


def create_buffers(
    flags,
    observation_space: spaces.Dict,
    action_space: spaces.Dict,
) -> Buffers:
    T = flags.unroll_length
    obs_specs = {
        key: dict(size=(T + 1, *val.shape),
                  dtype=to_torch_dtype[val.dtype.name])
        for key, val in observation_space.items()
    }
    action_specs = {}
    for key, val in action_space.items():
        action_specs[key] = dict(size=(T, ), dtype=torch.int64)
        action_specs[f"{key}_logp"] = dict(size=(T, ), dtype=torch.float32)
    specs = dict(
        reward=dict(size=(T, ), dtype=torch.float32),
        done=dict(size=(T, ), dtype=torch.bool),
        mask=dict(size=(T, ), dtype=torch.bool),
        episode_return=dict(size=(T, ), dtype=torch.float32),
        episode_step=dict(size=(T, ), dtype=torch.int32),
        value=dict(size=(T, ), dtype=torch.float32),
    )
    specs.update(obs_specs)
    specs.update(action_specs)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.zeros(**specs[key]).share_memory_())
    return buffers


def store(
    buffers: Buffers,
    free_indices: List[int],
    t: int,
    obs: Dict[int, Dict],
    agent_output: Dict[int, Dict],
    reward: Dict[int, float],
    done: Dict[int, float],
    info: Dict[int, Dict],
):
    indices_iter = iter(free_indices)
    """Store tensor in buffer."""
    for agent_id in obs.keys():
        index = next(indices_iter)
        for key, val in obs[agent_id].items():
            buffers[key][index][t, ...] = val
        if reward is not None:
            buffers["reward"][index][t, ...] = reward[agent_id]
        if done is not None:
            buffers["done"][index][t, ...] = done[agent_id]
        if info is not None:
            for key, val in info[agent_id].items():
                buffers[key][index][t, ...] = val
        if agent_output is not None:
            for key, val in agent_output[agent_id].items():
                buffers[key][index][t, ...] = val


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


@torch.no_grad()
def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
):
    try:
        logging.info(f"Actor {actor_index} started.")
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        env = Environment(gym_env)
        observation_space: spaces.Dict = gym_env.observation_space
        action_space: spaces.Dict = gym_env.action_space
        obs = env.initial()
        while True:
            free_indices = [free_queue.get() for _ in range(flags.num_agents)]
            if None in free_indices:
                break
            # rollout.
            for t in range(flags.unroll_length):
                timings.reset()
                # batch
                obs_batch, agent_ids = batch(
                    obs, filter_keys=observation_space.keys())
                # forward inference
                agent_output_batch = model(obs_batch)
                # unbatch
                agent_output = unbatch(agent_output_batch, agent_ids)
                # extract actions
                actions = {
                    agent_id: {
                        key: agent_output[agent_id][key].item()
                        for key in action_space.keys()
                    }
                    for agent_id in agent_output
                }

                timings.time("model")
                next_obs, reward, done, info = env.step(actions)
                timings.time("step")

                store(
                    buffers=buffers,
                    free_indices=free_indices,
                    t=t,
                    obs=obs,
                    agent_output=agent_output,
                    reward=reward,
                    done=done,
                    info=info,
                )
                timings.time("write")

                obs = next_obs

            store(
                buffers=buffers,
                free_indices=free_indices,
                t=flags.unroll_length,
                obs=obs,
                agent_output=None,
                reward=None,
                done=None,
                info=None,
            )

            for index in free_indices:
                full_queue.put(index)

        if actor_index == 0:
            logging.info(f"Actor {actor_index}: {timings.summary()}")

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error(f"Exception in worker process {actor_index}")
        traceback.print_exc()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings: prof.Timings,
) -> Dict[str, Tensor]:
    indices = [full_queue.get() for _ in range(flags.batch_size)]
    timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device) for k, t in batch.items()}
    timings.time("device")
    return batch


def learn(
    flags,
    actor_model: nn.Module,
    learner_model: nn.Module,
    batch: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Performs a learning (optimization) step."""
    learner_outputs = learner_model(batch, training=True)
    # Take final value function slice for bootstrapping.
    bootstrap_value = learner_outputs["value"][-1]
    value = learner_outputs["value"][:-1]

    # Move from obs[t] -> action[t] to action[t] -> obs[t].
    learner_outputs = {
        key: tensor[:-1]
        for key, tensor in learner_outputs.items()
    }

    reward = batch["reward"]
    discount = (~batch["done"]).float() * flags.discounting
    mask = batch["mask"].float()  # mask dead agent
    logits = []
    actions = []
    valid_actions = []
    behaviour_policy_logprobs = []
    for key in learner_outputs.keys():
        if key.endswith("_logits"):
            k = key.replace("_logits", "")
            logits.append(learner_outputs[key])
            actions.append(batch[k])
            behaviour_policy_logprobs.append(batch[f"{k}_logp"])
            valid_actions.append(batch[f"va_{k}"][:-1])

    gae_returns = advantage.gae(
        value=batch["value"],
        reward=reward,
        bootstrap_value=bootstrap_value,
        discount=discount,
        lambda_=1.0,
        mask=mask,
    )
    upgo_returns = advantage.upgo(
        value=batch["value"],
        reward=reward,
        bootstrap_value=bootstrap_value,
        discount=discount,
        mask=mask,
    )

    for i in range(flags.data_reuse):
        if i > 0:
            learner_outputs = learner_model(batch, training=True)
            action_names = learner_model.action_head.name2dim.keys()
            logits = [
                learner_outputs[f"{key}_logits"][:-1] for key in action_names
            ]
            value = learner_outputs["value"][:-1]

        total_loss, extra = loss.compute_ppo_loss(
            logits=logits,
            actions=actions,
            value=value,
            target_value=gae_returns.vs,
            behaviour_policy_logprobs=behaviour_policy_logprobs,
            advantage=gae_returns.advantages,
            upgo_advantage=upgo_returns.advantages,
            valid_actions=valid_actions,
            entropy_coef=flags.entropy_coef,
            value_coef=flags.value_coef,
            upgo_coef=flags.upgo_coef,
            clip_ratio=flags.clip_ratio,
            mask=mask,
        )
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(learner_model.parameters(),
                                             flags.grad_norm_clipping)
        optimizer.step()
    actor_model.load_state_dict(learner_model.state_dict())
    episode_end = (batch["done"] == True) & (batch["mask"] == True)
    episode_returns = batch["episode_return"][episode_end]
    episode_steps = batch["episode_step"][episode_end]

    def _reduce(func: Callable, x: torch.Tensor):
        if x.nelement() == 0:
            return float("nan")
        else:
            x = x.float()
            return func(x).item()

    # yapf: disable
    stats = {
        "mean_episode_return": _reduce(torch.mean, episode_returns),
        "mean_episode_step": _reduce(torch.mean, episode_steps),
        "max_episode_step": _reduce(torch.max, episode_steps),
        "min_episode_step": _reduce(torch.min, episode_steps),
        "total_loss": total_loss.item(),
        "advantage": (torch.sum(gae_returns.advantages * mask) / torch.sum(mask)).item(),
        "valid_data_frac": torch.mean(mask).item(),
        "grad_norm": grad_norm.item(),
        **extra,
    }
    # yapf: enable

    return stats


def start_process(
    flags,
    ctx,
    model: nn.Module,
    actor_processes: List[mp.Process],
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
):
    """Periodically restart actor process to prevent OOM, which may be caused by pytorch share_memory"""
    if len(actor_processes) > 0:
        logging.critical("Stoping actor process...")
        for actor in actor_processes:
            actor.terminate()
            actor.join()
            actor.close()

    while not free_queue.empty():
        free_queue.get()
    while not full_queue.empty():
        full_queue.get()
    for m in range(flags.num_buffers):
        free_queue.put(m)

    logging.critical("Starting actor process...")
    actor_processes = []
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)
    return actor_processes


def checkpoint(flags, model: nn.Module, step: int):
    if flags.disable_checkpoint:
        return
    checkpointpath = Path(flags.savedir).joinpath(flags.xpid)
    logging.info(f"Saving checkpoint to {checkpointpath}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "flags": vars(flags),
        },
        checkpointpath.joinpath(f"model_{step}.pt"),
    )


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = f"monobeast-{time.strftime('%Y%m%d-%H%M%S')}"
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
        symlink_to_latest=False,
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = int(
            max(2 * flags.num_agents * flags.num_actors, flags.batch_size))
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    actor_model = Net()
    learner_model = Net().to(device=flags.device)
    if flags.checkpoint_path is not None:
        logging.info(f"load checkpoint: {flags.checkpoint_path}")
        previous_checkpoint = torch.load(flags.checkpoint_path)
        learner_model.load_state_dict(previous_checkpoint["model_state_dict"])
    actor_model.share_memory()
    actor_model.load_state_dict(learner_model.state_dict())

    buffers = create_buffers(flags, env.observation_space, env.action_space)

    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    actor_processes = start_process(
        flags,
        ctx,
        actor_model,
        [],
        free_queue,
        full_queue,
        buffers,
    )
    optimizer = torch.optim.Adam(
        learner_model.parameters(),
        lr=flags.learning_rate,
    )

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "mean_episode_step",
        "max_episode_step",
        "min_episode_step",
        "policy_loss",
        "upgo_loss",
        "value_loss",
        "entropy_loss",
        "advantage",
        "policy_clip_frac",
        "grad_norm",
        "valid_data_frac",
    ]
    logger.info("# Step\t{}".format("\t".join(stat_keys)))

    step, stats = 0, {}

    def batch_and_learn():
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            # start = time.time()
            batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            if torch.sum(batch["mask"]) == 0:
                continue
            stats = learn(flags, actor_model, learner_model, batch, optimizer)
            timings.time("learn")
            to_log = dict(step=step)
            to_log.update({k: stats[k] for k in stat_keys})
            plogger.log(to_log)
            step += int(T * B / flags.num_agents)

        logging.info(f"Batch and learn: {timings.summary()}")

    learner_thread = threading.Thread(target=batch_and_learn,
                                      name="batch-and-learn")
    learner_thread.start()

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        last_restart_actor_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(30)

            if timer() - last_checkpoint_time > flags.checkpoint_interval:
                checkpoint(flags, learner_model, step)
                last_checkpoint_time = timer()

            if timer() - last_restart_actor_time > flags.restart_actor_interval: # yapf: disable
                actor_processes = start_process(
                    flags,
                    ctx,
                    actor_model,
                    actor_processes,
                    free_queue,
                    full_queue,
                    buffers,
                )
                last_restart_actor_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = f"Return per episode: {stats['mean_episode_return']:.1f}. "
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("nan"))
            logging.info(
                f"Steps {step} @ {sps:.1f} SPS. Loss {total_loss}. {mean_return}Stats:\n{pprint.pformat(stats)}"
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        learner_thread.join()
        logging.info(f"Learning after {step} steps.")
    finally:
        for _ in range(flags.num_actors * flags.num_agents):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(flags, learner_model, step)
    plogger.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    flags = parser.parse_args()
    flags.num_agents = flags.num_selfplay_team * TrainEnv.num_team_member
    train(flags)

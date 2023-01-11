from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from core.mask import MaskedPolicy


def compute_value_loss(x: Tensor, y: Tensor, mask: Tensor = None):
    if mask is None:
        mask = torch.ones_like(x)

    loss = F.mse_loss(x, y, reduction='none') * mask
    loss = torch.sum(loss) / torch.sum(mask)
    return loss


def compute_ppo_loss(
    logits: List[Tensor],
    actions: List[Tensor],
    value: Tensor,
    target_value: Tensor,
    behaviour_policy_logprobs: List[Tensor],
    advantage: Tensor,
    upgo_advantage: Tensor,
    valid_actions: List[Tensor] = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    upgo_coef: float = 0.2,
    clip_ratio: float = 0.2,
    mask: Tensor = None,
):
    device = advantage.device
    policy_loss = torch.tensor(0, dtype=torch.float32, device=device)
    upgo_loss = torch.tensor(0, dtype=torch.float32, device=device)
    entropy_loss = torch.tensor(0, dtype=torch.float32, device=device)
    policy_clip_frac = torch.tensor(0, dtype=torch.float32, device=device)

    adv = advantage
    upgo_adv = upgo_advantage

    if mask is None:
        mask = torch.ones_like(adv)

    for i, logit in enumerate(logits):
        action = actions[i]
        behaviour_logp = behaviour_policy_logprobs[i]
        if valid_actions is None:
            va = torch.ones_like(logit)
        else:
            va = valid_actions[i]

        dist = MaskedPolicy(logit, va)
        logp = dist.log_prob(action)
        ratio = torch.exp(logp - behaviour_logp)
        # policy loss
        surr1 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        surr2 = ratio * adv
        surr = -torch.min(surr1, surr2)
        surr = torch.sum(surr * mask) / torch.sum(mask)
        policy_loss += surr

        # upgo loss
        upgo_surr1 = torch.clamp(ratio, 1 - clip_ratio,
                                 1 + clip_ratio) * upgo_adv
        upgo_surr2 = ratio * upgo_adv
        upgo_surr = -torch.min(upgo_surr1, upgo_surr2)
        upgo_surr = torch.sum(upgo_surr * mask) / torch.sum(mask)
        upgo_loss += upgo_surr

        # entropy loss
        entropy = dist.entropy()
        entropy = torch.sum(entropy * mask) / torch.sum(mask)
        entropy_loss -= entropy
        policy_clip_frac += torch.mean(
            (ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)).float())

    # value loss
    value_loss = compute_value_loss(value, target_value, mask=mask)
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + upgo_coef * upgo_loss

    # log
    log = {
        "policy_loss": policy_loss.item(),
        "upgo_loss": upgo_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "policy_clip_frac": policy_clip_frac.item() / len(logits),
    }

    return loss, log

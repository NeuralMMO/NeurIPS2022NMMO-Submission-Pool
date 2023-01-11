import collections

import torch
from torch import Tensor

GAEReturns = collections.namedtuple("GAEReturns", "vs advantages")
UPGOReturns = collections.namedtuple("UPGOReturns", "vs advantages")


@torch.no_grad()
def gae(
    value: Tensor,
    reward: Tensor,
    bootstrap_value: Tensor,
    discount: Tensor,  # (1 - done) * gamma
    lambda_: float = 1.0,
    mask: Tensor = None,
):
    """
    value: [T, ...]
    reward: [T, ...]
    bootstrap_value: [...]
    discount: [T, ...]
    """
    T = value.shape[0]
    value = torch.cat([value, bootstrap_value.unsqueeze(dim=0)], dim=0)
    delta = reward + discount * value[1:] - value[:-1]
    last_gae_lam = torch.zeros_like(bootstrap_value)
    result = []
    for t in reversed(range(T)):
        last_gae_lam = delta[t] + discount[t] * lambda_ * last_gae_lam
        result.append(last_gae_lam)
    result.reverse()
    adv = torch.stack(result)
    return_ = adv + value[:-1]
    if mask is not None:
        adv *= mask
        return_ *= mask
    return GAEReturns(vs=return_, advantages=adv)


@torch.no_grad()
def upgo(
    value: Tensor,
    reward: Tensor,
    bootstrap_value: Tensor,
    discount: Tensor,  # (1 - done) * gamma
    mask: Tensor = None,
) -> UPGOReturns:
    T = value.shape[0]
    # Append bootstrapped value to get [v1, ..., v_t+1]
    value_t_plus_1 = torch.cat(
        [value[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    target_value = [bootstrap_value]
    for t in reversed(range(T)):
        upgo_return = reward[t] + discount[t] * torch.max(
            value_t_plus_1[t], target_value[-1])
        target_value.append(upgo_return)
    target_value.reverse()
    # Remove bootstrap value from end of target_values list
    target_value = torch.stack(target_value[:-1], dim=0)
    adv = target_value - value
    return_ = target_value
    if mask is not None:
        adv *= mask
        return_ *= mask
    return UPGOReturns(vs=return_, advantages=adv)

import torch


class MaskedPolicy:
    def __init__(self, logits, valid_actions=None):
        self.valid_actions = valid_actions
        if valid_actions is None:
            self.logp = logits - logits.logsumexp(dim=-1, keepdim=True)
        else:
            logits = valid_actions * logits + (1 - valid_actions) * (-1e8)
            self.logp = logits - logits.logsumexp(dim=-1, keepdim=True)

    def sample(self):
        # https://arxiv.org/abs/1411.0030
        u = torch.rand_like(self.logp, device=self.logp.device)
        action = torch.argmax(self.logp - torch.log(-torch.log(u)), axis=-1)
        return action

    def argmax(self):
        return torch.argmax(self.logp, axis=-1)

    def log_prob(self, action):
        action = action.long().unsqueeze(-1)
        logp = torch.gather(self.logp, -1, action).squeeze(-1)
        return logp

    def entropy(self):
        return -(torch.exp(self.logp) * self.logp).sum(dim=-1)

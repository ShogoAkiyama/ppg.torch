from collections import defaultdict

import torch

import torch_util as tu
from tree_util import tree_map
from vec_monitor2 import VecMonitor2


class Roller:
    def __init__(self, venv, model, initial_state, device, keep_buf=100):

        self.model = model
        if not isinstance(venv, VecMonitor2):
            venv = VecMonitor2(venv, keep_buf=keep_buf)
        self._venv = venv
        self.device = device
        self._state = initial_state

    @property
    def episode_lens(self):
        return [ep.len for ep in self._venv.ep_buf]

    @property
    def episode_rets(self):
        return [ep.ret for ep in self._venv.ep_buf]

    def singles_to_multi(self, single_steps):
        """
        Stack single-step dicts into arrays with leading axes (batch, time)
        """
        out = defaultdict(list)
        for d in single_steps:
            for (k, v) in d.items():
                out[k].append(v)

        return {k: torch.stack(v, dim=1).to(self.device) for (k, v) in out.items()}

    def multi_step(self, nstep) -> dict:
        """
        step vectorized environment nstep times, return results
        final flag specifies if the final reward, observation,
        and first should be included in the segment (default: False)
        """

        singles = [self.single_step() for _ in range(nstep)]
        out = self.singles_to_multi(singles)
        finalrew, out["finalob"], out["finalfirst"] = tree_map(
            tu.np2th, self._venv.observe())
        out["finalstate"] = self._state
        out["reward"] = torch.cat([out["lastrew"][:, 1:], finalrew[:, None]], dim=1)

        del out["lastrew"]
        return out

    def single_step(self) -> dict:
        """
        step vectorized environment once, return results
        """
        lastrew, ob, first = tree_map(tu.np2th, self._venv.observe())

        ac, other_outs = self.model.act(ob=ob)
        out = {
            'lastrew': lastrew,
            'ob': ob,
            'first': first,
            'ac': ac
        }

        self._venv.act(ac.cpu().numpy())
        for (k, v) in other_outs.items():
            out[k] = v
        return out

import torch
import itertools
from torch import distributions as td
import math
import numpy as np

from tree_util import tree_map
import torch_util as tu
from roller import Roller
from model.network import PhasicValueModel
from envs.reward_normalizer import RewardNormalizer


INPUT_KEYS = {"ob", "ac", "logp", "vtarg", "adv"}


class PPG:
    def __init__(self, venv, lr, kl_penalty, clip_param, device, comm,
                 gamma, nminibatch, n_epoch_pi, n_epoch_vf, interacts_total,
                 n_aux_epochs, n_pi, aux_lr, aux_mbsize, nstep):
        self.device = device
        self.kl_penalty = kl_penalty
        self.clip_param = clip_param
        self.entcoef = 0.01
        self.nstep = nstep

        self.venv = venv
        self.num_env = self.venv.num
        self.comm = comm
        self.ic_per_step = self.num_env * self.comm.size * self.nstep
        self.model = PhasicValueModel(
            venv.ob_space, venv.ac_space, device)
        self.model.to(device)

        tu.sync_params(self.model.parameters())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.roller = Roller(
            self.venv, self.model, self.model.init_state(self.num_env),
            self.device, keep_buf=100)

        self.reward_norm = RewardNormalizer(self.num_env)

        self.seg_buf = []

        self.gamma = gamma
        self.lambd = 0.95
        self.nminibatch = nminibatch
        self.n_epoch_pi = n_epoch_pi
        self.n_epoch_vf = n_epoch_vf
        self.interacts_total = interacts_total
        self.n_aux_epochs = n_aux_epochs
        self.n_pi = n_pi
        self.aux_opt = torch.optim.Adam(self.model.parameters(), lr=aux_lr)
        self.aux_mbsize = aux_mbsize
        self.envs_segs = torch.tensor(list(itertools.product(
            range(self.num_env), range(self.n_pi))))
        self.total_step = None

    def learn(self):
        for step in range(self.interacts_total):

            tu.sync_params(self.model.parameters())

            episode_rewards = np.zeros(self.n_pi)
            episode_lens = np.zeros(self.n_pi)

            for i in range(self.n_pi):
                seg = self.roller.multi_step(self.nstep)

                episode_rewards[i] = np.mean(self.roller.episode_rets)
                episode_lens[i] = np.mean(self.roller.episode_lens)

                seg["reward"] = self.reward_norm(seg["reward"], seg["first"])
                self.compute_advantage(seg)

                self.seg_buf.append(tree_map(lambda x: x.cpu(), seg))

                tensordict = {k: seg[k] for k in INPUT_KEYS}

                self.update(tensordict)

            self.compute_presleep_outputs()

            self.aux_update()

            self.seg_buf.clear()

            self.total_step = (step + 1) * self.n_pi * self.num_env * self.nstep
            print('---------------------------------------------')
            print(f'Total Step: {self.total_step}    '
                  f'Episode Reward: {np.mean(episode_rewards):.3f}   '
                  f'EpisodeLen: {np.mean(episode_lens):.3f}')
            print('---------------------------------------------')

    def update(self, tensordict):
        for _ in range(self.n_epoch_pi):
            for mb in self.minibatch_gen(tensordict):
                # self.train_pi_and_vf(mb)
                loss = self.compute_losses(**mb)
                self.opt.zero_grad()
                loss.backward()
                tu.warn_no_gradient(self.model, "PPO")
                tu.sync_grads(self.model.parameters())
                self.opt.step()

    def aux_update(self):
        total_loss = 0
        for i in range(self.n_aux_epochs):
            needed_keys = {"ob", "oldpd", 'vtarg'}
            segs = [{k: seg[k] for k in needed_keys} for seg in self.seg_buf]
            total_loss += self.aux_losses(segs)

        print(f'Aux loss: {total_loss / self.n_aux_epochs:.3f}')

    def make_minibatches(self, segs):
        """
        Yield one epoch of minibatch over the dataset described by segs
        Each minibatch mixes data between different segs
        """
        for idx in torch.randperm(len(self.envs_segs)).split(self.aux_mbsize):
            yield tu.tree_stack([
                tu.tree_slice(segs[seg_id], env_id)
                for (env_id, seg_id) in self.envs_segs[idx]])

    def compute_presleep_outputs(self, pdkey="oldpd", vpredkey="oldvpred"):
        def forward(ob):
            pd, vpred, _aux = self.model.forward(ob.to(self.device))
            return pd, vpred

        for seg in self.seg_buf:
            seg[pdkey], seg[vpredkey] = tu.minibatched_call(
                forward, self.aux_mbsize, ob=seg["ob"])

    def aux_losses(self, segs):
        total_loss = 0
        for mb in self.make_minibatches(segs):
            mb = tree_map(lambda x: x.to(self.device), mb)
            pd, _, aux = self.model(mb["ob"])
            kl_loss = td.kl_divergence(mb["oldpd"], pd).mean()
            # vf_aux, vf_true = self.model.compute_aux_loss(aux, mb)
            vf_aux = (0.5 * (aux['vpredaux'] - mb['vtarg']).pow(2)).mean()
            vf_true = (0.5 * (aux["vpredtrue"] - mb['vtarg']).pow(2)).mean()

            # auxiliary loss + vf distance + policy KL
            loss = vf_aux + vf_true + kl_loss

            self.aux_opt.zero_grad()
            loss.backward()
            tu.sync_grads(self.model.parameters())
            self.aux_opt.step()
            total_loss += loss.item()
        return total_loss

    def compute_losses(self, ob, ac, logp, vtarg, adv):
        pd, vpred, aux = self.model(ob=ob)
        newlogp = tu.sum_nonbatch(pd.log_prob(ac))
        # prob ratio for KL / clipping based on a (possibly) recomputed logp
        logratio = newlogp - logp
        ratio = torch.exp(logratio)

        pg_losses = torch.min(ratio * adv, torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv)

        entropy = tu.sum_nonbatch(pd.entropy()).mean()
        pi_kl = 0.5 * (logratio ** 2).mean()

        loss_policy = - pg_losses.mean() - self.entcoef * entropy + self.kl_penalty + pi_kl
        loss_value = (0.5 * (vpred - vtarg).pow(2)).mean()
        loss = loss_policy + loss_value

        return loss

    def minibatch_gen(self, data):
        for mbinds in torch.chunk(torch.randperm(self.num_env), self.nminibatch):
            yield tree_map(lambda x: x.to(self.device), tu.tree_slice(data, mbinds))

    def compute_advantage(self, seg):
        finalob, finalfirst = seg["finalob"], seg["finalfirst"]
        vpredfinal = self.model.v(finalob)
        reward = seg["reward"]
        vpred = torch.cat([seg["vpred"], vpredfinal[:, None]], dim=1)
        first = torch.cat([seg["first"], finalfirst[:, None]], dim=1)

        adv, vtarg = self.compute_gae(reward=reward, vpred=vpred, first=first)
        seg["vtarg"] = vtarg
        adv_mean, adv_var = tu.mpi_moments(self.comm, adv)
        seg["adv"] = (adv - adv_mean) / (math.sqrt(adv_var) + 1e-8)

    def compute_gae(self, vpred, reward, first):
        assert vpred.device == reward.device == first.device

        vpred, reward, first = (x.cpu() for x in (vpred, reward, first))
        first = first.float()

        assert first.dim() == 2
        assert vpred.shape == first.shape == (self.num_env, self.nstep + 1)

        adv = torch.zeros(self.num_env, self.nstep, dtype=torch.float32)

        lastgae = 0
        for t in reversed(range(self.nstep)):
            delta = reward[:, t] + (1.0 - first[:, t+1]) * self.gamma * vpred[:, t+1] - vpred[:, t]
            adv[:, t] = lastgae = delta + (1.0 - first[:, t+1]) * self.gamma * self.lambd * lastgae
        vtarg = vpred[:, :-1] + adv
        return adv.to(device=self.device), vtarg.to(device=self.device)

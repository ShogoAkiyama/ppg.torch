from .utils import *
from .impala_cnn import ImpalaCNN
import torch_util as tu
import torch.nn as nn


class PhasicValueModel(nn.Module):
    def __init__(self, obtype, actype, device):
        super().__init__()

        self.enc_keys = ['pi', 'vf']
        self.device = device

        pi_outsize, self.make_distr = tensor_distr_builder(actype)

        for key in self.enc_keys:
            self.set_encoder(
                key,
                ImpalaCNN(
                    obtype.shape,
                    outsize=256,
                    chans=(16, 32, 32)
                )
            )

        self.vhead = NormedLinear(
            self.get_encoder('vf').outsize, 1, scale=0.1)

        self.pi_head = NormedLinear(
            self.get_encoder('vf').outsize, pi_outsize, scale=0.1)
        self.aux_vf_head = NormedLinear(
            self.get_encoder('vf').outsize, 1, scale=0.1)

    def get_encoder(self, key):
        return getattr(self, key + "_enc")

    def set_encoder(self, key, enc):
        setattr(self, key + "_enc", enc)

    def forward(self, ob):
        x_out = {}

        for k in self.enc_keys:
            x_out[k] = self.get_encoder(k)(ob)

        pivec = self.pi_head(x_out['pi'])
        pd = self.make_distr(pivec)
        vpredaux = self.aux_vf_head(x_out['pi'])[..., 0]

        value = self.vhead(x_out['vf'])[..., 0]
        aux = {
            'vf': value,
            "vpredaux": vpredaux,
            "vpredtrue": value
        }

        return pd, value, aux

    def init_state(self, batchsize):
        return {k: th.zeros((batchsize, 0), device=self.device)
                for k in self.enc_keys}

    @tu.no_grad
    def act(self, ob):
        pd, vpred, _ = self(
            ob=tree_map(lambda x: x[:, None], ob),
        )
        ac = pd.sample()
        logp = sum_nonbatch(pd.log_prob(ac))
        return (
            tree_map(lambda x: x[:, 0], ac),
            dict(vpred=vpred[:, 0], logp=logp[:, 0]),
        )

    @tu.no_grad
    def v(self, ob):
        _pd, vpred, _ = self(
            ob=tree_map(lambda x: x[:, None], ob))
        return vpred[:, 0]

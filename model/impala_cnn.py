import math

from torch.nn import functional as F

from gym3.types import Real
from .utils import *
REAL = Real()


class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if getattr(self, "batch_norm", False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, "batch_norm", False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = NormedConv2d(inchan, outchan, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList(
            [CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, "pool", True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, "pool", True):
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    name = "ImpalaCNN"  # put it here to preserve pickle compat

    def __init__(self, inshape, outsize, chans=(16, 32, 32),
                 scale_ob=255., nblock=2, final_relu=True, **kwargs):
        super().__init__()
        self.scale_ob = scale_ob
        self.outsize = outsize
        h, w, c = inshape
        curshape = (c, h, w)
        s = 1 / math.sqrt(len(chans))  # per stack scale
        layers = nn.ModuleList()
        for outchan in chans:
            stack = CnnDownStack(
                curshape[0], nblock=nblock,
                outchan=outchan, scale=s, **kwargs)
            layers.append(stack)
            curshape = stack.output_shape(curshape)

        self.stacks = nn.Sequential(*layers)
        self.dense = NormedLinear(intprod(curshape), outsize, scale=1.4)
        self.final_relu = final_relu

    def forward(self, x):
        x = x.to(dtype=th.float32) / self.scale_ob

        b, t = x.shape[:-3]
        x = x.reshape(b * t, *x.shape[-3:])
        x = transpose(x, "bhwc", "bchw")
        x = self.stacks(x)
        x = x.reshape(b, t, *x.shape[1:])
        x = flatten_image(x)
        x = th.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x

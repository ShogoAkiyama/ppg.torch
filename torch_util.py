import collections
import functools
import os
import platform

import numpy as np
import torch as th
import torch.distributed as dist
import torch.distributions as dis
from mpi4py import MPI
import tree_util
import socket
import time
import random
import multiprocessing as mp


def format_model(mod, rms=False):
    """
    Return a str: a formatted table listing parameters and their sizes
    """
    import pandas

    rows = []
    ntotal = sum(p.numel() for p in mod.parameters())
    for name, param in sorted(mod.named_parameters()):
        shape = tuple(param.shape)
        numel = param.numel()
        std = "%0.5f" % float(param.std())
        crnt = [name, shape, numel, round(numel / ntotal * 100, 1),
                std, ((param ** 2).mean() ** 0.5).item()]
        rows.append(crnt)

    for name, module in mod.named_modules():
        numel = sum(p.numel() for p in module.parameters())
        if numel == 0:
            continue
        crnt = [name or "~total", "", numel, round(numel / ntotal * 100, 1), "", ""]
        rows.append(crnt)
    columns = ["path", "shape", "numel", "pct", "std", "rms"]
    if not rms:
        rows = [row[:-1] for row in rows]
        columns = columns[:-1]
    rows.sort(key=lambda x: x[0])
    df = pandas.DataFrame(rows, columns=columns)
    maxlen = df["path"].str.len().max()
    return df.to_string(
        index=False, formatters={"path": "{{:<{}s}}".format(maxlen).format})


def allsame(xs):
    """
    Returns whether all elements of sequence are the same
    """
    assert len(xs) > 0
    return all(x == xs[0] for x in xs[1:])


def contextmanager_to_decorator(cm):
    def decorator(fn):
        @functools.wraps(fn)
        def newfn(*args, **kwargs):
            with cm():
                return fn(*args, **kwargs)

        return newfn

    return decorator


no_grad = contextmanager_to_decorator(th.no_grad)


def torch_init_process_group(
    backend, comm=None, start_port=29500, attempts=10, jitter_seconds_per_rank=0.005
):
    """
    Setup torch distributed
    """
    from mpi4py import MPI
    from torch import distributed as dist

    if dist.is_initialized():
        # already initialized
        return

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
    os.environ["NCCL_SOCKET_NTHREADS"] = "8"
    # (clemens) this ordering is faster
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,2,7,6,4,5"
    if platform.system() == "Darwin":
        # By default, Gloo will try to resolve the hostname, eventually
        # time out, and then fall back to the local machine.
        # This makes it use the local machine right away
        os.environ["GLOO_SOCKET_IFNAME"] = "en0"
        # using localhost saves around 5 seconds per group creation
        hostname = "localhost"
    elif "RAPID_ID" in os.environ:  # Although this function is in rcall, I (joschu)
        # would like to use it from code launched by rapid. As of last time I tried,
        # gethostname() didn't work.
        hostname = socket.gethostbyname(socket.getfqdn())
    else:
        hostname = socket.gethostname()
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    # sometimes torch.dist doesn't clean up old processes.
    # this ensures processes are started
    # this outer loop tries different ports in case some are in use, this may be unnecessary since we should be able to usually
    # choose an unused port before calling this function
    for i in range(attempts):
        # we have to offset the port by the global rank of the broadcasting comm so that different groups don't clobber each other
        port = start_port + MPI.COMM_WORLD.Get_rank() * attempts + i
        port = comm.bcast(port, root=0)
        comm.Barrier()
        success = False
        # this inner loop retries in case we got "Connection reset by peer"
        for _ in range(3):
            # add some jitter to avoid overloading incoming connections on the master
            time.sleep(random.random() * jitter_seconds_per_rank * comm.size)
            try:
                os.environ["MASTER_PORT"] = str(port)
                # this takes 5 minutes to timeout, the timeout option doesn't seem to do anything
                dist.init_process_group(backend=backend, init_method=f"env://")

            except RuntimeError as e:
                _log(f"failed with error '{e}', trying again")

            # We check if we are initialized here because it helps to avoid errors of:
            # "trying to initialize the default process group twice!"
            if dist.is_initialized():
                success = True
                break

        successes = comm.allgather(success)
        if all(successes):
            # all ranks succeeded, we're done here
            break
        if success:
            # some machines didn't succeed, attempt to retry by destroying the process group
            dist.destroy_process_group()
    else:
        raise RuntimeError("Failed to init on any port")


def _get_local_rank_size(comm):
    """
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.
    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = collections.defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]


def torch_setup(device_type=None, gpu_offset=0):
    """
    Setup torch to use the correct device and number of threads.  This should be called before `torch_init_process_group`

    Returns the torch device to use
    """
    from mpi4py import MPI
    import torch

    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    local_rank, local_size = _get_local_rank_size(MPI.COMM_WORLD)
    if device_type == "cuda":
        device_index = (local_rank + gpu_offset) % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
    else:
        device_index = 0
    if "RCALL_NUM_CPU_PER_PROC" in os.environ:
        num_threads = int(os.environ["RCALL_NUM_CPU_PER_PROC"])
    else:
        num_threads = max(round(mp.cpu_count() // 2 / local_size), 1)
    torch.set_num_threads(num_threads)
    return torch.device(type=device_type, index=device_index)


def setup_dist(
    device_type=None,
    comm=None,
    backend=None,
    should_init_process_group=True,
    start_port=29500,
    gpu_offset=0,
    device='cuda'
):
    """
    Use MPI communicator to set up torch distributed.

    Sets two global variables:
    - DEFAULT_DEVICE is default device used by this process,
    - DEFAULT_COMM is the MPI communicator used to set up torch distributed
    """
    global DEFAULT_DEVICE, DEFAULT_COMM
    if device_type is None:
        device_type = device

    if (
        os.environ.get("PYTEST_RUNNING", "0") == "1"
        and os.environ.get("MPI_CALL_RUNNING", "0") != "1"
    ):
        assert comm.size == 1
        return

    DEFAULT_DEVICE = torch_setup(device_type=device_type, gpu_offset=gpu_offset)
    if should_init_process_group:
        backend = backend or ("nccl" if device_type == "cuda" else "gloo")
        if device_type == "cpu":
            assert (
                backend != "nccl"
            ), "nccl backend will not work with device_type='cpu'"
        DEFAULT_COMM = comm
        torch_init_process_group(backend=backend, start_port=start_port, comm=comm)


def dev():
    return DEFAULT_DEVICE


def np2th(nparr):
    dtype = th.float32 if nparr.dtype == np.float64 else None
    return th.from_numpy(nparr).to(device=dev(), dtype=dtype)


def all_mean_(x, group=dist.group.WORLD):
    dist_all_reduce(x, group=group)
    x /= dist_get_world_size(group=group)
    return x


def unflatten_to(newflat, xs):
    start = 0
    for x in xs:
        size = x.numel()
        end = start + size
        x.copy_(newflat[start:end].view(x.shape))
        start = end
    assert start == newflat.numel()


def is_distributed():
    return dist.is_initialized()


def dist_broadcast(*args, **kwargs):
    if not is_distributed():
        return
    dist.broadcast(*args, **kwargs)


def dist_all_reduce(*args, **kwargs):
    if not is_distributed():
        return
    dist.all_reduce(*args, **kwargs)


def dist_get_world_size(group=dist.group.WORLD):
    if not is_distributed():
        return 1
    return dist.get_world_size(group=group)


def flatten_tensors(xs):
    buf = xs[0].new_empty(sum(x.numel() for x in xs))
    i = 0
    for x in xs:
        buf[i:i + x.numel()].copy_(x.view(-1))
        i += x.numel()
    return buf


def sync_params(params, src_rank=0, group=dist.group.WORLD):
    """
    Send parameters from src_rank to all others in the group
    """
    datas = [p.data for p in params]
    flatvec = flatten_tensors(datas)
    dist_broadcast(flatvec, src=src_rank, group=group)
    unflatten_to(flatvec, datas)


def sync_grads(params, group=dist.group.WORLD):
    """
    Sync gradients for the provided params across all members of the specified group
    """
    if not is_distributed():
        assert group is dist.group.WORLD
        return
    if dist.get_world_size(group) == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    flatgrad = flatten_tensors(grads)
    all_mean_(flatgrad, group=group)
    unflatten_to(flatgrad, grads)


def _numpy_allmean(comm, x):
    out = np.zeros_like(x)
    comm.Allreduce(x, out)
    out /= comm.size
    return out


def mpi_moments(comm: MPI.Comm, x: th.Tensor) -> (float, float):
    mean_x_x2 = np.array([x.mean().item(), (x ** 2).mean().item()])
    mean_x_x2 = _numpy_allmean(comm, mean_x_x2)
    mean_x, mean_x2 = mean_x_x2
    var_x = mean_x2 - mean_x ** 2
    return float(mean_x), max(float(var_x), 0)


@functools.lru_cache()  # Just run once
def register_distributions_for_tree_util():
    tree_util.register_pytree_node(
        dis.Categorical,
        lambda d: ((d.logits,), None),
        lambda _keys, xs: dis.Categorical(logits=xs[0]),
    )
    tree_util.register_pytree_node(
        dis.Bernoulli,
        lambda d: ((d.logits,), None),
        lambda _keys, xs: dis.Bernoulli(logits=xs[0]),
    )


@functools.lru_cache()
def warn_no_gradient(model, task):
    for n, p in model.named_parameters():
        if p.grad is None:
            print(f"parameter '{n}' {p.shape} has no gradient for '{task}'")


@no_grad
def minibatched_call(fn, mbsize, *args, **kwargs):
    """
    Same result as fn(**kwargs) but breaking up the inputs
    into minibatches of size mbsize to avoid OOM errors
    """
    tensor_list, _ = tree_util.tree_flatten((args, kwargs))
    batchsize = tensor_list[0].shape[0]
    mbs = [
        fn(*tree_slice(args, inds), **tree_slice(kwargs, inds))
        for inds in th.arange(batchsize).split(mbsize)
    ]
    return tree_cat(mbs, dim=0)


def tree_stack(trees):
    return tree_util.tree_multimap(lambda *xs: th.stack(xs, dim=0), *trees)


def tree_cat(trees, dim=0):
    return tree_util.tree_multimap(lambda *xs: th.cat(xs, dim=dim), *trees)


def tree_slice(tree, sli):
    " slice tree data by (lambda x: x[sli]) function"
    return tree_util.tree_map(lambda x: x[sli], tree)


def sum_nonbatch(x, nbatchdim=2):
    return x.sum(dim=tuple(range(nbatchdim, x.dim()))) if x.dim() > nbatchdim else x


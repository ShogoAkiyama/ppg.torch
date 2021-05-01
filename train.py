import torch
import argparse
from mpi4py import MPI

from ppg import PPG
import torch_util as tu
from envs.envs import get_venv


def main(env_name="coinrun", distribution_mode="hard",
         interacts_total=100_000_000, num_envs=64, n_epoch_pi=1, n_epoch_vf=1,
         gamma=.999, aux_lr=5e-4, lr=5e-4, nminibatch=8, aux_mbsize=4,
         clip_param=.2, kl_penalty=0.0, n_aux_epochs=6, n_pi=32, nstep=256):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comm = MPI.COMM_WORLD

    tu.setup_dist(comm=comm, device=device.type)
    tu.register_distributions_for_tree_util()

    # make env
    venv = get_venv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode)

    ppg = PPG(venv, lr, kl_penalty, clip_param, device, comm, gamma,
              nminibatch, n_epoch_pi, n_epoch_vf, interacts_total,
              n_aux_epochs, n_pi, aux_lr, aux_mbsize, nstep)
    ppg.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PPG training arguments.')
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--n_epoch_pi', type=int, default=1)
    parser.add_argument('--n_epoch_vf', type=int, default=1)
    parser.add_argument('--n_aux_epochs', type=int, default=6)
    parser.add_argument('--n_pi', type=int, default=32)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--kl_penalty', type=float, default=0.0)
    parser.add_argument('--arch', type=str, default='dual')   # 'shared', 'detach', or 'dual'
    parser.add_argument('--nstep', type=int, default=256)

    args = parser.parse_args()

    main(env_name=args.env_name, num_envs=args.num_envs, n_epoch_pi=args.n_epoch_pi,
         n_epoch_vf=args.n_epoch_vf, n_aux_epochs=args.n_aux_epochs, n_pi=args.n_pi,
         nstep=args.nstep)

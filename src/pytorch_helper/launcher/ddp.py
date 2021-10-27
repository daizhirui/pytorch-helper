import os
from typing import Callable
from typing import Sequence

from torch.distributed import init_process_group
from torch.distributed import is_gloo_available
from torch.distributed import is_mpi_available
from torch.distributed import is_nccl_available

__all__ = [
    'launch_ddp_task'
]


def launch_ddp_task(
    gpu_id: Sequence[int], n_gpus: int, run_task_func: Callable,
    *run_task_arg
):
    """ default function used to launch a DistributedDataParallel process

    :param gpu_id: Sequence of gpu devices
    :param n_gpus: The number of gpus in the group
    :param run_task_func: Callable to run the task
    :param run_task_arg: arguments for `run_task_func`
    """
    ddp_port = int(os.environ['DDP_PORT'])
    if is_nccl_available():
        backend = 'nccl'
    elif is_mpi_available():
        backend = 'mpi'
    elif is_gloo_available():
        backend = 'gloo'
    else:
        raise RuntimeError('No available distributed communication package')
    init_process_group(
        backend=backend, init_method=f'tcp://localhost:{ddp_port}',
        world_size=n_gpus, rank=gpu_id)
    # tasks must be built after initializing the process group
    run_task_func([gpu_id], *run_task_arg)

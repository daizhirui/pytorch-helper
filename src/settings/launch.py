import os

from torch.distributed import init_process_group


def launch_ddp_task(gpu_id, n_gpus, run_task_func, main_args, task_option):
    ddp_port = int(os.environ['DDP_PORT'])
    init_process_group(
        backend='nccl', init_method=f'tcp://localhost:{ddp_port}',
        world_size=n_gpus, rank=gpu_id)
    # tasks must be built after initializing the process group
    run_task_func([gpu_id], main_args, task_option)

# Copyright (c) Zhirui Dai
"""
This file contains functions for waiting gpus to be ready when there are other
processes using the requested gpus.
"""

import time
from typing import List

import gpustat
import torch

from pytorch_helper.utils import log
from .dist import is_distributed


def collect_gpu(gpu_id: int , mb_size: int = None) -> torch.Tensor:
    """ collect the memory of a gpu

    :param gpu_id: gpu index
    :param mb_size: int of memory size in MB to collect
    :return: torch.Tensor of the requested memory
    """
    log.info(__name__, f'Collect gpu {gpu_id}')
    device = torch.device('cuda', gpu_id)
    if mb_size is None:
        q = gpustat.GPUStatCollection.new_query()[gpu_id]
        mb_size = q.memory_total
    MB = 2 ** 18
    size = int(0.85 * mb_size * MB)
    block = torch.empty(size, dtype=torch.float32).to(device)
    return block


def wait_gpus(
        gpus_to_wait: List[int], collect=True, pre_release=True
):
    """ wait requested gpus' become available (memory becomes empty)

    :param gpus_to_wait: list of gpu indices
    :param collect: Bool to collect the gpu when its memory becomes empty
    :param pre_release: Bool to release all the collected GPU memory before
        returning from this function
    """
    distributed = is_distributed()
    gpus_to_wait.sort()
    gpus_not_ready = True
    blocks = dict()
    while gpus_not_ready:
        time.sleep(5)
        gpus_not_ready = False
        queries = gpustat.GPUStatCollection.new_query()

        for i, gpu_id in enumerate(gpus_to_wait):
            q = queries[gpu_id]
            if len(q.processes) > 0:
                log.info(
                    __name__, f'GPU {gpu_id} is used, check again in 5 seconds'
                )
                gpus_not_ready = True
                break
            else:
                log.info(__name__, f'GPU {gpu_id} is ready')
                if collect:
                    blocks[gpu_id] = collect_gpu(
                        i if distributed else gpu_id, q.memory_total
                    )

    log.info(__name__, f'GPUs {gpus_to_wait} are ready!')
    del blocks
    if pre_release:
        torch.cuda.empty_cache()

# Copyright (c) Zhirui Dai
"""
This file contains functions for waiting gpus to be ready when there are other
processes using the requested gpus.
"""

import time
from typing import List

import torch

from pytorch_helper.utils import log
from .dist import is_distributed
from .dist import synchronize
from .gpustat import GPUStatCollection


def collect_gpu(gpu_id: int, mb_size: int = None) -> torch.Tensor:
    """ collect the memory of a gpu

    :param gpu_id: gpu index
    :param mb_size: int of memory size in MB to collect
    :return: torch.Tensor of the requested memory
    """
    log.info(__name__, f'Collect gpu {gpu_id}')
    device = torch.device('cuda', gpu_id)
    if mb_size is None:
        q = GPUStatCollection.new_query()[gpu_id]
        mb_size = q.memory_total
    MB = 1024 * 1024
    size = int(0.80 * mb_size * MB / 4)
    block = torch.empty(size, dtype=torch.float32).to(device)
    return block


def wait_gpus(
        gpus: List[int], collect=True, pre_release=True, sync=True
):
    """ wait requested gpus' become available (memory becomes empty)

    :param gpus: list of gpu indices
    :param collect: Bool to collect the gpu when its memory becomes empty
    :param pre_release: Bool to release all the collected GPU memory before
        returning from this function
    :param sync: Bool to synchronize all the processes such that they release
        the gpu memory together
    """
    distributed = is_distributed()
    gpus.sort()
    gpus_not_ready = True
    blocks = dict()
    while gpus_not_ready:
        time.sleep(5)
        gpus_not_ready = False
        queries = GPUStatCollection.new_query()

        for i, gpu_id in enumerate(gpus):
            q = queries[gpu_id]
            if q.processes is not None and len(q.processes) > 0:
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

    log.info(__name__, f'GPUs {gpus} are ready!')

    if sync:
        synchronize()

    del blocks
    if pre_release:
        torch.cuda.empty_cache()

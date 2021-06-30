import time

import gpustat
import torch

from utils import log


def collect_gpu(gpu_id, mb_size=None):
    device = torch.device('cuda', gpu_id)
    if mb_size is None:
        q = gpustat.GPUStatCollection.new_query()[gpu_id]
        mb_size = q.memory_total
    MB = 2 ** 18
    size = int(0.85 * mb_size * MB)
    block = torch.empty(size, dtype=torch.float32).to(device)
    return block


def wait_gpus(gpus_to_wait, collect=True, pre_release=False):
    gpus_not_ready = True
    blocks = dict()
    while gpus_not_ready:
        time.sleep(5)
        gpus_not_ready = False
        queries = gpustat.GPUStatCollection.new_query()

        for gpu_id in gpus_to_wait:
            q = queries[gpu_id]
            if len(q.processes) > 0:
                log.info(f'GPU {gpu_id} is used, check again in 5 seconds')
                gpus_not_ready = True
                break
            else:
                log.info(f'GPU {gpu_id} is ready')
                if collect:
                    blocks[gpu_id] = collect_gpu(gpu_id, q.memory_total)

    log.info(f'GPUs {gpus_to_wait} are ready!')
    del blocks
    if pre_release:
        torch.cuda.empty_cache()

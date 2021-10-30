import os


def ready_for_torch(assert_=False):
    result = 'CUDA_VISIBLE_DEVICES' in os.environ
    if assert_:
        assert result, 'The environment is not ready for this module yet.'
    return result


def set_cuda_visible_devices(gpus: list):
    # make sure that the order is the same as the one shown in nvidia-smi
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


def get_cuda_visible_devices():
    return os.environ['CUDA_VISIBLE_DEVICES'].split(',')

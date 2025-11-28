import subprocess
import os
import torch
# from ..utils.lockfile import MultiUserGlobalLockFile
from typing import List, Dict, Optional

gpu_fake_usage = []


def get_memory_usage() -> Optional[Dict[int, int]]:
    try:
        proc = subprocess.Popen("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits".split(" "),
                         stdout=subprocess.PIPE)
        lines = [s.strip().split(" ") for s in proc.communicate()[0].decode().split("\n") if s]
        return {int(g[0][:-1]): int(g[1]) for g in lines}
    except:
        return None


def get_free_gpus() -> Optional[List[int]]:
    try:
        free = []
        proc = subprocess.Popen("nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits".split(" "),
                                stdout=subprocess.PIPE)
        uuids = [s.strip() for s in proc.communicate()[0].decode().split("\n") if s]

        proc = subprocess.Popen("nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits".split(" "),
                                stdout=subprocess.PIPE)

        id_uid_pair = [s.strip().split(", ") for s in proc.communicate()[0].decode().split("\n") if s]
        for i in id_uid_pair:
            id, uid = i

            if uid not in uuids:
                free.append(int(id))

        return free
    except:
        return None


def _fix_order():
    os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


# def _create_gpu_usage(n_gpus: int):
#     global gpu_fake_usage

#     for i in range(n_gpus):
#         a = torch.FloatTensor([0.0])
#         a.cuda(i)
#         gpu_fake_usage.append(a)

def _create_gpu_usage(n_gpus: int):
    global gpu_fake_usage

    available_gpus = torch.cuda.device_count()
    print("="*100)
    print("Available GPUs:", available_gpus)
    print("="*100)
    for i in range(n_gpus):
        if i >= available_gpus:
            raise RuntimeError(f"Invalid device ordinal: {i}. Only {available_gpus} GPUs available.")

        a = torch.FloatTensor([0.0])
        a.cuda(i)
        gpu_fake_usage.append(a)


def allocate(n:int = 1):
    _fix_order()
    # with MultiUserGlobalLockFile("gpu_allocation_lock"):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("WARNING: trying to allocate %d GPUs, but CUDA_VISIBLE_DEVICES already set to %s" %
                (n, os.environ["CUDA_VISIBLE_DEVICES"]))
        return

    allocated = get_free_gpus()
    if allocated is None:
        print("WARNING: failed to allocate %d GPUs" % n)
        return
    allocated = allocated[:n]

    if len(allocated) < n:
        print("There is no more free GPUs. Allocating the one with least memory usage.")
        usage = get_memory_usage()
        if usage is None:
            print("WARNING: failed to allocate %d GPUs" % n)
            return

        inv_usages = {}

        for k, v in usage.items():
            if v not in inv_usages:
                inv_usages[v] = []

            inv_usages[v].append(k)

        min_usage = list(sorted(inv_usages.keys()))
        min_usage_devs = []
        for u in min_usage:
            min_usage_devs += inv_usages[u]

        min_usage_devs = [m for m in min_usage_devs if m not in allocated]

        n2 = n - len(allocated)
        if n2>len(min_usage_devs):
            print("WARNING: trying to allocate %d GPUs but only %d available" % (n, len(min_usage_devs)+len(allocated)))
            n2 = len(min_usage_devs)

        allocated += min_usage_devs[:n2]

    os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(a) for a in allocated])
    _create_gpu_usage(len(allocated))


# def use_gpu(gpu:str = "auto", n_autoalloc: int = 1):
#     _fix_order()

#     gpu = gpu.lower()
#     if gpu in ["auto", ""]:
#         allocate(n_autoalloc)
#     elif gpu.lower()=="none":
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     else:
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#         _create_gpu_usage(len(gpu.split(",")))

#     return len(os.environ.get("CUDA_VISIBLE_DEVICES","").split(","))

# def use_gpu(gpu: str = "auto", n_autoalloc: int = 1):
#     _fix_order()

#     gpu = gpu.lower()
#     if gpu in ["auto", ""]:
#         allocate(n_autoalloc)
#     elif gpu.lower() == "none":
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     else:
#         # Validate GPU indices
#         available_gpus = get_free_gpus()
#         if available_gpus is None:
#             print("WARNING: No available GPUs found.")
#             return 0

#         requested_gpus = [int(g) for g in gpu.split(",") if g.isdigit()]
#         valid_gpus = [g for g in requested_gpus if g in available_gpus]

#         if len(valid_gpus) != len(requested_gpus):
#             print("WARNING: Some requested GPUs are not available. Using available ones only.")

#         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, valid_gpus))
#         _create_gpu_usage(len(valid_gpus))

#     return len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))



def use_gpu(gpu: str = "auto", n_autoalloc: int = 1):
    _fix_order()

    gpu = gpu.lower()
    if gpu in ["auto", ""]:
        allocate(n_autoalloc)
    elif gpu.lower() == "none":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Validate GPU indices
        available_gpus = torch.cuda.device_count()
        requested_gpus = [int(g) for g in gpu.split(",") if g.isdigit()]
        valid_gpus = [g for g in requested_gpus if g < available_gpus]

        if len(valid_gpus) != len(requested_gpus):
            print("WARNING: Some requested GPUs are not available. Using available ones only.")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, valid_gpus))
        _create_gpu_usage(len(valid_gpus))

    return len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
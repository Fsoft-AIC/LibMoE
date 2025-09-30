import torch
import time
import argparse
from multiprocessing import Process
import random


def allocate_gpu_memory(cuda_id, spare_mem=0.5, wtime=5):
    device = torch.device(f"cuda:{cuda_id}")
    while True:
        with torch.cuda.device(device):

            # Get total GPU memory in bytes for the specified device
            total_memory = torch.cuda.mem_get_info()[0]
            total_memory_gb = total_memory / (1024 ** 3)
            print(f"CUDA:{cuda_id} - Total GPU memory: {total_memory_gb:.2f} GB")

            # Memory to leave unused in bytes
            spare = int(spare_mem * (1024 ** 3))

            # Calculate memory to allocate
            memory_to_allocate = total_memory - spare
            if memory_to_allocate <= 0:
                print(f"CUDA:{cuda_id} - Insufficient memory to allocate.")
            else:
                print(f"CUDA:{cuda_id} - Allocating approximately {memory_to_allocate / (1024 ** 3):.2f} GB...")

                # Allocate memory
                try:
                    tensor_size = memory_to_allocate // 4 + random.randint(1, 100)    # Divide by 4 as PyTorch tensors use 4 bytes per float32
                    allocation_tensor = torch.empty(tensor_size, dtype=torch.float32, device=device)
                    print(f"CUDA:{cuda_id} - Memory allocated successfully. {spare_mem} GB left unused.")
                except RuntimeError as e:
                    print(f"CUDA:{cuda_id} - Error during memory allocation: {e}")

            # Sleep for 24 hours
            print(f"CUDA:{cuda_id} - Wait {wtime}s for next check...")
            import datetime
            print(time.time())
            time.sleep(wtime)


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--cuda_ids",
        type=int,
        nargs="+",
#        required=True,
	default=[1,3,4,5],
        help="List of CUDA device IDs.",
    )
    parser.add_argument(
        "--spare_mem",
        type=float,
        default=0.5,
        help="",
    )

    parser.add_argument(
	"--time",
	type=float,
	default=5,
	help="",
    )

    args = parser.parse_args()

    processes = []

    # Start a separate process for each CUDA device
    for cuda_id in args.cuda_ids:
        p = Process(target=allocate_gpu_memory, args=(cuda_id, args.spare_mem, args.time))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

    # /cm/archive/anonymous/miniconda3/envs/moeut/bin/python3.10 scripts/experiments/main.py --cuda_ids 4 5 6 --spare_mem 3

import torch
import time
import argparse
import threading
import psutil
from typing import List, Tuple
import numpy as np

def get_gpu_utilization(device: int) -> float:
    """Get GPU utilization percentage using nvidia-smi."""
    try:
        return torch.cuda.utilization(device)
    except:
        return 0.0

def perform_intensive_computations(tensor_list: List[torch.Tensor], device: torch.device, target_util: float = 0.9):
    """Perform continuous computations to maintain high GPU utilization."""
    batch_size = 1024

    # Create persistent tensors for computations
    compute_tensors = [
        torch.randn(batch_size, batch_size, device=device),
        torch.randn(batch_size, batch_size, device=device),
        torch.randn(batch_size, 1, device=device)
    ]

    while True:
        current_util = get_gpu_utilization(device.index)

        if current_util < target_util:
            # Increase computation intensity
            for _ in range(max(1, int((target_util - current_util) * 10))):
                # Matrix operations
                result = torch.matmul(compute_tensors[0], compute_tensors[1])
                result = torch.nn.functional.relu(result)

                # Complex operations
                result = torch.fft.fft2(result)
                result = torch.abs(result)

                # Non-linear transformations
                result = torch.sin(result) + torch.cos(result)
                result = torch.sqrt(torch.abs(result))

                # Convolution operations
                result = torch.nn.functional.conv2d(
                    result.unsqueeze(0).unsqueeze(0),
                    torch.randn(1, 1, 3, 3, device=device),
                    padding=1
                ).squeeze()

                # Update compute tensors
                compute_tensors[0] = result / result.norm()
                compute_tensors[1] = torch.matmul(compute_tensors[1], compute_tensors[0])

        # Small delay to prevent GPU overheating
        time.sleep(0.01)

def monitor_gpu_stats(device: torch.device) -> Tuple[float, float, float, float]:
    """Monitor GPU memory usage and utilization."""
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
    utilization = get_gpu_utilization(device.index)

    print(f"\rGPU {device.index} | "
          f"Usage: {utilization:3.0f}% | "
          f"Memory: {allocated_memory:.1f}/{total_memory:.1f} GB "
          f"({(allocated_memory/total_memory)*100:.1f}%)", end="")

    return allocated_memory, total_memory, reserved_memory, utilization

def maintain_gpu_load(target_gpu: int = 0, target_memory: float = 0.8, target_util: float = 0.9):
    """Maintain specified GPU memory and utilization levels."""
    device = torch.device(f"cuda:{target_gpu}")
    tensor_list = []
    min_allocation = 64 * 1024 * 1024  # 64 MB minimum allocation
    allocation_size = 256 * 1024 * 1024  # Start with 256 MB allocations

    print(f"\nInitializing GPU {target_gpu} load management...")
    print(f"Target memory usage: {target_memory*100:.1f}%")
    print(f"Target utilization: {target_util*100:.1f}%")

    # Start computation thread with device passed directly
    compute_thread = threading.Thread(
        target=perform_intensive_computations,
        args=(tensor_list, device, target_util),
        daemon=True
    )
    compute_thread.start()

    # Initial memory allocation
    while True:
        try:
            tensor = torch.rand((allocation_size // 4,), device=device, dtype=torch.float32)
            tensor_list.append(tensor)
            allocated_memory, total_memory, reserved_memory, utilization = monitor_gpu_stats(device)

            if allocated_memory / total_memory > target_memory:
                print("\nReached target memory allocation")
                break

        except RuntimeError as e:
            print(f"\nAllocation failed, reducing size: {e}")
            allocation_size = max(allocation_size // 2, min_allocation)
            if allocation_size == min_allocation:
                print("\nReached minimum safe allocation size")
                break

    print("\nMonitoring GPU load...")
    consecutive_fails = 0

    while True:
        time.sleep(1)

        # Monitor and adjust memory usage
        allocated_memory, total_memory, reserved_memory, utilization = monitor_gpu_stats(device)
        memory_usage = allocated_memory / total_memory

        # Adjust memory if needed
        if memory_usage < target_memory:
            try:
                new_allocation_size = int((total_memory - allocated_memory) * 0.5 * 1024 ** 3 // 4)
                new_allocation_size = max(min_allocation, min(new_allocation_size, 512 * 1024 * 1024))

                if new_allocation_size >= min_allocation:
                    tensor = torch.rand((new_allocation_size // 4,), device=device, dtype=torch.float32)
                    tensor_list.append(tensor)
                    consecutive_fails = 0

            except RuntimeError as e:
                consecutive_fails += 1
                if consecutive_fails >= 3:
                    new_allocation_size = new_allocation_size // 2
                    consecutive_fails = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU memory and utilization management script.")
    parser.add_argument("--data", type=str, help="A dummy argument for demonstration purposes")
    parser.add_argument("--bs", type=str, help="A dummy argument for demonstration purposes")
    parser.add_argument("--block", type=str, help="A dummy argument for demonstration purposes")
    parser.add_argument("--nheads", type=str, help="A dummy argument for demonstration purposes")
    parser.add_argument("--nlayers", type=str, help="A dummy argument for demonstration purposes")
    parser.add_argument("--lr", type=str, help="A dummy argument for demonstration purposes")
    args = parser.parse_args()

    # Start GPU load management
    maintain_gpu_load(target_gpu=0, target_memory=0.35, target_util=0.01)

    # /cm/archive/anonymous/miniconda3/envs/moeut/bin/python3.10 scripts/experiments/main_triton.py --data slimpajama-8000 --bs 48 --nlayers 6 --nheads 8 --lr 0.001
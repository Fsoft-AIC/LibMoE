import torch

# Kiểm tra GPU
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# Tạo tensor trên GPU
x = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
print("Tensor on GPU:", x)

import torch

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA not available.")

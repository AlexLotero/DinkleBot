import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

import torch

# Simple CUDA test
if torch.cuda.is_available():
    print("CUDA is available.")
    tensor = torch.randn(3, 3).cuda()
    print(tensor)
else:
    print("CUDA is not available.")
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return at least 1
print(torch.cuda.get_device_name(0))  # Should return your GPU model

# import nvidia_smi
import torch

# nvidia_smi.nvmlInit()

# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

# info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Using", torch.cuda.device_count(), "GPUs")
# print("Total memory:", info.total/1e9)
# print("Free memory:", info.free/1e9)
# print("Used memory:", info.used/1e9)
import json, platform, sys
try:
    import torch
    torch_info={"available":torch.cuda.is_available(),"version":torch.__version__,"device_count":torch.cuda.device_count()}
except ImportError:
    torch_info={"available":False,"version":None,"device_count":0}
print(json.dumps({"python":sys.version,"platform":platform.platform(),"torch":torch_info}, indent=2))

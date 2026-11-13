"""Run in the intended Colab/runtime; does not install or train anything."""
import json
import platform
import sys

report = {"python": sys.version, "platform": platform.platform()}
try:
    import torch
    report.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        report.update(device=device.name, capability=torch.cuda.get_device_capability(0), vram_bytes=device.total_memory)
        layer = torch.nn.Linear(76, 10, device="cuda")
        loss = layer(torch.randn(32, 76, device="cuda")).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        report["forward_backward"] = "passed"
except Exception as error:
    report["error"] = str(error)
print(json.dumps(report, indent=2))
if "error" in report:
    sys.exit(1)

import sys, torch
sys.path.insert(0, '/content/r04-depth-adaptation-code-4f5c566')
import scripts.adapt_r04_depth as driver
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
print('DETERMINISTIC_POLICY', torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, flush=True)
driver.main()

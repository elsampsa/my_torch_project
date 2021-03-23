"""A script:

- Load pre-trained neural net
- Modify the neural net (add layers, etc.)
- Save the new neural net

Run locally
""" 
import torch
from model import getModule, mod_

def readModule(module, pth_file, use_gpu = True):
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print('using device %s' % (device))

    if pth_file is not None:
        module.load_state_dict(
            torch.load(pth_file, map_location = device)
            )
    else:
        print("random weights")


path = "/home/sampsa/cnn/kaggle/my_torch_project_2/tmp/resnet50-19c8e357.pth"
module = getModule(original = True)
readModule(module, path, use_gpu = True)
mod_(module)
torch.save(module.state_dict(), "modded.pth")
print(module)

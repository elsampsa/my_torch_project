import sys, os, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T # ToTensor
from torchvision import models 
"""https://pytorch.org/docs/stable/torchvision/models.html

For example:

    import torchvision.models as models                                                                    
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

pth files are downloaded to:

::

    $HOME/.cache/torch/checkpoints/

"""
from my_torch_project.tools import quickLog, getDataFile
from my_torch_project.datamodel import image2Tensor

logger = quickLog(__name__, logging.INFO)

@torch.no_grad()
def init_weights(m):
    if hasattr(m, "weights"):
        m.weight.fill_(1.0)

"""IMPLEMENT YOUR nn.Module subclasses here :)

class YourNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.all_layers = []
        self.n_channels = 1
        self.n_classes = 1

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            # "superpixel" kernel
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # etc.
            )
        # ...
        self.all_layers = [self.net]


    def reset(self):
        for layer in self.all_layers:
            layer.apply(init_weights)

    def forward(self, x):
        t = self.net(x)
"""

def mod_(module):
    # transfer-learning modding
    num_ftrs = module.fc.in_features
    module.fc = nn.Linear(num_ftrs, 5)

def getModule(original = False):
    """instantiate your nn.Module subclass with this function
    here we just take a ready-made ResNet50
    check out the actual implementation here:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    transfer learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    module = models.resnet50() # returns an object whose class has subclassed the nn.Module
    # module = models.resnext50_32x4d()
    # NOTE:
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    if original:
        pass
    else:
        mod_(module)
    return module


def getLRList(module, w0, w1):
    """Adjust different learning rate for resnet50's bottom layers
    """
    names0 = [
        # layer 0
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        # rest of the layers
        "layer1",
        "layer2",
        "layer3"
        # layer 4 is left floatin'
    ]

    names1 = [
        "layer4",
        "avgpool",
        "fc"
    ]

    """optimizer wants this kind of dictionary:

    ::

        {'params': model.base.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
        ...

    """
    lis = []
    for name in names0:
        lis.append({'params' : getattr(module, name).parameters(), 'lr': w0})
    for name in names1:
        lis.append({'params' : getattr(module, name).parameters(), 'lr': w1})
    return lis


class ModuleWrap:
    """A wrapper for this PyTorch model for easy deployment
    """
    def __init__(self, pth_file = None, use_gpu = True, original = False):
        self.logger = quickLog("my_torch_project", logging.INFO)
        self.original = original

        if pth_file is not None:
            assert(os.path.exists(pth_file))
        
        if use_gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        self.logger.info('using device %s', device)

        self.net = getModule(original = self.original)
        
        if pth_file is not None:
            self.net.load_state_dict(
                torch.load(pth_file, map_location = device)
                )
        else:
            self.logger.warning("random weights")

        for param in self.net.parameters(): # this is only for inference
            param.requires_grad = False 

        self.classnames = []
        with open(getDataFile("imagenet_classes.txt"), 'r') as f:
            for line in f:
                self.classnames.append(line.strip())
        # self.classnames.insert(0, "nada")


    def __str__(self):
        return str(self.net)


    def cat2classname(self, i):
        return self.classnames[i]

    def __call__(self, img):
        """Apply necessary image transformations
        
        ::

            input: numpy image (h, w, c)
            => do necessary transformations, add batch dimension
            => tensor with (batch, c, h, w)
            => feed to neural net
            => transform output to whatever you want

        In this particular example case:

        input: normal rgb image (h, w, c)

        output: top five classes
        """
        t = image2Tensor(img)
        t = t.unsqueeze(0) # add minibatch dimension
        # print(">feeding with", t.shape)
        self.net.eval() # do _not_ forget!
        output = self.net(t)
        # print(output.shape) # torch.Size([1, 5]) # with batch dimension

        if self.original:
            # 1000 categories of ImageNet
            # The output has unnormalized scores, hence softmax:
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            res = []
            for i in range(top5_prob.size(0)):
                res.append((
                    self.classnames[top5_catid[i]],
                    top5_prob[i].item()
                ))
        else:
            # transfer-learning-modded
            # 5 categories
            cat_to_label = {
                "0": "Cassava Bacterial Blight (CBB)", 
                "1": "Cassava Brown Streak Disease (CBSD)", 
                "2": "Cassava Green Mottle (CGM)", 
                "3": "Cassava Mosaic Disease (CMD)", 
                "4": "Healthy"
            }
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            # print(">", top_prob, top_catid)
            res = top_catid.item()
            # print(">", res)

        return res


"""Detector classes that use a certain pth file saved into the directory "data/"
"""
class Detector1(ModuleWrap):
    """This detector class uses "model_1.pth" that is in the "data/" directory

    In order to have it in the package, remember to graft the data/ directory
    in MANIFEST.in

    You can use your neural net detector like this:

    ::

        from my_torch_project.model import Detector1
        detector = Detector1()
        res = detector(img)

    """
    def __init__(self, use_gpu = False):
        super().__init__(pth_file = getDataFile("model_1.pth"), use_gpu = use_gpu)


def process_cl_args():
    import argparse    

    def str2bool(v):
        r = v.lower() in ("yes", "true", "True", "t", "1")
        return r

    parser = argparse.ArgumentParser(usage="""
model.py command

    commands:

    push        feed an image to the neural net
    info        describe the neural net

    options:

    --image     image filename (push)

    --weights   filename of model parameters

    --orig      use original library neural net instead
                of the modified one

    """)
    parser.add_argument("command",    
                        action="store", type=str,                 
                        help="action")

    parser.add_argument("--image",    
                        action="store", type=str,                 
                        required = False, default = None,
                        help="image filename")

    parser.add_argument("--weights",    
                        action="store", type=str,                 
                        required = False, default = None,
                        help="weight file")

    parser.add_argument("--orig",    
                        action="store", type=str2bool,
                        required = False, default = False,
                        help="use unmodded neural net or not")

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args


def main():
    """for quick testing / debugging
    """
    from skimage.io import imread

    parsed_args, unparsed_args = process_cl_args()
    comm = parsed_args.command

    #print(">", parsed_args.orig)
    #sys.exit(0)

    m = ModuleWrap(pth_file = parsed_args.weights, original = parsed_args.orig)

    if comm == "info":
        print(m)
        sys.exit(0)

    if comm == "push":
        if parsed_args.image is None:
            print("needs an image")
            sys.exit(1)
        fname = parsed_args.image
        img = imread(fname)
        res = m(img)
        print("got", res)

    else:
        print("unknown command", comm)


if __name__ == "__main__":
    main()
    # python3 model.py --orig=true --weights=/home/sampsa/.cache/torch/checkpoints/resnext50_32x4d-7cdf4587.pth --image=data/dog.jpg push
    # python3 model.py --orig=false push --image ../../data/my_torch_project/train_images/3192927904.jpg

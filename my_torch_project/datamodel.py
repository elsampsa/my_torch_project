from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

"""PIL to and from:
I = numpy.asarray(PIL.Image.open('test.jpg'))
im = PIL.Image.fromarray(numpy.uint8(I))
"""

pil_image_preprocess = T.Compose([  # requires PIL.Image
    T.Resize(256), 
    T.CenterCrop(224), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image2Tensor(img):
    """RGB (h, w, c) image to pytorch tensor (b, c, h, w)

    img must be of uint8 type
    """
    pil_img = Image.fromarray(img)
    # img = resize(img, input_image_size)
    # t = torch.from_numpy(img).type(torch.FloatTensor)
    # t = t.permute(2, 0, 1) # => (c, h, w)
    # t = tensor_preprocess(t) # accepts PIL.Image or a tensor image
    t = pil_image_preprocess(pil_img)
    # t = t.unsqueeze(0) # add batch dimension => (b, c, h, w) # this is done by the loader..?
    # print(">", t.shape)
    return t


def tensor2Image(t):
    """Pytorch tensor (b, c, h, w) to RGB (h, w, c) image
    """
    t = t.squeeze(0) # => (c, h, w)
    t = t.permute(1, 2, 0) # => (h, w, c)
    img = t.numpy()
    # print(">", img.min())
    img = img - img.min()
    img = (img / img.max()) * 255.
    return img.astype(np.uint8)


"""key differences to keras' "Sequence" subclassing:
in Sequence you define batch size and __getitem__
return batches
in pytorch Dataset returns individual elements, while
DataLoader _uses_ Dataset to generate batches
it also shuffles the Dataset if required, etc.
(i.e. no shuffling in Dataset subclass required)

What __getitem__ returns is also flexible as you write
yourself the training method
"""
class CustomDataset(Dataset):
    """Read files from a directory

    Each file has an associated entry in a json file:

    ::

        image_id,label
        1000015157.jpg,0
        1000201771.jpg,3
        100042118.jpg,1
        1000723321.jpg,1
        ...

    """

    def __init__(self, dirname, csv_file, n_classes, n_max = None, only_y = False):
        # self.x, self.y = x_set, y_set
        assert os.path.exists(dirname), "that directory does not exist"
        assert os.path.exists(csv_file), "that csv file does not exist"

        self.dirname=dirname
        self.csv_file=csv_file
        self.n_classes=n_classes
        self.only_y = only_y

        """Depending on your loss function, you might need hot-encoded vector or just class index
        self.class_tensor = torch.from_numpy(
            np.eye(self.n_classes)
        ).type(torch.FloatTensor)
        """
        # self.class_tensor[n] gives hot-encoded vector

        first=True
        self.cache=[]
        with open(self.csv_file,"r") as f:
            for i, line in enumerate(f):
                if first:
                    first=False
                else:
                    cols=line.split(",")
                    filename=cols[0]
                    classnum=int(cols[1])
                    self.cache.append((filename,classnum))
                if (n_max is not None and i > n_max):
                    break


    def __len__(self):
        return len(self.cache)


    def __getitem__(self, idx):
        """idx: batch index
        """
        filename, classnum = self.cache[idx]
        if self.only_y:
            x = None
        else:
            path = os.path.join(self.dirname, filename)
            img = imread(path)
            x = image2Tensor(img)
        y = torch.tensor(classnum)
        # print(">dataset returning", x.shape, y.shape)
        return x, y

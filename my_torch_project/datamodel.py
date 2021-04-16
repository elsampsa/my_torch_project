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

pil_image_preprocess = T.Compose([  # requires PIL.Image or numpy array
    T.Resize((256,256)),
    # T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image2Tensor(img): # , size = 256):
    """RGB (h, w, c) image to pytorch tensor (b, c, h, w)

    img must be of uint8 type
    """
    # pil_img = Image.fromarray(img)
    t = T.Compose([
        # T.Resize((size, size)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # depends..
    ])(img) #(pil_img)
    return t


def mask2Tensor(img): # , size = 256):
    """RGB (h, w, 1) image to pytorch tensor (b, 1, h, w)

    img must be of uint8 type
    """
    # pil_img = Image.fromarray(img)
    t = T.Compose([
        # T.Resize((size, size)),
        T.ToTensor()
    ])(img)
    return t


def tensor2Image(t, sizetup=None):
    """Pytorch tensor (b, c, h, w) to RGB (h, w, c) image

    sizetup: (ysize, xsize)
    """
    t = t.squeeze(0) # => (c, h, w)
    t = t.permute(1, 2, 0) # => (h, w, c)
    img = t.numpy()
    # print(">", img.min())
    #img = img - img.min()
    img = (img / (img.max() - img.min())) * 255.
    img = img.astype(np.uint8)
    """
    return T.Compose([
        T.Resize((ysize, xsize))
    ])(img)
    """
    if sizetup is None:
        return img
    else:
        ysize = sizetup[0]
        xsize = sizetup[1]
        return cv2.resize(img, (xsize, ysize))


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


class UNetDataset(Dataset):
    """Read files from a directory "dirname"


    ::

        dirname/
            images/      
                000000.png
                000001.png
                ...

            masks/
                000000.png
                000001.png
                ...
    """

    def __init__(self, dirname, n_max = None):
        # self.x, self.y = x_set, y_set
        assert os.path.exists(dirname), "that directory does not exist"

        self.dirname=dirname

        self.imgdir = os.path.join(self.dirname, 'images')
        self.maskdir = os.path.join(self.dirname, 'masks')

        assert os.path.exists(self.imgdir), "no images directory"
        assert os.path.exists(self.maskdir), "no masks directory"
        
        lis = glob.glob(os.path.join(self.imgdir,'*.png'))
        lis.sort()
        lis2 = glob.glob(os.path.join(self.maskdir,'*.png'))
        lis2.sort()

        self.cache = []
        cc = 0
        for imgpath in lis:
            #sti = imgpath.split(os.path.sep)[-1].split(".")[0] # "/path/to/000200.png" => "000200"
            #num = int(sti) # "000200" => 200
            maskpath = lis2[cc]
            self.cache.append((imgpath, maskpath))
            cc += 1
            if (n_max is not None) and (cc >= n_max):
                break


    def __len__(self):
        return len(self.cache)


    def __getitem__(self, idx):
        """idx: batch index
        """
        imgpath, maskpath = self.cache[idx]

        img = imread(imgpath)
        mask = RGB2bin(imread(maskpath))
        # mask = np.expand_dims(mask, axis=2) # add third dimension
        # print(">", img.shape)
        # print(">", mask.shape)
        
        t_img = image2Tensor(img)
        t_mask = mask2Tensor(mask)

        return t_img, t_mask

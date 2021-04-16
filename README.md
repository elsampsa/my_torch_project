
# My Torch Project

*mainly for my own use - documentation not that great*

## Features

- Training your model
- Packaging your model
- Training & interference *also* through notebooks
- ..and *also* via docker

Discover yourself.  :)

## Usage

- Use ```reinit.bash``` to reinit this scaffold to your needs
- Install locally in development mode with ```pip3 install --user -e .```
- Dummy-test inference with ```python3 model.py```

You can use ``make.bash`` to create a python package with the .pth neural net weights baked in.

After installing, using your trained model is as neat as this:
```
from my_torch_project.model import Detector1
detector = Detector1()
res = detector(img)
```

## Files
```
your_project/
    model.py        : define your model here
                      define also a module wrapper for
                      weight loading and transformations
                      (say, for numerical values => labels, etc.)

    datamodel.py    : define data transformations, say:
                      tensor <=> image
                      define also data loaders (Dataset subclasses)

    eval.py         : define your evaluation metrics

    trainer.py      : define trainer (training scheme, losses, evaluation steps, etc.)


train/              : training aux dir

    train.py        : cli for running the trainer

    checkpoints/
                    : .pth weight files

    runs/
                    : tensorflow logs

data/
                    : any persistent data for distribution with this module
                      (for example, the final weights)
```

## Copyright & License

(c) 2021 Sampsa Riikonen 

WTFPL

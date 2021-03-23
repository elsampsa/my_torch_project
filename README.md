
# My Torch Project

*mainly for my own use - poorly documented*

Features:

- Training your model
- Packaging your model
- Training & interference through notebooks as well

Discover yourself.  :)

Use ```reinit.bash``` to reinit this scaffold to your needs.

```
your_project/
    model.py        : define your model here
                      define also a module wrapper for
                      weight loading and transformations
                      (say, for numerical values to labels, etc.)

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

# Copyright & License

(c) 2021 Sampsa Riikonen 

WTFPL

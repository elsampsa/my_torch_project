# Using external GPU with Ubuntu 18.04 LTS

Here we are using an external gpu with a laptop with the aid of [this enclosing gadget](https://www.verkkokauppa.com/fi/product/23196/ktqkk/Razer-Core-X-grafiikkatelakka).

After that you can use any nvidia beast GPU card you wish for training neural nets with your laptop.

This also makes it possible to upgrade your GPU capabilities.

## Graphics driver set-up

Check that ```uname -r``` gives you
```
5.3.0-40-generic
```

Then, while using that kernel, type:
```
sudo apt-get purge nvidia-*
```

Next, load cuda + nivida driver + opengl + etc. package from Nvidia's pages [here](https://developer.nvidia.com/cuda-downloads)

It's maybe a good idea not to use the latest release candidate, but a legacy version instead, so go to "Legacy Releases".

Therein, just follow the instructions.

Finally, reboot your computer & cross your fingers.

## Install thunderbold admin tools

```
sudo apt-get install thunderbolt-tools
```

## Activate egpu

External GPU works with simple hot-plugging (no need to keep it connected during bootup).  The product comes with a usb-c cable.

See that egpu is visible in your system with
```
sudo boltctl
```

Remember that **your laptop has several thunderbolt ports and not all are born equal.**  

If the egpu doesn't show up, try connecting your cable to a different usb-c port on your laptop.

Once the egpu is visible, authorize it *para siempre* with:
```
sudo tbtadm approve-all
```

After that, you should be able to see the two gpus with
```
nvidia-smi
```

## Nvidia package installations, cuda etc.

Different cuda version are installed under
```
/usr/local/cuda-VERSION
```
The link path for alternative versions goes like this:
```
/usr/local/cuda -> /etc/alternatives/cuda -> /usr/local/cuda-11.3
```
So let's concentrate on
```
/usr/local/cuda
```
You should add to your ``$PATH``
```
/usr/local/cuda/bin
```
Now the command ``nvcc`` works.  To see the cuda version, do:
```
nvcc --version
```
You can also do this:
```
head /usr/local/cuda/version.json
```
To see which nvidia packages you have installed, do this:
```
apt list --installed | grep "nvidia"
```

## Headbang

If ``nvidia-smi`` gives you "couldn't communicate with the nvidia driver".  Your nvidia driver is not installed / loaded properly.

Check out these:

- https://forums.developer.nvidia.com/t/nvidia-driver-is-not-loaded-ubuntu-18-10/70495/7
- https://askubuntu.com/questions/927199/nvidia-smi-has-failed-because-it-couldnt-communicate-with-the-nvidia-driver-ma

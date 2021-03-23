import logging
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .eval import eval_net
from .model import getLRList
from .tools import quickLog

from torch.utils.tensorboard import SummaryWriter

def trainer1(net=None, device=None, epochs=5, batch_size=1, lr=0.001, 
    save_cp=True, cp_dir = "checkpoints/", log_dir=None, train_loader=None, 
    val_loader=None, show_val_progress = True):

    assert net is not None, "please provide net as nn.Module object"
    assert device is not None, "please provide torch device"
    assert train_loader is not None, "please provide train_loader"
    assert val_loader is not None, "please provide val_loader"

    n_val = len(val_loader.dataset)
    n_train = len(train_loader.dataset)

    multiclass=True

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}', log_dir = log_dir)
    global_step = 0

    logger = quickLog("trainer", logging.INFO)

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # lr_list = getLRList(net, lr*0.1, lr) # bottom layers, upper layers

    optimizer = optim.RMSprop(
        net.parameters(), # all parameters with the same LR
        # lr_list, # different LR for different layers
        lr=lr, 
        weight_decay=1e-8,  # avoid exploding gradient
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        momentum=0.9
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if multiclass else 'max', patience=2)
    if multiclass:
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=nn%20crossentropyloss#torch.nn.CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # WARNING: CrossEntropyLoss is not expecting hot-encoded vector but an indice instead
        # "CrossEntropyLoss takes integer class labels for its target, specifically of type LongTensor"
        # "This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class"
    else:
        # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
        criterion = nn.BCEWithLogitsLoss()
        # "BCEWithLogitsLoss() does take a FloatTensor target"

    try:

        for epoch in range(epochs):
            # mother class for different components is the Module class: https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module#torch.nn.Module
            # train method: "Sets the module in training mode .. this has any effect only on certain modules.."
            net.train()

            epoch_loss = 0
            # print(">EPOCH", epoch, "len", len(train_loader))
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='imgs') as pbar:
                for i, (x, y) in enumerate(train_loader): # DataLoader is an iterable: https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
                    # print(">x, y:", x.shape, y.shape)
                    # x: (b, c, h, w)
                    # y: hot-encoded vector
                    #
                    # tensor has a method to "send" the data to the gpu
                    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to
                    x = x.to(device=device, dtype=torch.float32)
                    y_type = torch.long if multiclass else torch.float32
                    y = y.to(device=device, dtype=y_type)

                    y_pred = net(x)

                    # print("y_pred, y", y_pred.shape, y.shape)
                    loss = criterion(y_pred, y) # applies the loss function..
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True
                    # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step() # from where does optimizer get the dloss/dx value..
                    # optimizer has reference to model parameters from net.parameters() (see above)
                    # ..each parameter has a "memory" of previous values in order to calculate the gradients..?
                    # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
                    # indeed: https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/10
                    # optimizer & loss are not connected in any way, so how the optimizer get's the gradients..!?
                    # this is it...?
                    # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step

                    # pbar.update(i)
                    # pbar.update(1)
                    pbar.update(batch_size)
                    #print(">", i, len(train_loader))
                    global_step += 1

                    n = 5
                    if ( (i % (len(train_loader)//n) == 0) ):
                        # print("\nEVAL at", i, "\n")
                        # at each n.th part of training data, do eval
                        """
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        """
                        # launch the validation step
                        net.eval() # evaluation mode
                        #print("\nENTERING EVAL_NET\n")
                        val_ce, val_acc = eval_net(
                            net, 
                            multiclass, 
                            val_loader, 
                            device,
                            logger,
                            show_progress = show_val_progress
                        )
                        #print("\nEXITED EVAL_NET\n")
                        net.train() # back to training mode

                        scheduler.step(val_ce)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                        if multiclass:
                            #logging.info('Validation cross entropy: {}'.format(val_score))
                            #writer.add_scalar('Loss/test', val_score, global_step)
                            logger.info('Validation CE: %s, Accuracy : %s, LR: %s',
                                val_ce, val_acc, optimizer.param_groups[0]['lr'])
                            writer.add_scalar('Loss/test', val_ce, global_step)
                            writer.add_scalar('Accuracy', val_acc, global_step)
                        else:
                            #logging.info('Validation binary cross entropy: {}'.format(val_score))
                            #writer.add_scalar('Loss/test', val_score, global_step)
                            pass

                        # writer.add_images('images', imgs, global_step)
                        #if net.n_classes == 1:
                        #    writer.add_images('masks/true', true_masks, global_step)
                        #    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

            if save_cp:
                try:
                    os.mkdir(cp_dir)
                    logger.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(
                        net.state_dict(),
                        os.path.join(cp_dir,f'CP_epoch{epoch + 1}.pth')
                    )
                logger.info(f'Checkpoint {epoch + 1} saved !')

    except KeyboardInterrupt:
        torch.save(
            net.state_dict(), 
            os.path.join(cp_dir,'INTERRUPTED.pth')
        )
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    writer.close()




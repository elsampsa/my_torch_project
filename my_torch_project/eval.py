import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from tqdm.notebook import tqdm
# from tqdm.autonotebook import tqdm

from sklearn.metrics import accuracy_score

def eval_net(net, multiclass, loader, device, logger, show_progress = True):
    """net: a nn.Module in the evaluation mode"""
    ## net.eval()
    y_type = torch.long if multiclass else torch.float32
    n_val = len(loader.dataset)
    tot = 0
    grounds=[]
    values=[]
    if not show_progress:
        logger.info("please wait for validation")

    with tqdm(total=n_val, desc='Validation round', unit='batch') as pbar:
        # no f-way to make two simultaneous tqdms to work in a notebook..
        for x, y in loader:
            # print("eval: x, y", x.shape, y.shape)
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=y_type)

            with torch.no_grad():
                y_pred = net(x)

            if multiclass:
                # print("> y, y_pred", y.shape, y_pred.shape)
                """
                y: batch of result vectors
                y_pred: batch of integer values (predictions)
                """
                ## "This criterion combines log_softmax and nll_loss in a single function."
                ## the first argument is in the vector form, while the second is
                ## just the class index
                tot += F.cross_entropy(y_pred, y).item()
                #print("\ny_pred>", y_pred)
                #print("ny>", y)
                #print("tot>", tot)
                probabilities = torch.nn.functional.softmax(y_pred, dim=1)
                #print("probs>", probabilities)
                top_prob, top_catid = torch.topk(probabilities, 1)
                for i in range(y.shape[0]): # loop over batch results
                    grounds.append(y[i].item())
                    values.append(top_catid[i].item())
                #print("grounds", grounds)
                #print("values", grounds)
                #grounds.append(y.item())
                #values.append(top_catid.item())
            else:
                tot += F.binary_cross_entropy_with_logits(y_pred, y).item()
                # TODO: precision, recall, F1-score, etc
                pass
            if show_progress:
                pbar.update()

    ## net.train()
    if multiclass:
        return tot/n_val, accuracy_score(grounds, values)
    else:
        return tot/n_val, None # TODO



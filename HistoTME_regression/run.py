import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
from tensorboardX import SummaryWriter
from scipy.stats import pearsonr

from model import *
from tqdm import tqdm
from data import *
from utils import EarlyStopper

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
	"""
	Utility function to set seed values for RNG for various modules
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False


class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Predicting immunotherapy response for lung cancer using weakly supervised learning")
        self.parser.add_argument("--save", dest="save", action="store", required=True)
        self.parser.add_argument("--load", dest="load", action="store")
        self.parser.add_argument("--lr", dest="lr", action="store", default=1e-4, type=float)
        self.parser.add_argument("--epochs", dest="epochs", action="store", default=40, type=int)
        self.parser.add_argument("--batch_size", dest="batch_size", action="store", default=1, type=int)
        self.parser.add_argument("--num_workers", dest="num_workers", action="store", default=8, type=int)
        self.parser.add_argument("--bag_size", dest="bag_size", action="store", default=-1, type=int)
        self.parser.add_argument("--dataset", dest="dataset", action="store", type=str, default='tme')
        self.parser.add_argument("--embed", dest="embed", action="store", default='uni', type=str)
        self.parse()
        #self.check_args()

    def parse(self):
        self.opts = self.parser.parse_args()
    def __str__(self):
        return ("All Options:\n" + "".join(["-"] * 45) + "\n" + "\n".join(["{:<18} -------> {}".format(k, v) for k, v in vars(self.opts).items()]) + "\n" + "".join(["-"] * 45) + "\n")
    

def run(args, epoch, mode, dataloader, model, optimizer, multitask_list):
    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()
    else:
        assert False, "Wrong Mode:{} for Run".format(mode)
    model_name = model.__class__.__name__.lower()

    accum_iter = 8 # batch accumulation parameter

    n_token = 5

    loss_fn = torch.nn.HuberLoss()
    #loss_fn = torch.nn.MSELoss()

    losses = []
    correct = 0
    count = {'SUNY_correct':0, 'TCGA_correct':0, 
                'SUNY_total':0, 'TCGA_total':0}

    multitask_preds = {}
    multitask_truth = {}
    for key in multitask_list:    
        multitask_preds[key] = []
        multitask_truth[key] = []
    counter=0

    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data_it, data in enumerate(dataloader):
            features = data['features'].to(device)
            loss_dict = {}

            if model_name == 'multitask_abmil': 
                attn, multitask_slide_preds= model(features)

                for key in data['multitask_labels'].keys():
                    multitask_label = data['multitask_labels'][key].float().to(device)
                    multitask_pred = multitask_slide_preds[key].squeeze(dim=2)

                    multitask_loss = loss_fn(multitask_pred, multitask_label)

                    loss_dict[key] = multitask_loss.reshape(1)
                    
                    multitask_preds[key].append(multitask_pred.detach().cpu().numpy())
                    multitask_truth[key].append(multitask_label.detach().cpu().numpy())
                    
                    
            loss = torch.cat(list(loss_dict.values())).mean()
            
            if mode == "train":
                loss.backward()
                if ((data_it + 1) % accum_iter == 0) or (data_it + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
            losses.append(loss.item())
            t.set_postfix(loss=losses[-1])
            t.update()

    epoch_loss = sum(losses) / len(losses)

    train_file = open(f"logs/{args.embed}/{args.save}/training_log.txt","a")
    train_file.write(f"##### Epoch {epoch} {mode.upper()} #####\n")

    rmse_list = []
    for key in multitask_preds.keys():
        if not multitask_preds[key]:
            continue

        multitask_pred_all = np.concatenate(multitask_preds[key], 0)
        multitask_truth_all = np.concatenate(multitask_truth[key], 0)
        r = pearsonr(multitask_pred_all.squeeze(), multitask_truth_all.squeeze())
        mse = np.mean((multitask_pred_all - multitask_truth_all)**2)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
        print(f'{key} rmse = {rmse} | pearson r = {r}')
        train_file.write(f'{key} rmse = {rmse} | pearson r = {r}\n')

    accuracy = sum(rmse_list) / len(rmse_list)
    train_file.write(f"Overall RMSE = {accuracy}\n")
    train_file.write(f"Loss = {epoch_loss}\n \n")
    train_file.close()

    return epoch_loss, accuracy

def main(args):
    train_dataset, val_dataset, test_dataset, feat_dim, multitask_list = load_dataset(args.dataset, args.embed)
    test_dataset=pd.DataFrame()
    task_counts = {}
    for task in multitask_list:
        task_counts[task] = 1

    logRoot = f"logs/{args.embed}/{args.save}/"
    runRoot = f"runs/{args.embed}/{args.save}/"
    if not os.path.exists(logRoot):
        os.makedirs(logRoot)
        os.makedirs(runRoot)

    bag_size=None if args.bag_size==-1 else args.bag_size

    train_loader = build_mil_loader(args, train_dataset, "train", bag_size, task_counts)
    if not val_dataset.empty:
        val_loader = build_mil_loader(args, val_dataset, "val", bag_size, task_counts)
    else:
        test_loader = build_mil_loader(args, test_dataset, "test", None, task_counts)
    
    print("Dataset Split: {}".format(len(train_dataset)))
    print("Number of tasks: {}".format(len(multitask_list)))

    model = multitask_ABMIL(task_counts, feat_dim, 1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    best_train_loss, best_val_loss = float("inf"), float("inf")

    logger = SummaryWriter(logdir = runRoot)
    
    #reset training log
    train_log = os.path.join(logRoot, 'training_log.txt')
    open(train_log,"w").close()

    best_valid_epoch = 0
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    for epoch in range(args.epochs):
        train_loss, train_acc = run(args, epoch, "train", train_loader, model, optimizer, multitask_list)
        print("Train Epoch Loss: {}, RMSE: {}".format(train_loss, train_acc))
        logger.add_scalar("Train Loss", train_loss, epoch)

        val_loss, val_acc = run(args, epoch, "val", val_loader, model, optimizer, multitask_list)
        print("Val Epoch Loss: {}, RMSE: {}".format(val_loss,val_acc))
        print(" ")
        logger.add_scalar("Val Loss", val_loss, epoch)

        is_best_loss = False
        if val_loss < best_val_loss:
            best_epoch, best_train_loss, best_val_loss, is_best_loss = epoch, train_loss, val_loss, True
            best_valid_epoch = epoch
        
        model.save_checkpoint(logRoot, optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)

        if early_stopper.early_stop(val_loss):
            print("#### Early stopped...")
            break

    with open(train_log, 'a') as f:
        f.write(f"\nEpoch for best validation loss : {best_valid_epoch} \n"), print(f"\nEpoch for best validation loss : {best_valid_epoch}")
        f.write("Train Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_train_loss)), print("Train Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_train_loss))
        f.write("\nVal Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_val_loss)), print("Val Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_val_loss))
        f.write("")

if __name__ == "__main__":
    set_seed(1)
    args = Options()
    print(args)

    main(args.opts)

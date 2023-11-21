from operator import index
import os
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import torch
import datetime, pickle, random
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from scipy.stats import spearmanr

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-c", "--chain", default='beta', help="Specify mode of train, alpha/beta")
parser.add_argument("-s", "--seed", default=1234, type=int, help="Specify the random seed")
parser.add_argument("-rp", "--resultpath", help="Specify the model")
parser.add_argument("-m", "--modelpath", help="Specify the model")
parser.add_argument("-cp", "--codepath", help="Specify the modelpath")
parser.add_argument("-n", "--modelname", default="VitTCR", help="Specify the model")
args = parser.parse_args()

import sys 
sys.path.append(args.codepath)
from TcrPepTransform_utils import *
from TcrPepTransform_beta import *
from utils_train import *


def valid_step(model, features, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels) # torch type
        pred = softmax(predictions.cpu().detach().numpy()) # numpy type
        labels = labels.cpu().numpy()
        acc = model.acc_func(y_true=labels, y_pred=pred.argmax(1)) # numpy type 
        auc_roc = model.auc_roc_func(y_true=labels, y_score=pred[:,1]) # numpy type
        auc_pr = model.auc_pr_func(y_true=labels, y_score=pred[:,1]) # numpy type
        F1_score = model.f1_score_func(y_true=labels, y_pred=pred.argmax(1)) # numpy type
    return loss.item(), acc, auc_roc, auc_pr, F1_score


def eval_model(model_initial, epochs, dl_valid):
    dfhistory = pd.DataFrame(columns=['epoch', 'eval_loss', 'eval_acc', 'eval_auc_roc', 'eval_auc_pr', 'eval_f1_score'])
    for epoch in range(1, epochs+1):
        starttime = datetime.datetime.now()
      
        model = model_initial
        params = "model/{}_epochs.pt".format(str(int(epoch)))
        checkpoint = torch.load(params, map_location='cpu')
        model.load_state_dict(checkpoint)

        for _,(features_valid, labels_valid) in enumerate(dl_valid):
            features_valid, labels_valid = features_valid, labels_valid
            features_valid = features_valid.float()
            labels_valid = labels_valid[:, 1].long()
            val_loss, val_acc, val_auc_roc, val_auc_pr, val_F1_score = valid_step(model, features=features_valid, labels=labels_valid)

        info = (epoch, val_loss, val_acc, val_auc_roc, val_auc_pr, val_F1_score)
        dfhistory.loc[epoch-1] = info
        dfhistory.to_csv("dfhistory_eval_tmp", header=True, sep='\t', index=False)
        endtime = datetime.datetime.now()
        totaltime = endtime-starttime
        print("Epoch: {}      totaltime: {}".format(epoch, totaltime))
    return dfhistory


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed==0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkidr(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ =="__main__":
    modelname='VitTCR'
    result_path = '{}/Prediction_for_Model_{}'.format(args.resultpath, modelname)
    mkidr(path=result_path)

    setup_seed(args.seed)
    testfile = "../Covid19/epimut_TCRs_atchley_20220920.pickle"
    with open(testfile, 'rb') as f:
        dicts_valid = pickle.load(f)
    if args.chain=='beta':
        print("Beta-related data loading...")
        data_valid = np.array(dicts_valid['combined_map_beta'])
        data_valid = torch.from_numpy(data_valid)
    elif args.chain=='alpha':
        print("Alpha-related data loading...")
        data_valid = np.array(dicts_valid['combined_map_alpha'])
        data_valid = torch.from_numpy(data_valid)

    label_valid = np.zeros(len(data_valid))
    label_valid_initialized = Labels_Initialization(num_classes=2, labels=label_valid)
    dataset_valid = MyDataset(data=data_valid, labl=label_valid_initialized)
    dl_valid = DataLoader(dataset_valid, batch_size=len(data_valid), shuffle=True, drop_last=True, num_workers=3)
    

    repeats=[]
    folds=[]
    epochs=[]
    coefs=[]
    pvalue=[]
    for repeat in range(5):
        for fold in range(5):
            model = TcrPepTransform_single(input_height=20, 
                                        input_width=12, 
                                        in_chans=5, 
                                        patch_size=4, 
                                        num_classes=2, 
                                        embed_dim=256, 
                                        depth=1, 
                                        num_heads=4, 
                                        mlp_ratio=4, 
                                        qkv_bias=True, 
                                        drop_rate=0.1, 
                                        attn_drop_rate=0.05, 
                                        drop_path_rate=0, 
                                        act_layer=nn.GELU)
            history = pd.read_csv("{}/metrics_repeat{}/Test_modelvitTCR_lr0.001_fold{}.tsv".format(args.modelpath, repeat, fold), sep='\t')
            ind = history['loss'].idxmin()
            epoch = history.iloc[ind]['epoch']
            params = "{}/model_repeat{}/fold{}/{}_epochs.pt".format(args.modelpath, repeat, fold, str(int(epoch)))  
            checkpoint = torch.load(params, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()
            
            pred = model(data_valid.float())
            pred = softmax(pred.cpu().detach().numpy())
            pred = pd.DataFrame(pred)
            pred.columns = ['prob_0', 'prob_1']
            pred_total = pd.DataFrame({'cdr3':np.array(dicts_valid['cdr3']), 'peptide':np.array(dicts_valid['peptide']), 'percent_activation':np.array(dicts_valid['percent_activation']), 'prob_1':np.array(pred.prob_1)})
            pred_total.to_csv('{}/prediction_repeat{}_fold{}_epoch_{}_prediction.csv'.format(result_path, repeat, fold, epoch), index=False)
            coef, p = spearmanr(pred_total.prob_1, pred_total.percent_activation)
            coefs.append(coef)
            pvalue.append(p)
            epochs.append(epoch)
            repeats.append(repeat)
            folds.append(fold)
            # repeats.append('repeat{}_fold{}'.format(repeat,fold))
    # summary_coef_model = pd.DataFrame({'model':modelname, 'epoch': epochs, 'spearmanr':coefs, 'pvalue':pvalue, 'expr':repeats})
    summary_coef_model = pd.DataFrame({'model':modelname, 'epoch': epochs, 'spearmanr':coefs, 'pvalue':pvalue, 'repeat':repeats, 'fold':folds})
    summary_coef_model.to_csv('{}/summary_coef_vit.csv'.format(result_path),index=False)
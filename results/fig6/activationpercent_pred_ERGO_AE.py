import os
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import torch
import torch.nn as nn
import datetime, pickle, random, csv,torchmetrics
from random import shuffle
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from scipy.stats import spearmanr


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-s", "--seed", default=1234, type=int, help="Specify the random seed")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--ae_file")
# parser.add_argument("-rp", "--resultpath", default="../", help="Specify the model")
parser.add_argument("-rp", "--resultpath", default="../", help="Specify the model")
parser.add_argument("-m", "--modelpath", default="../ERGO_ae_healthy", help="Specify the model")
parser.add_argument("-n", "--modelname", default="ERGO_AE", help="Specify the model")
args = parser.parse_args()

import sys 
sys.path.append(args.modelpath) 
import ae_utils as ae
from ERGO_models import AutoencoderLSTMClassifier
 

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

def predict(model, batches, device):
    model.eval()
    # shuffle(batches)
    loss_function = nn.BCELoss()
    test_total_preds = torch.Tensor()
    test_total_labels = torch.Tensor()
    with torch.no_grad():
        for batch in batches:
            tcrs, padded_peps, pep_lens, batch_signs = batch
            tcrs = tcrs.to(device)
            padded_peps = padded_peps.to(device)
            batch_signs = torch.tensor(batch_signs).to(device)
            probs = model(tcrs, padded_peps, pep_lens)
            probs = probs.squeeze(-1)
            test_total_preds = torch.cat((test_total_preds, probs.detach().cpu()), 0)
            test_total_labels = torch.cat((test_total_labels, batch_signs.detach().int().cpu()), 0)
    return test_total_preds

setup_seed(args.seed)
if __name__ =="__main__":
    modelname=args.modelname
    result_path = '{}/Prediction_for_Model_{}'.format(args.resultpath, modelname)
    mkidr(path=result_path)


    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    params = {}
    params['wd'] = 0
    params['epochs'] = 1000
    params['lstm_dim'] = 500
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['enc_dim'] = 100
    params['train_ae'] = True
    # params['batch_size'] = 512
    print("device", args.device)
    args.ae_file = '{}/TCR_Autoencoder/tcr_ae_dim_'.format(args.modelpath) + str(params['enc_dim']) + '.pt'
    checkpoint = torch.load(args.ae_file, map_location=args.device)
    params['max_len'] = checkpoint['max_len']

    # 数据初始化
    testfile = "../Covid19/epimut_TCRs_20220920.csv"
    test_data = pd.read_csv(testfile)
    params['batch_size'] = len(test_data)
    test_tcrs, test_peps, test_signs = list(test_data['cdr3b']), list(test_data['peptide']), list(test_data['percent_activation'])
    test_tcrs1, test_peps1, test_signs1 = list(test_data['cdr3b']), list(test_data['peptide']), list(test_data['percent_activation'])
    test_signs = list(map(float, test_signs))
    test_batches = ae.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])

    # 模型初始化
    repeats=[]
    epochs=[]
    coefs=[]
    pvalue=[]
    folds=[]
    for repeat in range(5):
        for fold in range(5):
            params['batch_size'] = len(test_data)
            history = pd.read_csv("{}/metrics_repeat{}/Test_modelERGO_lr0.001_fold{}.tsv".format(args.modelpath, repeat, fold), sep='\t')
            ind = history['loss'].idxmin()
            epoch = history.iloc[ind]['epoch']
            epoch = '{:03d}'.format(int(epoch))

            for root, dirs, files in os.walk("{}/model_repeat{}/fold{}".format(args.modelpath, repeat, fold)):
                for file in files:
                    if epoch in file:
                        model = AutoencoderLSTMClassifier(params['emb_dim'], args.device, params['max_len'], 21, params['enc_dim'], params['batch_size'], args.ae_file, params['train_ae'])
                        model.to(args.device)
                        PATH = os.path.join(root, file)
            model.load_state_dict(torch.load(PATH, map_location=args.device))

            #开始计算
            pred = predict(model, test_batches, args.device)
            pred = pred.cpu().detach().numpy()
            pred = pd.DataFrame(pred)    
            pred.columns = ['prob_1']
            pred_total = pd.DataFrame({'cdr3':test_tcrs1, 'peptide':test_peps1, 'percent_activation':test_signs1, 'prob_1':np.array(pred.prob_1)})         
            pred_total.to_csv('{}/prediction_repeat{}_fold{}_epoch_{}_prediction.csv'.format(result_path, repeat, fold, epoch), index=False)
            coef, p = spearmanr(pred_total.prob_1, pred_total.percent_activation)
            coefs.append(coef)
            pvalue.append(p)
            epochs.append(epoch)
            repeats.append(repeat)
            folds.append(fold)
    summary_coef_model = pd.DataFrame({'model':modelname, 'epoch': epochs, 'spearmanr':coefs, 'pvalue':pvalue, 'repeat':repeats, 'fold':folds})
    summary_coef_model.to_csv('{}/summary_coef_ergo.csv'.format(result_path),index=False)
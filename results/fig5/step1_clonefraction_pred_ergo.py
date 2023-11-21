import os
# gpu_id = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import torch
import torch.nn as nn
import pickle
import numpy as np
import csv, re
import pandas as pd
import datetime
from random import shuffle
from argparse import ArgumentParser

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--repeat", default=1)
parser.add_argument("--ae_file")
parser.add_argument("-m", "--modelpath", default="../ERGO_ae_healthy", help="Specify the model")
args = parser.parse_args()

import sys 
sys.path.append(args.modelpath)
from ERGO_models import AutoencoderLSTMClassifier
import ae_utils as ae


def mkidr(path):
    if not os.path.exists(path):
        os.makedirs(path)

def predicting_ergo(model, batches, device):
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
			# test_total_labels = torch.cat((test_total_labels, batch_signs.detach().int().cpu()), 0)
	return test_total_preds


if __name__=='__main__':
    modelname   = 'ERGO_AE'
    result_path = '../result_of_{}/1.Prediction'.format(modelname)
    mkidr(path=result_path)

    # Model initialization
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
    args.ae_file='{}/TCR_Autoencoder/tcr_ae_dim_{}.pt'.format(args.modelpath, str(params['enc_dim']))
    checkpoint = torch.load(args.ae_file, map_location=args.device)
    params['max_len'] = checkpoint['max_len']

    for repeat in range(5):
        for fold in range(5):
            
            # for aa, bb, cc in os.walk("{}/metric_repeat{}".format(args.modelpath, repeat)):
            #     if len(cc) == 10:
            #         print('Model has been trained and validated successfully!')

            for donor in range(1,5):
                clonefraction = pd.read_csv('../../0.Data/10x_Genomics/data_processed/step4.clonalfrac/clonalfrac_donor{}.csv'.format(donor))  
                for root, dirs, files in os.walk('../../0.Data/10x_Genomics/data_processed/step5.combine/donor{}/'.format(donor)):
                    for file in files:
                        if "cdr3b_pep" in file and ".csv" in file:
                            validfile = root+file
                            hla_restriction = re.findall(".*cdr3b_pep(.*).csv",validfile)
                            if hla_restriction == ['']:
                                hla_restriction = "total"
                            else:
                                hla_restriction = hla_restriction[0].split('_')[1]
                            test_data = pd.read_csv(validfile)
                            params['batch_size'] = len(test_data)
                            test_tcrs, test_peps, test_signs = list(test_data['cdr3b']), list(test_data['peptide']), list(test_data['Binding'])
                            test_signs = list(map(float, test_signs))
                            test_batches = ae.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])


                            history = pd.read_csv("{}/metrics_repeat{}/Test_modelERGO_lr0.001_fold{}.tsv".format(args.modelpath, repeat, fold), sep='\t')
                            ind = history['loss'].idxmin()
                            epoch = history.iloc[ind]['epoch']
                            epoch = '%03d' % epoch  
                            for root_temp, dirs, paramfiles in os.walk("{}/model_repeat{}/fold{}/".format(args.modelpath, repeat, fold)):
                                for paramfile in paramfiles:
                                    if "epoch{}".format(epoch) in paramfile:
                                        checkpoint_path = root_temp+paramfile
                            model = AutoencoderLSTMClassifier(params['emb_dim'], args.device, params['max_len'], 21, params['enc_dim'], params['batch_size'], args.ae_file, params['train_ae'])
                            model.to(args.device)
                            model.load_state_dict(torch.load(checkpoint_path))
                            pred = predicting_ergo(model=model, batches=test_batches, device=args.device)
                            percent = []
                            for seq in np.array(test_data['cdr3b']):
                                values = clonefraction[clonefraction.clonotype==seq]['percent']
                                percent.append(values.item())
                            pred_total = pd.DataFrame({'prob_1':pred, 'cdr3':np.array(test_data.cdr3b), 'percent':percent})
                            pred_total.to_csv('{}/Donor_{}_HLA_{}_repeat{}_fold{}.csv'.format(result_path, donor, hla_restriction, repeat, fold), index=False) 
                            print("Calculation for donor_{} hla_{} in repeat_{}_fold{}".format(donor, hla_restriction, repeat, fold)) 
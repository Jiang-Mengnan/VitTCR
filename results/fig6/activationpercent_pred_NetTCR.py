from argparse import ArgumentParser
from operator import index
import sklearn.metrics as metrics,re
import os, sys,pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras_metrics as km
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate
from keras.optimizers import adam_v2
from keras.initializers import glorot_normal
from keras.activations import sigmoid
from sklearn.metrics import roc_auc_score
import keras.backend as K
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.callbacks import Callback
from copy import deepcopy
from scipy.stats import spearmanr

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-rp", "--resultpath", default="../", help="Specify the model")
parser.add_argument("-m", "--modelpath", default="../hla_A0301_IEDB/NetTCR2.0_healthy", help="Specify the model")
parser.add_argument("-n", "--modelname", default="NetTCR", help="Specify the model")
args = parser.parse_args()

import sys 
sys.path.append(args.modelpath)
import utils


def predicting_nettcr(mdl, tcrb_test, pep_test):
    prediction = mdl([tcrb_test, pep_test])
    prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)
    prediction = prediction.numpy()
    pred_prob = np.squeeze(prediction)
    return pred_prob

def nettcr_ab():
    pep_in = Input(shape=(12, 20))
    cdrb_in = Input(shape=(20, 20))
       
    pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
    pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
    pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
    pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
    pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool9 = GlobalMaxPooling1D()(pep_conv9)

    cdrb_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool1 = GlobalMaxPooling1D()(cdrb_conv1)
    cdrb_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool3 = GlobalMaxPooling1D()(cdrb_conv3)
    cdrb_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool5 = GlobalMaxPooling1D()(cdrb_conv5)
    cdrb_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool7 = GlobalMaxPooling1D()(cdrb_conv7)
    cdrb_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool9 = GlobalMaxPooling1D()(cdrb_conv9)
    

    pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
    cdrb_cat = concatenate([cdrb_pool1, cdrb_pool3, cdrb_pool5, cdrb_pool7, cdrb_pool9])


    cat = concatenate([pep_cat, cdrb_cat], axis=1)
    
    dense = Dense(32, activation='sigmoid')(cat)
        
    out = Dense(1, activation='sigmoid')(dense)
    
    model = (Model(inputs=[cdrb_in, pep_in],outputs=[out]))
    
    return model

def mkidr(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
if __name__ =="__main__":
    # 数据初始化
    modelname = args.modelname
    result_path = '{}/Prediction_for_Model_{}'.format(args.resultpath, modelname)
    mkidr(path=result_path)

    # 数据初始化
    testfile = "../Covid19/epimut_TCRs_20220920.csv"
    test_data = pd.read_csv(testfile)
    encoding = utils.blosum50_20aa
    pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 12)
    tcrb_test = utils.enc_list_bl_max_len(test_data.cdr3b, encoding, 20)
    
    # 模型初始化
    repeats=[]
    folds=[]
    epochs=[]
    coefs=[]
    pvalue=[]
    for repeat in range(5):
        for fold in range(5):
            mdl = nettcr_ab()

            history = pd.read_csv("{}/metrics_repeat{}/Test_modelNetTCR-2.0_lr0.001_fold{}.tsv".format(args.modelpath, repeat, fold), sep='\t')
            ind = history['loss'].idxmin()
            epoch = history.iloc[ind]['epoch']
            epoch = '%03d' % epoch  
            for root_temp, dirs, files in os.walk('{}/model_repeat{}/fold{}/'.format(args.modelpath, repeat, fold)):
                for paramfile in files:
                    if "epoch{}".format(epoch) in paramfile:
                        checkpoint_path = root_temp+paramfile
            mdl.load_weights(checkpoint_path)
            
            pred = predicting_nettcr(mdl=mdl, tcrb_test=tcrb_test, pep_test=pep_test)
            pred_total = pd.DataFrame({'cdr3':np.array(test_data.cdr3b), 'peptide':np.array(test_data.peptide), 'percent_activation':np.array(test_data.percent_activation), 'prob_1':pred})
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
    summary_coef_model.to_csv('{}/summary_coef_nettcr.csv'.format(result_path), index=False)        
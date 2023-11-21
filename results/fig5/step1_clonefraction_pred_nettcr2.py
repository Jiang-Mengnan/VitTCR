from argparse import ArgumentParser
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


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-m", "--modelpath", default="../NetTCR2.0_healthy/", help="Specify the model")
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


if __name__=='__main__':
    # Data loading
    modelname   = 'NetTCR2.0'
    result_path = '../result_of_{}/1.Prediction'.format(modelname)
    mkidr(path=result_path)


    for repeat in range(5):
        for fold in range(5):
            for donor in range(1,5):
                clonefraction = pd.read_csv('../../0.Data/10x_Genomics/data_processed/step4.clonalfrac/clonalfrac_donor{}.csv'.format(donor))  
                for root, dirs, files in os.walk('../../0.Data/10x_Genomics/data_processed/step5.combine/donor{}/'.format(donor)):
                    for file in files:
                        if "cdr3b_pep" in file and ".csv" in file:
                            validfile = root+file
                            hla_restriction = re.findall(".*cdr3b_pep(.*).csv", validfile)
                            if hla_restriction == ['']:
                                hla_restriction = "total"
                            else:
                                hla_restriction = hla_restriction[0].split('_')[1]

                                test_data = pd.read_csv(validfile)
                                encoding = utils.blosum50_20aa
                                pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 12)
                                tcrb_test = utils.enc_list_bl_max_len(test_data.cdr3b, encoding, 20)
                                y_test = np.array(test_data.Binding)

                                
                                history = pd.read_csv("{}/metrics_repeat{}/Test_modelNetTCR-2.0_lr0.001_fold{}.tsv".format(args.modelpath, repeat, fold), sep='\t')
                                ind = history['loss'].idxmin()
                                epoch = history.iloc[ind]['epoch']
                                epoch = '%03d' % epoch  
                                mdl = nettcr_ab()
                                for root_temp, dirs, files in os.walk('{}/model_repeat{}/fold{}/'.format(args.modelpath, repeat, fold)):
                                    for paramfile in files:
                                        if "epoch{}".format(epoch) in paramfile:
                                            checkpoint_path = root_temp+paramfile
                                mdl.load_weights(checkpoint_path)
                                pred = predicting_nettcr(mdl=mdl, tcrb_test=tcrb_test, pep_test=pep_test)
                                percent = []
                                for seq in np.array(test_data['cdr3b']):
                                    values = clonefraction[clonefraction.clonotype==seq]['percent']
                                    percent.append(values.item())
                                pred_total = pd.DataFrame({'prob_1':pred, 'cdr3':np.array(test_data.cdr3b), 'percent':percent})
                                pred_total.to_csv('{}/Donor_{}_HLA_{}_repeat{}_fold{}.csv'.format(result_path, donor, hla_restriction, repeat, fold), index=False) 
                                print("Calculation for donor_{} hla_{} in repeat_{}_fold{}".format(donor, hla_restriction, repeat, fold)) 

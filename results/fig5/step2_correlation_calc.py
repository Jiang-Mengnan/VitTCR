from cmath import exp
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-i", "--inputdir", default="../result_of_VitTCR/1.Prediction")
parser.add_argument("-o", "--outputdir", default="../result_of_VitTCR/2.Correlation")
args = parser.parse_args()


def mkidr(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__=="__main__":
    mkidr(path = args.outputdir)    
    
    for repeat in range(5):
        for fold in range(5):
            donors = []
            hlas = []
            spearmanrs = []
            pvalues = []
            exprs=[]

            allfiles = os.listdir(args.inputdir)
            subsets = [i for i in allfiles if 'repeat{}_fold{}.csv'.format(repeat, fold) in i]
            if len(subsets)>0:
                for root, dirs, files in os.walk(args.inputdir):           
                    for file in files:        
                        if 'repeat{}_fold{}.csv'.format(repeat, fold) in file:
                            donor = file.split('_')[1]
                            hla = file.split('_')[3]
                            expr='repeat{}_fold{}'.format(repeat, fold)
                            inputfile = "{}/{}".format(root,file)
                            df = pd.read_csv(inputfile)
                            coef, p = spearmanr(df.prob_1, df.percent)
                            donors.append(donor)
                            hlas.append(hla)
                            exprs.append(expr)
                            
                            spearmanrs.append(coef)
                            pvalues.append(p)
                cor_pred_percent = pd.DataFrame({'Donor':donors, 'HLA': hlas, 'Repeat':exprs, 'spearmanr':spearmanrs, 'p-value':pvalues})
                cor_pred_percent.to_csv('{}/Spearmanr_repeat{}_fold{}.csv'.format(args.outputdir, repeat, fold), index=False)
            else:
                print('Files does not exist')

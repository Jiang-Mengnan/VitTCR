from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-i", "--inputdir", default="../result_of_VitTCR/2.Correlation")
parser.add_argument("-o", "--outputdir", default="../result_of_VitTCR/3.Summary")
parser.add_argument("-m", "--modelname", default="VitTCR")
parser.add_argument("-r", "--restriction", default="A0201")
args = parser.parse_args()


def mkidr(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__=="__main__":
    mkidr(path = args.outputdir)
    aimpath = '{}/Summary_{}_{}.tsv'.format(args.outputdir, args.modelname, args.restriction)
    
    if not os.path.exists(aimpath):
        df_total=pd.DataFrame() 
        for repeat in range(5):
            for fold in range(5):
                filename='{}/Spearmanr_repeat{}_fold{}.csv'.format(args.inputdir,repeat, fold)
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    df_temp=df[df.HLA==args.restriction]                    

                    df_temp['expr']='repeat{}_fold{}'.format(repeat,fold)
                    df_total = pd.concat([df_total, df_temp], axis=0)
        df_total.to_csv(aimpath, sep='\t', index=False)

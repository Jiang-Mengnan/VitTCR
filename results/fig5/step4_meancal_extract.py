from unittest.case import DIFF_OMITTED
import pandas as pd 
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-pr", "--probability", default="../result_of_VitTCR/1.Prediction")
parser.add_argument("-cr", "--coefficient", default="../result_of_VitTCR/2.Correlation")
parser.add_argument("-r", "--restriction", default="A0201")
args = parser.parse_args()



if __name__=='__main__':
    ################################################################
    # Calculate the mean of predicted probabilities and affinities.
    ################################################################
    for donorID in [1,2]:
        hla_restriction = args.restriction

        df_total = pd.DataFrame()
        for repeat in range(5):
            for fold in range(5):
                df=pd.read_csv('{}/Donor_{}_HLA_{}_repeat{}_fold{}.csv'.format(args.probability, donorID, hla_restriction, repeat, fold))
                df['expr']='repeat{}_fold{}'.format(repeat, fold)
                df_total=pd.concat((df_total, df), axis=0)
        
        group = df_total.groupby('cdr3')

        prob_1_mean = pd.DataFrame(group['prob_1'].mean())
        prob_1_mean['cdr3']=prob_1_mean.index
        prob_1_mean=prob_1_mean.reset_index(drop=True)

        percent_mean = pd.DataFrame(group['percent'].mean())
        percent_mean['cdr3']=percent_mean.index
        percent_mean=percent_mean.reset_index(drop=True)

        df_mean=pd.merge(prob_1_mean, percent_mean,on='cdr3')
        df_mean.to_csv('{}/Average_Donor_{}_HLA_{}.csv'.format(args.probability, donorID, hla_restriction),index=False)

    
    ################################################################
    #  Extract the correlation coefficients for each Donor
    ################################################################
    df_total=pd.DataFrame()
    for repeat in range(5):
        for fold in range(5):
            df=pd.read_csv('{}/Spearmanr_repeat{}_fold{}.csv'.format(args.coefficient, repeat, fold))
            df['expr']='repeat{}_fold{}'.format(repeat, fold)
            df_total=pd.concat((df_total,df), axis=0)

    for donorID in [1,2]:
        df_donor=df_total[(df_total.Donor==donorID) & (df_total.HLA==args.restriction)]
        df_donor.to_csv('{}/Extracted_Spearmanr_Donor{}_HLA_{}.csv'.format(args.coefficient, donorID, args.restriction), index=False)
            

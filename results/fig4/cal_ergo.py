from textwrap import indent
import numpy as np 
import pandas as pd 
import os, time



def ppv(inpath):
    df_ppv=pd.DataFrame(columns=['TP', 'FP', 'TN', 'FN'])
    for repeat in range(5):
        for fold in range(5):
            df_repeat=pd.read_csv('{}/allinfo_repeat{}_fold{}.tsv'.format(inpath, repeat, fold),sep='\t')
            info=df_repeat.confusion_list.value_counts().to_frame()
            info.columns=['repeat{}_fold{}'.format(repeat,fold)]
            info = info.T
            df_ppv = pd.concat((df_ppv, info), axis=0)
    df_ppv['TP'].fillna(0, inplace=True) 
    df_ppv['FP'].fillna(0, inplace=True) 
    df_ppv['TN'].fillna(0, inplace=True) 
    df_ppv['FN'].fillna(0, inplace=True) 
    df_ppv=df_ppv.drop(df_ppv[df_ppv.TP==0].index)
    df_ppv=df_ppv.drop(df_ppv[df_ppv.FP==0].index)
    df_ppv=df_ppv.drop(df_ppv[df_ppv.TN==0].index)
    df_ppv=df_ppv.drop(df_ppv[df_ppv.FN==0].index)
    
    percent_PPVs=[]
    percent_PPV1s=[]
    for ind in df_ppv.index:
        counts_of_TP=df_ppv.loc[ind]['TP']
        counts_of_FP=df_ppv.loc[ind]['FP']
        percent_PPV="{:.2%}".format(counts_of_TP/(counts_of_TP+counts_of_FP))
        percent_PPV1="{:.4f}".format(counts_of_TP/(counts_of_TP+counts_of_FP))
        percent_PPVs.append(percent_PPV)
        percent_PPV1s.append(percent_PPV1)
    df_ppv['PPV']=percent_PPVs
    df_ppv['PPV1']=percent_PPV1s
    
    return df_ppv


if __name__=='__main__':
    # =====
    # PPV
    # =====
    inpath = "../clustered/ERGO_ae_healthy"
    df_ppv_trainset_only = ppv(inpath='{}/{}'.format(inpath, 'PPV_trainset_only'))
    df_ppv_trainset_only['Type'] = "Trainset-only"
    df_ppv_trainset_only['Fold'] = df_ppv_trainset_only.index
    # print(df_ppv_trainset_only)

    inpath = "../clustered/ERGO_ae_healthy"
    df_ppv_both_clustered = ppv(inpath='{}/{}'.format(inpath, 'PPV_both_clustered'))
    df_ppv_both_clustered['Type'] = "Clustered"
    df_ppv_both_clustered['Fold'] = df_ppv_both_clustered.index
    # print(df_ppv_both_clustered)

    inpath = "../original/ERGO_ae_healthy"
    df_ppv_original = ppv(inpath='{}/{}'.format(inpath, 'PPV_original'))
    df_ppv_original['Type'] = "Original"
    df_ppv_original['Fold'] = df_ppv_original.index
    # print(df_ppv_original)

    inpath = "../original/ERGO_ae_healthy"
    df_ppv_testset_only = ppv(inpath='{}/{}'.format(inpath, 'PPV_testset_only'))
    df_ppv_testset_only['Type'] = "Testset-only"
    df_ppv_testset_only['Fold'] = df_ppv_testset_only.index

    df_ppv_all = pd.concat([df_ppv_original, df_ppv_trainset_only, df_ppv_testset_only, df_ppv_both_clustered], axis=0)
    

    # ===============
    # AUROC and AUPR
    # ===============
    inpath = "../clustered/ERGO_ae_healthy"
    df_auroc_aupr_both_clustered = pd.read_csv('{}/performance_summary_independent_both_clustered.tsv'.format(inpath), sep='\t')
    df_auroc_aupr_both_clustered = df_auroc_aupr_both_clustered[['val_loss_min_index_AUC_test', 'val_loss_min_index_PR_test']]
    df_auroc_aupr_both_clustered = df_auroc_aupr_both_clustered.iloc[0:25]
    df_auroc_aupr_both_clustered['Type'] = "Clustered"
    df_auroc_aupr_both_clustered['Fold'] = df_ppv_testset_only.index

    inpath = "../clustered/ERGO_ae_healthy"
    df_auroc_aupr_trainset_only = pd.read_csv('{}/performance_summary_independent_trainset_only.tsv'.format(inpath), sep='\t')
    df_auroc_aupr_trainset_only = df_auroc_aupr_trainset_only[['val_loss_min_index_AUC_test', 'val_loss_min_index_PR_test']]
    df_auroc_aupr_trainset_only = df_auroc_aupr_trainset_only.iloc[0:25]
    df_auroc_aupr_trainset_only['Type'] = "Trainset-only"
    df_auroc_aupr_trainset_only['Fold'] = df_ppv_testset_only.index

    inpath = "../original/ERGO_ae_healthy"
    df_auroc_aupr_testset_only = pd.read_csv('{}/performance_summary_independent_testset_only.tsv'.format(inpath), sep='\t')
    df_auroc_aupr_testset_only = df_auroc_aupr_testset_only[['val_loss_min_index_AUC_test', 'val_loss_min_index_PR_test']]
    df_auroc_aupr_testset_only = df_auroc_aupr_testset_only.iloc[0:25]
    df_auroc_aupr_testset_only['Type'] = "Testset-only"
    df_auroc_aupr_testset_only['Fold'] = df_ppv_testset_only.index

    inpath = "../original/ERGO_ae_healthy"
    df_auroc_aupr_original = pd.read_csv('{}/performance_summary_independent_original.tsv'.format(inpath), sep='\t')
    df_auroc_aupr_original = df_auroc_aupr_original[['val_loss_min_index_AUC_test', 'val_loss_min_index_PR_test']]
    df_auroc_aupr_original = df_auroc_aupr_original.iloc[0:25]
    df_auroc_aupr_original['Type'] = "Original"
    df_auroc_aupr_original['Fold'] = df_ppv_testset_only.index

    df_auroc_aupr = pd.concat([df_auroc_aupr_original, 
                               df_auroc_aupr_trainset_only, 
                               df_auroc_aupr_testset_only, 
                               df_auroc_aupr_both_clustered], axis=0)
    
    
    # ================
    # Split and merge
    # ================
    df_auroc = pd.DataFrame()
    df_auroc['Fold'] = df_auroc_aupr['Fold']
    df_auroc['Value'] = df_auroc_aupr['val_loss_min_index_AUC_test']
    df_auroc['Type'] = df_auroc_aupr['Type']
    df_auroc['Metric'] = "AUROC"
    print(df_auroc)

    df_aupr = pd.DataFrame()
    df_aupr['Fold'] = df_auroc_aupr['Fold']
    df_aupr['Value'] = df_auroc_aupr['val_loss_min_index_PR_test']
    df_aupr['Type'] = df_auroc_aupr['Type']
    df_aupr['Metric'] = "AUPR"
    print(df_aupr)

    df_ppv = pd.DataFrame()
    df_ppv['Fold'] = df_ppv_all['Fold']
    df_ppv['Value'] = df_ppv_all['PPV1']
    df_ppv['Type'] = df_ppv_all['Type']
    df_ppv['Metric'] = 'PPV'
    print(df_ppv)

    df_total = pd.concat([df_auroc, df_aupr, df_ppv], axis=0)
    df_total.to_csv('metrics_ergo.tsv', sep='\t', index=False)

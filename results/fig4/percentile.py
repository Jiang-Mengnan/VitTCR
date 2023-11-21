import pandas as pd 
import numpy as np


#========
# train
#========
# summary
train_original_train = pd.read_csv('../../datasets/cross_validation/fold0/train.tsv', sep='\t')
train_original_valid = pd.read_csv('../../datasets/cross_validation/fold0/test.tsv', sep='\t')
train_original = pd.concat([train_original_train, train_original_valid], axis=0)

train_clustered_train = pd.read_csv('../../datasets/cross_validation_clustered/fold0/train.tsv', sep='\t')
train_clustered_valid = pd.read_csv('../../datasets/cross_validation_clustered/fold0/test.tsv', sep='\t')
train_clustered = pd.concat([train_clustered_train, train_clustered_valid], axis=0)

train_original = pd.DataFrame(train_original['peptide'].value_counts())
train_original.columns = ['cogtinate_cdr3_counts']
train_original['epitope'] = train_original.index
rank = train_original['cogtinate_cdr3_counts'].rank(method='min', ascending=False)
train_original['rank'] = rank

# calculate the percentile
rank_info = pd.DataFrame({'rank':list(set(rank))})
percentile = rank_info['rank'].rank(method='min', ascending=False)
rank = rank_info['rank'].rank(method='min', ascending=False)
percentile = (rank / len(rank_info)) * 100
rank_info['percentile'] = percentile
train_original = pd.merge(train_original, rank_info, on='rank')

# connect the percentile info with filtered epitopes
train_clustered = pd.DataFrame(train_clustered['peptide'].value_counts())
train_clustered['epitope'] = train_clustered.index
train_clustered = pd.merge(train_clustered, train_original, on='epitope')
train_clustered = train_clustered[['epitope', 'percentile']]
train_clustered.to_csv('filtered_epitope_percentile_train.tsv',sep='\t')

train_original = train_original[['epitope','percentile']]
train_original.to_csv('original_epitope_percentile_train.tsv',sep='\t')


#========
# test
#========
test_original  = pd.read_csv('../../datasets/independent_testset/independent_test/independent_testset.tsv', sep='\t')
test_clustered = pd.read_csv('../../datasets/independent_testset/independent_test_clustered/independent_test_clustered.tsv', sep='\t')

test_original = pd.DataFrame(test_original['peptide'].value_counts())
test_original.columns = ['cogtinate_cdr3_counts']
test_original['epitope'] = test_original.index
rank = test_original['cogtinate_cdr3_counts'].rank(method='min', ascending=False)
test_original['rank'] = rank

# calculate the percentile
rank_info = pd.DataFrame({'rank':list(set(rank))})
percentile = rank_info['rank'].rank(method='min', ascending=False)
rank = rank_info['rank'].rank(method='min', ascending=False)
percentile = (rank / len(rank_info)) * 100
rank_info['percentile'] = percentile
test_original = pd.merge(test_original, rank_info, on='rank')

# connect the percentile info with filtered epitopes
test_clustered = pd.DataFrame(test_clustered['peptide'].value_counts())
test_clustered['epitope'] = test_clustered.index
test_clustered = pd.merge(test_clustered, test_original, on='epitope')
test_clustered = test_clustered[['epitope', 'percentile']]
test_clustered.to_csv('filtered_epitope_percentile_test.tsv',sep='\t')

test_original = test_original[['epitope','percentile']]
test_original.to_csv('original_epitope_percentile_test.tsv',sep='\t')
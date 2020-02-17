from fastai.tabular  import *
import pandas as  pd
import pdb

trn_df = pd.read_csv('train.csv',index_col='ID_code')

procs = [FillMissing, Normalize]

#data = (TabularList.from_df(trn_df,procs=procs)
#        .split_by_rand_pct(0.33,seed=42)
#       .label_from_df('target')
#       .databunch(num_workers=16))

valid_ids = range(len(trn_df)-50000,len(trn_df))
data = TabularDataBunch.from_df('.',trn_df,dep_var='target',valid_idx=valid_ids,bs=128)

learn = tabular_learner(data,[100,2])

pdb.set_trace()
learn.lr_find()
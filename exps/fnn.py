from fastai.tabular import *
from fastai.metrics import *
from fastai.callbacks import CSVLogger
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pdb
import os

trn_df = pd.read_csv('../train.csv',index_col='ID_code')
tst_df = pd.read_csv('../test.csv',index_col='ID_code')

trn_df.reset_index(inplace=True)
trn_df.drop('ID_code',inplace=True,axis=1)

trn_1_rows = trn_df.loc[trn_df.target == 1].index
trn_0_rows = trn_df.loc[trn_df.target == 0].index
trn_rows = trn_1_rows[-15000:].append(trn_0_rows[:20000])
val_rows = trn_1_rows[:5098].append(trn_0_rows[-159902:])

procs = [Normalize]

trn_1 = trn_df.loc[trn_df.target == 1,:]
trn_0 = trn_df.where(trn_df.target == 0)[:len(trn_1)]
trn = trn_1.append(trn_0,ignore_index=True)

data = (TabularList.from_df(trn_df,procs=procs,cont_names=tst_df.columns)
        .split_by_idx(val_rows)
       .label_from_df('target')
       .add_test(TabularList.from_df(tst_df))
       .databunch(bs=64,num_workers=16))

# data = (TabularList.from_df(trn_df,procs=procs,cont_names=tst_df.columns)
#         .split_by_rand_pct(0.3,seed=42)
#        .label_from_df('target')
#        .add_test(TabularList.from_df(tst_df))
#        .databunch(bs=64,num_workers=16))

def struct_res_block(nf, dp,dense:bool=True, norm_type:Optional[NormType]=NormType.Batch, bottle:bool=True):
    "Resnet block of `nf` features fro structural data"
    #norm2 = norm_type
    #if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf//2 if bottle else nf
    return SequentialEx(nn.Linear(nf, nf_inner),
                        nn.ReLU(),
                        nn.BatchNorm1d(nf_inner),
                        nn.Linear(nf_inner, nf),
                        nn.ReLU(),
                        nn.BatchNorm1d(nf),
                        MergeLayer(dense),
                        #nn.Linear(2*nf,no),
                        nn.ReLU(),
                        nn.Dropout(0.1*dp),
                        nn.BatchNorm1d(2*nf),)

class struct_resnet(nn.Module):
    def __init__(self,lyrs,dps):
        super().__init__()
        self.lyrs = [struct_res_block(lyrs[i],dps[i]) for i in range(0,len(lyrs))]
        self.body = nn.Sequential(*self.lyrs)
        self.head = nn.Sequential(nn.Linear(2*lyrs[len(lyrs)-1],256),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(256),
                                    #nn.Dropout(0.1),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(64),
                                    #nn.Dropout(0.1),
                                    nn.Linear(64,2)
                                )
        self.net = nn.Sequential(self.body,self.head)

    def forward(self,A,B):
        return self.net(B)

#learn = tabular_learner(data,[512,256,128,64,32,8],metrics=accuracy,callback_fns=[CSVLogger])#256,128,64,8

learn = Learner(data,struct_resnet([200,400,800],dps=[0,0,0]),loss_func=nn.CrossEntropyLoss(),metrics=[accuracy,Precision(),Recall(),FBeta(beta=1)],callback_fns=[CSVLogger])
#learn.load('fnn')

#pdb.set_trace()

#learn.lr_find()
#learn.model.load_state_dict(torch.load('../models/balanced_resnet_500ep'))#models/balanced_resnet_500ep

learn.fit_one_cycle(10,max_lr=0.1)
learn.fit_one_cycle(10,max_lr=0.1)
learn.fit_one_cycle(10,max_lr=0.1)
learn.fit_one_cycle(10,max_lr=0.1)
learn.fit_one_cycle(10,max_lr=0.1)

p = learn.save('resnet',return_path=True)
torch.save(learn.model.state_dict(),'models/Resnet_new')
print(p)

# data1 = (TabularList.from_df(trn_df,procs=procs,cont_names=tst_df.columns)
#         .split_by_rand_pct(0.3,seed=42)
#        .label_from_df('target')
#        .add_test(TabularList.from_df(tst_df))
#        .databunch(bs=64,num_workers=16))

# learn.data = data1

#os.rename('history.csv','history_stage1.csv')

#learn.fit_one_cycle(500,max_lr=slice(0.01))

#p = learn.save('resnet_stage2',return_path=True)
#torch.save(learn.model.state_dict(),'models/Resnet_new_stage2')
#print(p)
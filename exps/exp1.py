from fastai.tabular import *
import pandas as pd
import numpy as np
import pdb 

trn_df = pd.read_csv('../train.csv',index_col='ID_code')
tst_df = pd.read_csv('../test.csv',index_col='ID_code')

trn_1 = trn_df.loc[trn_df.target == 1,:][:10]
trn_0 = trn_df.where(trn_df.target == 0)[:10]
trn = trn_1.append(trn_0,ignore_index=True)

trn.target = trn.target.astype(int)

trn_ = trn_df[:20]
procs = [Normalize]

data = (TabularList.from_df(trn,procs=procs,cont_names=tst_df.columns)
        .split_none()
       .label_from_df('target')
       .add_test(TabularList.from_df(tst_df))
       .databunch(bs=20,num_workers=2))

# def net_block(in_,out):
#     return nn.Sequential(nn.Linear(in_,out),
#                      nn.ReLU(inplace=True),
#                      nn.BatchNorm1d(out))

# class Mod(nn.Module):
#     def __init__(self,layers):
#         super().__init__()
#         net_blks = []
#         for i in range(0,len(layers)-1):
#             net_blks.append(net_block(layers[i],layers[i+1]))
#         self.mod = nn.Sequential(*net_blks)
        
#     def forward(self,A,B):
#           return self.mod(A[1])

# model = Mod([200,350,200,100,64,8,2])

# learn = Learner(data,model)

learn = tabular_learner(data,[200,160,50,8],metrics=accuracy)
#pdb.set_trace()

#learn.lr_find()
learn.fit_one_cycle(3000,max_lr=1)
pth = learn.save('exp1',return_path=True)
print(pth)

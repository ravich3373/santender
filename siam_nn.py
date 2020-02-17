from fastai.tabular import *
from fastai.metrics import accuracy
from fastai.layers import BCEWithLogitsFlat
from fastai.callbacks import CSVLogger
import pandas as pd
import pdb
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

from functional import seq


def is_even(num): return num % 2 == 0

class SiamDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.targets = ds.y.items
        self.candidates_1 = list(np.where(self.targets == 1)[0])
        self.candidates_0 = list(np.where(self.targets == 0)[0])
        self.candidates = [self.candidates_0,self.candidates_1]
    def __len__(self):
        return 2 * len(self.ds)
    def __getitem__(self, idx):
        if is_even(idx):
            return self.sample_same(idx // 2)
        else: return self.sample_different((idx-1) // 2)
    def sample_same(self, idx):
        target_val = self.targets[idx]        
        self.pick = np.random.randint(0,len(self.candidates[ target_val ]))
        return self.construct_example(self.ds[idx][0], self.ds[self.candidates[target_val][self.pick]][0], 1)
    def sample_different(self, idx):
        target_val = self.targets[idx]

        self.pick = np.random.randint(0,len(self.candidates[ (target_val+1)%2 ]))
        return self.construct_example(self.ds[idx][0], self.ds[self.candidates[(target_val+1)%2][self.pick]][0], 0)
    
    def construct_example(self, A, B, class_idx):
        return [A, B], class_idx

class linrelbn(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.lin = nn.Linear(x,y)
        self.rel = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(y)

    def forward(self,A):
        return  self.bn(self.rel(self.lin(A)))

class SiamModel(nn.Module):
    def __init__(self,lyrwds):
        super().__init__()
        self.net_ = nn.Sequential(*[linrelbn(lyrwds[i],lyrwds[i+1]) for i in range(0,len(lyrwds)-1)])
        self.head = nn.Linear(2,1)
        
    def forward(self,A,B):
        x1,x2 = seq(A[1],B[1]).map(self.net_)
        return self.head((x1-x2).pow(2))

trn_df = pd.read_csv('train.csv',index_col='ID_code')
tst_df = pd.read_csv('test.csv',index_col='ID_code')

procs = [Normalize]

data = (TabularList.from_df(trn_df,procs=procs,cont_names=tst_df.columns)
        .split_by_rand_pct(0.33,seed=42)
       .label_from_df('target')
       .add_test(TabularList.from_df(tst_df))
       .databunch(num_workers=16))

bs = 256
num_workers = 16

trn_dl = DataLoader(SiamDataset(data.train_ds),
                   batch_size=bs
                    ,num_workers=num_workers)
val_dl = DataLoader(SiamDataset(data.valid_ds),
                   batch_size=bs
                    ,num_workers=num_workers)


dbunch = TabularDataBunch(trn_dl,val_dl)

model = SiamModel([200,160,80,2]) #200,350,200,64,8,2
#model    

learn = Learner(dbunch,model,loss_func=BCEWithLogitsFlat(),metrics=[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)],callback_fns=[CSVLogger])


learn.fit_one_cycle(200,max_lr=10)

pth = learn.save('5000eps_siam_nn_1',return_path=True)
import sys
import os

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import argparse
import json
import random
import torch
import glob
import mlflow

import numpy as np

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from mlflow.tracking import MlflowClient

from aiutils import *

#####################################################################
# DATASET
#####################################################################

class CustomDataset(Dataset):
    
    def __init__(
        self,x,y
    ):
        x=x.iloc[:,:].values
        y=y.values
        
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
            
    def __len__(self):
        return len(self.y)
    
    def __getitem__( self, ii ):
        return {'x': self.x[ii], 'y': self.y[ii]}

#####################################################################
# DATALOADER
#####################################################################

def get_dataloader(
    x,y,
    batch_size=100,
    num_workers=1
):
    dataset = CustomDataset(x,y)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataloader


#####################################################################
# MODEL
#####################################################################

from torch import nn

class Regressor(nn.Module):
    
    def __init__(self, num_features=4,n_output=1):
        """
        """
        super(Regressor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(500, n_output),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

#####################################################################
# TRAIN LOOP
#####################################################################


def train_loop(
    model,
    train_dataloader,
    vali_dataloader,
    n_epochs = 300,
    weights_fn = './last.pt',
    optimizer = 'Here add something like > optim.Adam(model.parameters(), lr=0.0001)',
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    criterion = nn.MSELoss(),
    metrics_lst = [],
    metrics_func_lst = [],
    print_every_n_epochs = 10
):
    
    os.makedirs( os.path.dirname( weights_fn ) , exist_ok=True )
    
    model = model.to(device)
    
    mode_lst = ['train','vali']
    
    metrics_lst = ['loss'] + metrics_lst
    metrics_func_lst = [criterion] + metrics_func_lst

    metrics_dict = {
        mode:{ met:[] for met in metrics_lst }
        for mode in mode_lst
    }
    
    for epoch in range(n_epochs):

        for mode in mode_lst:

            if mode=='train':
                aux = model.train()
                dataloader = train_dataloader
            else:
                aux = model.eval()
                dataloader = vali_dataloader

            metrics_batch = { met:[] for met in metrics_lst }
            
            for sample in dataloader:

                x = sample['x'].to( device )
                y = sample['y'].to( device )

                pred = model.forward(x)
                loss = criterion( pred , y)
                
                if mode=='train':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                for f,met in zip( metrics_func_lst, metrics_lst ):
                    
                    if met=='loss':
                        metrics_batch[met].append( loss.item() )
                    else:
                        metrics_batch[met].append( f(pred,y).item() )

            for met in metrics_lst:
                
                metrics_dict[mode][met].append( np.mean(metrics_batch[met]) )
                
        # Save weigths
        s_dict = model.state_dict()
        torch.save( s_dict , weights_fn )
        
        if print_every_n_epochs:
            
            if epoch%print_every_n_epochs==0:
                
                print('*********************')
                print(f'epoch\t\t{epoch}')
                
                for mode in mode_lst:
                    
                    for met in metrics_lst:
                        
                        print(f'{mode}_{met}\t\t{metrics_dict[mode][met][-1]}')
            
        for mode in mode_lst:
                
            for met in ['loss']+metrics_lst:
                    
                curr_epoch_met = metrics_dict[mode][met][-1]
                # print(f"{mode}{met}", curr_epoch_met, epoch)
                #writer.add_scalar(f"{mode}/{met}", curr_epoch_met, epoch)
                mlflow.log_metric(f"{mode}M{met}", curr_epoch_met, epoch)
    
    results = {
        **{ 'train'+k:metrics_dict['train'][k] for k in metrics_dict['train']},
        **{ 'vali'+k:metrics_dict['vali'][k] for k in metrics_dict['vali']}
    }
    
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='./model/last.pt', help='weights path')
    parser.add_argument('--outputpath', type=str, default='./model', help='json config file')
    parser.add_argument('--instrument', type=str, default='CBOT.ZS', help='SOYA or CORN code')
    
    parser.add_argument('--random_state', type=int, default=123, help='random_state')
    parser.add_argument('--batch_sz', type=int, default=100, help='batch_sz')
    parser.add_argument('--n_epochs', type=int, default=1000, help='n_epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    
    opt = parser.parse_args()
    
    df = add_features(read_data())
    
    df_obs = df[df['observation']=='Settle']
    
    feats = ['Ndays', 'Nmatmonth', 'Nmatyear']
    
    X = df_obs[df_obs['instrument']==opt.instrument][feats]
    y = df_obs[df_obs['instrument']==opt.instrument][['value']]
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,random_state=opt.random_state)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,random_state=opt.random_state)


    print('Getting dataloaders...')
    
    train_dataloader = get_dataloader(
        X_train, y_train,
        batch_size=opt.batch_sz,
        num_workers=1
    )
    
    vali_dataloader = get_dataloader(
        X_val, y_val,
        batch_size=opt.batch_sz,
        num_workers=1
    )
    
    print('Building model...')
    
    model = Regressor(
        num_features=len(feats),
        n_output=1
    )
    
    print('Start training...')
    
    client = MlflowClient()
    experiment_name = "TorchModelComplex2"
    
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        # already exists
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id=current_experiment['experiment_id']
    
    with mlflow.start_run(experiment_id = experiment_id):
    
        mlflow.log_param('batch',opt.batch_sz)
        mlflow.log_param('epoch',opt.n_epochs)
        mlflow.log_param('lr',opt.learning_rate)
        mlflow.log_param('type', ('SOYABEANS' if opt.instrument=='CBOT.ZS' else 'CORN'))
    
        results = train_loop(
            model,
            train_dataloader,
            vali_dataloader,
            n_epochs             = opt.n_epochs,
            weights_fn           = opt.weights,
            optimizer            = optim.Adam(model.parameters(), lr=opt.learning_rate),
            device               = 'cuda' if torch.cuda.is_available() else 'cpu',
            criterion            = nn.MSELoss(),
            metrics_lst          = [],
            metrics_func_lst     = [],
            print_every_n_epochs = 10
        )

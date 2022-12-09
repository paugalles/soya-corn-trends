import sys
import os
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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)


from aiutils import *

from train import Regressor, get_dataloader


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='./model/last.pt', help='weights path')
    parser.add_argument('--outputpath', type=str, default='./model', help='json config file')
    parser.add_argument('--instrument', type=str, default='CBOT.ZS', help='SOYA or CORN code')
    parser.add_argument('--random_state', type=int, default=123, help='random_state')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    
    opt = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df = add_features(read_data())
    
    df_obs = df[df['observation']=='Settle']
    
    feats = ['Ndays', 'Nmatmonth', 'Nmatyear']
    
    X = df_obs[df_obs['instrument']==opt.instrument][feats]
    y = df_obs[df_obs['instrument']==opt.instrument][['value']]
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,random_state=opt.random_state)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,random_state=opt.random_state)
    
    train_dataloader = get_dataloader(
        X_train, y_train,
        batch_size=int(10e10),
        num_workers=1
    )
    
    vali_dataloader = get_dataloader(
        X_val, y_val,
        batch_size=int(10e10),
        num_workers=1
    )
    
    test_dataloader = get_dataloader(
        X_test, y_test,
        batch_size=int(10e10),
        num_workers=1
    )
    
    print('Building model...')
    
    model = Regressor(
        num_features=len(feats),
        n_output=1
    )
    
    model.load_state_dict( torch.load(opt.weights) )
    
    model.eval()
    model.to(device)
    
    for enu,dataloader in enumerate([
        train_dataloader,
        vali_dataloader,
        test_dataloader
    ]):
        
        for sample in dataloader:
            
            x,y = sample['x'], sample['y']
            pred = model.forward(x.to(device))
            
            if enu==0:
                X_train['pred'] = [float(p) for p in pred.cpu().detach()]
            elif enu==1:
                X_val['pred'] = [float(p) for p in pred.cpu().detach()]
            elif enu==2:
                X_test['pred'] = [float(p) for p in pred.cpu().detach()]
    
    results = pd.DataFrame(data={
        'partition':['train','val','test'],
        'mean_abs_err':[mean_absolute_error(yy, xx['pred']) for xx,yy in zip([X_train, X_val, X_test],[y_train, y_val, y_test])],
        'mean_sqrt_err':[mean_squared_error(yy, xx['pred']) for xx,yy in zip([X_train, X_val, X_test],[y_train, y_val, y_test])],
        'median_abs_err':[median_absolute_error(yy, xx['pred']) for xx,yy in zip([X_train, X_val, X_test],[y_train, y_val, y_test])],
        'R2':[r2_score(yy, xx['pred']) for xx,yy in zip([X_train, X_val, X_test],[y_train, y_val, y_test])]
    })
    
    X_train['partition'] = 'train'
    X_val['partition'] = 'val'
    X_test['partition'] = 'test'
    
    y_train['partition'] = 'train'
    y_val['partition'] = 'val'
    y_test['partition'] = 'test'
    
    X = pd.concat([X_train, X_val, X_test])
    y = pd.concat([y_train, y_val, y_test])
    
    X['y'] = y.value
    
    print([(yy, xx['pred']) for xx,yy in zip([X_test],[y_test])])
    print(results)

    
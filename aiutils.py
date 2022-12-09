import math
import sys

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import (
    date,
    datetime,
    timedelta
)

# from backports.datetime_fromisoformat import MonkeyPatch
# MonkeyPatch.patch_fromisoformat()


EXP_MONTH_CODE = {c:i/11 for i,c in enumerate(iter('FGHJKMNQUVXZ'))}

MONTH2CODE = {
    'January':'F',
    'February':'G',
    'March':'H',
    'April':'J',
    'May':'K',
    'June':'M',
    'July':'N',
    'August':'Q',
    'September':'U',
    'October':'V',
    'November':'X',
    'December':'Z'
}

CODE2MONTH = {v: k for k, v in MONTH2CODE.items()}

def read_data( fn: str = 'commodities.csv' ):
    return pd.read_csv(fn)

def add_features(
    df: pd.DataFrame,
    extra_feats: bool = False,
) -> pd.DataFrame:
    """
    N as a prefix stands for normalized
    """
    # add days
    # dd = [date.fromisoformat(d[:-1] + '+00:00') for d in df['date']]
    dd = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ") for d in df['date']]
    
    dd = [(d-dd[0]).days for d in dd]
    df['days'] = dd
    
    # Add normalized days (season, part of the year)
    df['Ndays'] = [d/365 for d in dd]
    
    # Add maturity season and year
    df['Nmatmonth'] = [EXP_MONTH_CODE[m[0]] for m in df['maturity']]
    matyear = np.asarray([int(m[1:]) for m in df['maturity']])
    df['Nmatyear'] = [(m-matyear.min())/(matyear.max()-matyear.min()) for m in matyear]
    
    # Instrument CORN or SOYA
    df['Ninstrument'] = df['instrument']=='CBOT.ZS'
    df["Ninstrument"] = df["Ninstrument"].astype(int)
    
    # This, rather than feats, are for plotting later
    df['maturity_month'] = [CODE2MONTH[m[0]] for m in df['maturity']]
    df['maturity_year'] = [int(m[1:]) for m in df['maturity']]
    
    # High, Low, Settle to normval
    def hls(name):
        if name == 'High':
            return 1
        elif name == 'Settle':
            return 0.5
        elif name == 'Low':
            return 0
        
    df["Nobs"] = [hls(o) for o in df["observation"]]
    
    
    if extra_feats:
    
        def iscorn(val):
            return (1 if val=='CBOT.ZC' else 0)

        df['iscorn'] = [iscorn(v) for v in df['instrument'].values]

        degree = 10
        plst = range(degree)

        for p in plst:

            df[f'Ndays{p}']     = np.power(df['Ndays'],p/2)
            df[f'Nmatmonth{p}'] = np.power(df['Nmatmonth'],p/2)
            df[f'Nmatyear{p}']  = np.power(df['Nmatyear'],p/2)

        df['NdaysS']     = np.sin(df['Ndays']*np.pi)
        df['NmatmonthS'] = np.sin(df['Nmatmonth']*np.pi)
        df['NmatyearS']  = np.sin(df['Nmatyear']*np.pi)

        df['NdaysC']     = np.cos(df['Ndays']*np.pi)
        df['NmatmonthC'] = np.cos(df['Nmatmonth']*np.pi)
        df['NmatyearC']  = np.cos(df['Nmatyear']*np.pi)
    
    return df


def plot_prediciton(df_plot: pd.DataFrame, kind: str= 'CBOT.ZS') -> None:
    """
    Plots the prediction yearly trend and maturity trends.
    
    Args:
        df_plot. Dataframe containing the 
    """

    for kind in [kind]:
        
        fig, ax = plt.subplots(2, 1,figsize=(15,10))

        name = ('SOYABEANS' if kind=='CBOT.ZS' else 'CORN')
        
        ax[0].set_title(name)
        ax[1].set_title(name)

        handler1 = sns.scatterplot(
            data=df_plot[df_plot['instrument']==kind], x="days", y="pred", hue="maturity", legend='full',
            palette = sns.color_palette("Set1", len(df_plot[df_plot['instrument']==kind]['maturity'].unique())), # It is important that the colors are random here
            ax = ax[0]
        )

        ax[0].set_ylabel('value [USD]')
        ax[0].set_xlabel('day of the year 2021 when the operation occurs')

        plt.legend(ncol=13, bbox_to_anchor=(0.5, 0.99), loc="lower center")
        
        handler2 = sns.boxplot(
            data=df_plot[df_plot['instrument']==kind], 
            x="maturity_month", y="pred", hue="maturity_year", ax=ax[1],
            # boxprops=dict(alpha=.2)
        )

        ax[1].set_ylabel('value [USD]')
    
    return None


def plot_prediciton_vs_gt(df_plot: pd.DataFrame, kind: str= 'CBOT.ZS') -> None:
    """
    Plots the prediction yearly trend and maturity trends.
    
    Args:
        df_plot. Dataframe containing the 
    """

    for kind in [kind]:
        
        fig, ax = plt.subplots(1, 1,figsize=(15,10))

        name = ('SOYABEANS' if kind=='CBOT.ZS' else 'CORN')
        
        ax.set_title(name)

        handler2 = sns.boxplot(
            data=df_plot[df_plot['instrument']==kind], 
            x="maturity_month", y="pred", hue="maturity_year", ax=ax,
            boxprops=dict(alpha=.4)
        )

        ax.set_ylabel('value [USD]')
        ax.set_xlabel('day of the year 2021 when the operation occurs')

        plt.legend(ncol=13, bbox_to_anchor=(0.5, 0.99), loc="lower center")
        
        handler2 = sns.boxplot(
            data=df_plot[df_plot['instrument']==kind], 
            x="maturity_month", y="value", hue="maturity_year", ax=ax,
            boxprops=dict(alpha=.9)
        )
    
    return None
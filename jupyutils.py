import math

import tabulate

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from typing import List, Any
from IPython.display import HTML, display

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


def display_lst_as_table(data: List[List[Any]], title: str = 'Options per column') -> None:
    """
    Example:
        data = [["Sun",696000,1989100000],
                 ["Earth",6371,5973.6],
                 ["Moon",1737,73.5],
                 ["Mars",3390,641.85]]
                 
        display_lst_as_table(data)

    Args:
        data: A list of data to display
    """
    display(HTML(f'<h3>{title}</h3>'))
    table = tabulate.tabulate(data, tablefmt='html')
    display(HTML(table))


def generic_display(
    df_lst: List[pd.DataFrame],
    mode: str = 'hist',
    skip_cols: List[str] = ['date'],
    fig_ncol = 1,
    fig_nrow = 3
) -> None:
    """
    This can be used to plot histograms or scatterplots
    Args:
        df_lst: A list of dataframes
        mode: displaying mode, hist or scatter
        skip_cols: columns in the dataframe to skip
        fig_ncol: Number of suplot cols
        fig_nrow: Number of suplot rows
    """
    
    assert mode in ['hist', 'scatter'], f'Unsupported mode {mode}'
    
    fig_navg = (fig_ncol + fig_nrow)/2
    
    dfref = df_lst[0]

    colslst = [c for c in dfref.columns if c not in skip_cols]
    
    for enu, col in enumerate(colslst):
        
        i = enu%fig_nrow
        
        if i == 0:
            
            if enu != 0:
                plt.show()
                
            fig, axes = plt.subplots(
                1, fig_nrow,
                figsize=(
                    15*fig_nrow/fig_navg,
                    15*fig_ncol/fig_navg
                ),
                squeeze = False
            )
            
            fig.autofmt_xdate(rotation=45)
        
        nopts = '/'.join([f'{len(df[col].unique())}' for df in df_lst])
        
        ax = axes.flatten()
        
        ax[i].set_title(f'{col}:{nopts}')
        
        for df in df_lst:
            
            if mode == 'hist':
                
                ax[i].hist(df[col], edgecolor='None', alpha = 0.5)
                
            elif mode=='scatter':
                
                if not 'days' in df:
                    print('Warning, no days in df, calculating it with add_days_col...')
                    print('Warning, this assumes that the earliest day is the same for Soya and corn')
                    df = add_days_col(df)
                
                ax[i].scatter(df['days'], df[col], edgecolor='None', alpha = 0.5)
        
        fig.autofmt_xdate()
        
def hist(
    df_lst: List[pd.DataFrame],
    skip_cols: List[str] = ['date'],
) -> None:
    """This is a particularization of 'generic display' for mode 'hist'"""
    generic_display(df_lst, mode='hist', skip_cols=skip_cols)

def add_days_col(df: pd.DataFrame) -> pd.DataFrame:
    """Adds days to dataframe, a transformation of datetime into days from first date"""
    dd = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ") for d in df['date']]
    # dd = [datetime.fromisoformat(d[:-1] + '+00:00') for d in df['date']]
    dd = [(d-dd[0]).days for d in dd]
    df['days'] = dd
    return df
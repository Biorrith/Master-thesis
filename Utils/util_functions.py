import pandas as pd

def fill_dx(subj_df):
    # # if(not series.isnull().sum().sum()):
    #     # return series
    # #Interpolate missing values between two identical diagnoses
    # forward = series.ffill()
    # backward = series.bfill()
    # filled = forward.where(forward == backward)  # Interpolates between same diagnoses    
    # return filled
    subj_df = subj_df.sort_values(by='M')
    for i in range(1, len(subj_df) - 1):
        if pd.isna(subj_df.iloc[i]['DX']):
            if subj_df.iloc[i-1]['DX'] == subj_df.iloc[i+1]['DX']:
                subj_df.at[subj_df.index[i], 'DX'] = subj_df.iloc[i-1]['DX']
    return subj_df


def filter_rows_event(group):
    #If DX at baseline
    if any(((group['DX'] == 2) & (group['M'] == 0))):
        return pd.DataFrame()

    #If only one measurement
    if group.shape[0] < 2:
        return pd.DataFrame()

    #Remove all after first diagnosis
    min_year = group[group['DX'] == 2]['M'].min()
    if pd.notna(min_year):
        return group[group['M'] <= min_year]
    return group

def get_baseline(group):
    baseline = group.iloc[0].copy()
    baseline['Event'] = group.iloc[-1]['Event']
    baseline['M'] = group.iloc[-1]['M']
    return baseline

def get_events(df_inp, filter=True, only_baseline=True):
    df = df_inp.copy()
    df = df.dropna(subset=['DX'])
    df['Event'] = (df['DX']==2).astype(bool)
    if filter:
        df = df.sort_values(['RID', 'M']).groupby('RID').apply(filter_rows_event).reset_index(drop=True)
    if only_baseline:
        df = df.sort_values(['RID', 'M']).groupby('RID').apply(get_baseline).reset_index(drop=True)        
    return df


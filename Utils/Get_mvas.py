import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing


sys.path.append("../")
from Utils.util_functions import fill_dx, get_events, get_baseline

mvas_mri_MCI_path = '../Datasets/MVAS/image_roi_means_MCI.csv'
mvas_demography_MCI_path = '../Datasets/MVAS/demography_MCI.csv'
mvas_ftp_path = '../Datasets/MVAS/FTP_median_ADNI_rois.csv'
mvas_apoe4_demography_path = '../Datasets/MVAS/APOE4/demography_APOE4.csv'
mvas_apoe4_image_roi_means_path = '../Datasets/MVAS/APOE4/image_roi_means_APOE4.csv'


def combine_left_right(df):
    df['frontal_gm'] = df['frontal_gm_left'] + df['frontal_gm_right']
    df['frontal_wm'] = df['frontal_wm_left'] + df['frontal_wm_right']
    
    df['temporal_gm'] = df['temporal_gm_left'] + df['temporal_gm_right']
    df['temporal_wm'] = df['temporal_wm_left'] + df['temporal_wm_right']

    df['parietal_gm'] = df['parietal_gm_left'] + df['parietal_gm_right']
    df['parietal_wm'] = df['parietal_wm_left'] + df['parietal_wm_right']

    df['occipital_gm'] = df['occipital_gm_left'] + df['occipital_gm_right']
    df['occipital_wm'] = df['occipital_wm_left'] + df['occipital_wm_right']

    df['hippocampus'] = df['hippocampus_left'] + df['hippocampus_right']
    df['thalamus'] = df['thalamus_left'] + df['thalamus_right']
    df['caudate'] = df['caudate_left'] + df['caudate_right']
    df['putamen'] = df['putamen_left'] + df['putamen_right']
    return df



def get_mvas_single_param():
    mvas_mri = pd.read_csv(mvas_mri_MCI_path)
    columns = mvas_mri['param'].unique()
    drop_columns = ['PK_bp', 'PWI_CBF', 'PWI_CBV', 'PWI_CTH', 'PWI_MTT', 'PWI_RTH', 'FTP_SUVR']
    filtered = [item for item in columns if item not in drop_columns]
    mvas_mri = mvas_mri[mvas_mri['param'].isin(filtered)]

    mvas_demography = pd.read_csv(mvas_demography_MCI_path)
    mvas_ftp = pd.read_csv(mvas_ftp_path)
    mvas_ftp_demo = pd.merge(mvas_ftp, mvas_demography, on=['mr_id', 'visit'], how='outer').drop('param', axis=1)

    params = mvas_mri['param'].unique()
    mvas_mri.columns
    
    datasets = []
    for param in params:
        param_mri = mvas_mri[mvas_mri['param'] == param].drop(['param'], axis=1)
        merge = pd.merge(param_mri, mvas_ftp_demo, on=['mr_id', 'visit'], how='right')
        
        merge = rename_columns(merge)
        merge = combine_left_right(merge)


        drop_columns = ['frontal_gm_left', 'frontal_wm_left',
        'frontal_gm_right', 'frontal_wm_right', 'temporal_gm_left',
        'temporal_wm_left', 'temporal_gm_right', 'temporal_wm_right',
        'parietal_gm_left', 'parietal_wm_left', 'parietal_gm_right',
        'parietal_wm_right', 'occipital_gm_left', 'occipital_wm_left',
        'occipital_gm_right', 'occipital_wm_right', 'hippocampus_left',
        'hippocampus_right', 'thalamus_left', 'thalamus_right', 'caudate_left',
        'caudate_right', 'putamen_left', 'putamen_right',]
        
        if(param == 'Volume_mm3'):
            columns_to_divide = [col for col in merge.columns if col.endswith(('left', 'right'))]
            colummns_to_divide2 = ['hippocampus', 'thalamus', 'caudate', 'putamen']
            for column in columns_to_divide+colummns_to_divide2:
                merge[column] = merge[column]/merge['whole_brain']

        
        merge.drop(['whole_brain', 'nawm', 'gray_matter'], axis=1, inplace=True)
        
        
        merge_event = get_events(merge)
        datasets.append((param, merge, merge_event))
    return datasets


def rename_columns(df_inp):
    df = df_inp.copy()
    df['DX'] = df['Diagnosis'].replace({'healthy': 0, 'mci': 1, 'ad': 2})
    df['M'] = df['visit'].replace({'bl': 0, 'm24': 24})
    df['RID'] = df['mr_id']

    df.drop(['Diagnosis', 'visit', 'mr_id'], axis=1, inplace=True)
    return df


def flatten_mri_roi(df_inp):
    df = df_inp.copy()
    
    columns = df['param'].unique()
    drop_columns = ['PK_bp', 'PWI_CBF', 'PWI_CBV', 'PWI_CTH', 'PWI_MTT', 'PWI_RTH', 'FTP_SUVR', 'SEPWI_MTT', 'SEPWI_RTH']
    filtered = [item for item in columns if item not in drop_columns]
    df = df[df['param'].isin(filtered)]
    
    df = combine_left_right(df)

    drop_columns = ['frontal_gm_left', 'frontal_wm_left',
        'frontal_gm_right', 'frontal_wm_right', 'temporal_gm_left',
        'temporal_wm_left', 'temporal_gm_right', 'temporal_wm_right',
        'parietal_gm_left', 'parietal_wm_left', 'parietal_gm_right',
        'parietal_wm_right', 'occipital_gm_left', 'occipital_wm_left',
        'occipital_gm_right', 'occipital_wm_right', 'hippocampus_left',
        'hippocampus_right', 'thalamus_left', 'thalamus_right', 'caudate_left',
        'caudate_right', 'putamen_left', 'putamen_right', ]

    df.drop(drop_columns, axis=1, inplace=True)

    df_flattened = df.pivot_table(index=['mr_id', 'visit'], 
                                              columns='param', 
                                              values=df.columns[3:], 
                                              aggfunc='first')
    df_flattened.columns = [f'{col}_{param}' for param, col in df_flattened.columns]
    df_flattened.reset_index(inplace=True)
    columns_to_convert = df_flattened.columns.difference(['mr_id', 'visit'])
    df_flattened[columns_to_convert] = df_flattened[columns_to_convert].apply(pd.to_numeric, errors='coerce')


    min_max_scaler = preprocessing.MinMaxScaler()
    columns_to_divide = [col for col in df_flattened.columns if col.startswith('Volume_mm3')]
    for column in columns_to_divide:
        df_flattened[column] = df_flattened[column]/df_flattened['Volume_mm3_whole_brain']#)*10
        df_flattened[column] = min_max_scaler.fit_transform(df_flattened[[column]]) 


    remove_cols = ['whole_brain', 'nawm', 'gray_matter']
    columns_to_remove = [col for col in df_flattened.columns if any(s in col for s in remove_cols)]
    df_flattened.drop(columns_to_remove, axis=1, inplace=True)

    return df_flattened 


def get_mvas_combined_params():
    #Load in and remove non-used parameters
    mvas_mri = pd.read_csv(mvas_mri_MCI_path)
    mvas_demography = pd.read_csv(mvas_demography_MCI_path)
    mvas_ftp = pd.read_csv(mvas_ftp_path)
    mvas_ftp_demo = pd.merge(mvas_ftp, mvas_demography, on=['mr_id', 'visit'], how='outer').drop('param', axis=1)

    mvas_mri_flattened = flatten_mri_roi(mvas_mri)


    merge = pd.merge(mvas_mri_flattened, mvas_ftp_demo, on=['mr_id', 'visit'], how='inner')
    merge = rename_columns(merge)
    merge['COLPROT'] = 'MVAS'
    df_event = get_events(merge)
    


    apoe4_demo = pd.read_csv(mvas_apoe4_demography_path)
    apoe4_mri = pd.read_csv(mvas_apoe4_image_roi_means_path)
    apoe4_mri_flattened = flatten_mri_roi(apoe4_mri)
    apoe4_merge = pd.merge(apoe4_demo, apoe4_mri_flattened, on=['mr_id', 'visit'], how='inner')
    apoe4_merge = rename_columns(apoe4_merge)
    apoe4_merge['COLPROT'] = 'MVAS_APOE4'


    return merge, df_event, apoe4_merge


import pandas as pd
import numpy as np
import sys
import os

sys.path.append("../")
from Utils.util_functions import fill_dx, get_events, get_baseline
from Utils.Get_mvas import get_mvas_combined_params
from Utils.Get_adni import get_adni3, get_merge

mvas_mri_MCI_path = '../Datasets/MVAS/image_roi_means_MCI.csv'
mvas_demography_MCI_path = '../Datasets/MVAS/demography_MCI.csv'
mvas_ftp_path = '../Datasets/MVAS/FTP_median_ADNI_rois.csv'
mvas_apoe4_demography_path = '../Datasets/MVAS/APOE4/demography_APOE4.csv'
mvas_apoe4_image_roi_means_path = '../Datasets/MVAS/APOE4/image_roi_means_APOE4.csv'



def combine_mvas_adni(verbose=False):
    mvas_mci, _, mvas_apoe4 = get_mvas_combined_params()
    adni = get_adni3(verbose=verbose)
    
    pd.options.mode.chained_assignment = None
    #!Process ADNI3
    columns_to_keep = ['RID', 'AGE', 'PTEDUCAT', 'APOE4', 'CDRSB', 'ADAS11', 
                   'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'LDELTOTAL', 
                   'FAQ', 'MOCA', 'DX', 'PTGENDER_Male', 'M','ENTORHINAL_SUVR', 
                   'INFERIOR_TEMPORAL_SUVR', 'TAU_METAROI', ]
    adni = adni[columns_to_keep]
    adni['COLPROT'] = 'ADNI'
    adni['REMOVE'] = 0

    #! Process MVAS
    max_id = adni['RID'].max()
    id_changes_mci = {}
    mvas_mci['RID'] = mvas_mci['RID'].apply(
        lambda x: id_changes_mci.setdefault(x, x + max_id)
    )

    max_id = mvas_mci['RID'].max()
    id_changes_mci = {}
    mvas_apoe4['RID'] = mvas_apoe4['RID'].apply(
        lambda x: id_changes_mci.setdefault(x, x + max_id)
    )


    #Keep all subjects with DX=1 at baseline and remove all else
    mvas_mci_sa = get_events(mvas_mci)
    mvas_subjects = mvas_mci_sa[mvas_mci_sa['DX'] == 1]['RID'].unique()
    mvas_mci['REMOVE'] = mvas_mci['RID'].apply(lambda x: 0 if x in mvas_subjects else 1)
    mvas_apoe4['REMOVE'] = 1
    
    mvas = pd.concat([mvas_mci, mvas_apoe4], axis=0)

    combined_dict = {
    'sex_male': 'PTGENDER_Male',
    'age': 'AGE',
    'apoe4': 'APOE4', 
    'Diagnosis': 'DX', 
    'mmse': 'MMSE',
    'moca': 'MOCA', 
    'mr_id': 'RID',
    'education_years': 'PTEDUCAT',
    'Entorhinal': 'ENTORHINAL_SUVR', 
    'Inferior_temporal': 'INFERIOR_TEMPORAL_SUVR',
    'Meta_ROI': 'TAU_METAROI',
    'cdr_sob': 'CDRSB'
    }

    mvas = mvas.rename(columns=combined_dict)
    mvas_extra_subjects = mvas[~mvas['RID'].isin(mvas_subjects)]['RID'].unique()
    
    #! Combine the two
    merge = pd.concat([mvas, adni], ignore_index=True)
    if verbose:
        print(f"Events in MVAS: {get_events(mvas)['Event'].value_counts()}")
        print(f"Events in ADNI: {get_events(adni)['Event'].value_counts()}")
        print(f"Event in Merge: {get_events(merge)['Event'].value_counts()}")
    merge = pd.get_dummies(merge, 'COLPROT', drop_first=True)



    
    #Split all users that wont be used for training and testing in seperate dataframe
    #Get the MVAS subjects that were removed (APOE4, HC, AD)
    removed_rows = merge[merge['REMOVE'] == 1]
    extra_imputation_data = removed_rows[removed_rows['RID'].isin(mvas_extra_subjects)]
    extra_imputation_data['REMOVE'] = 1


    #Get M24 for MVAS MCI subjects
    extra_mvas_mci = merge[(merge['RID'].isin(mvas_subjects))&(merge['M']!=0)]
    extra_mvas_mci['REMOVE'] = 1
    
    #Merge
    merge = merge[merge['REMOVE'] == 0]
    merge_event = get_events(merge)


    pd.options.mode.chained_assignment = 'warn'
    return merge_event, extra_mvas_mci, extra_imputation_data#imputation_extra_other

def post_imputation_processing(df_train, df_test):
    
    if 'REMOVE' in df_train.columns:
        df_train = df_train[df_train['REMOVE'] == 0]
        df_train = df_train.drop('REMOVE', axis=1)
        
    if 'REMOVE' in df_test.columns:
        df_test = df_test[df_test['REMOVE'] == 0]
        df_test= df_test.drop('REMOVE', axis=1)

    # df_train = get_events(df_train)
    # df_test = get_events(df_test)
    
    df_train.drop(['DX', 'RID', 'COLPROT_MVAS', 'COLPROT_1', 'COLPROT_MVAS_APOE4'], axis=1, inplace=True, errors='ignore')
    df_test.drop(['DX',  'RID', 'COLPROT_MVAS', 'COLPROT_1', 'COLPROT_MVAS_APOE4'], axis=1, inplace=True, errors='ignore')

    return df_train, df_test
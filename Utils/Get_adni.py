
import pandas as pd
import numpy as np
from loguru import logger
import sys
import os
from sklearn import preprocessing

sys.path.append("../")
from Utils.util_functions import fill_dx, get_events
# from Data_imputation import get_merge

mvas_mri_MCI_path = '../Datasets/MVAS/image_roi_means_MCI.csv'
mvas_demography_MCI_path = '../Datasets/MVAS/demography_MCI.csv'
mvas_ftp_path = '../Datasets/MVAS/FTP_median_ADNI_rois.csv'
mvas_apoe4_demography_path = '../Datasets/MVAS/APOE4/demography_APOE4.csv'
mvas_apoe4_image_roi_means_path = '../Datasets/MVAS/APOE4/image_roi_means_APOE4.csv'
adni_merge_path = '../Datasets/ADNI/ADNIMERGE_02Apr2024.csv'

def get_adni3(pure=False, prepare_sa=False, verbose=False, reduce=True):
    pd.options.mode.chained_assignment = None

    adni_merge = get_merge(cutoff=0.99, cohort=['ADNI3'], 
                           drop_viscode=False, 
                           datapath=adni_merge_path, 
                           remove_dx=False,
                           fill_dx_manually=True,
                           pure=pure,
                           prepare_sa=prepare_sa)

    #Load in ADNI FTP and modify a bit
    adni_ftp = pd.read_csv('../Datasets/ADNI/BAIPETNMRCFTP_08_17_22_15May2024.csv')
    adni_ftp = adni_ftp[(adni_ftp['COLPROT'] == 'ADNI3') & (adni_ftp['ORIGPROT'] == 'ADNI3')]
    adni_ftp['VISCODE'] = adni_ftp['VISCODE2']

    adni_ftp.drop(['VISCODE2', 'EXAMDATE', 'LONIUID',
            'RUNDATE', 'STATUS', 'MODALITY', 
            'update_stamp', 'COLPROT', 'ORIGPROT'], axis=1,  inplace=True)

    #Combine ADNI
    adni3 = pd.merge(adni_merge, adni_ftp, on=['RID', 'VISCODE'], how='outer')
    adni3['M'] = adni3.groupby('RID')['M'].transform(lambda x: x - x.min())

    if verbose:
        print(f"ADNI3 shape before reducing: {adni3.shape}")
    """
    1. Keep all subjects that have an event.
    2. Keep a percentage of subjects who did not have an event, but had FTP at baseline
    """
    if reduce:
        #1. All subjects that have event
        event_df = get_events(adni3)
        event_df = event_df[event_df['DX'] == 1]
        event_subjects = event_df[event_df['Event'] == 1]['RID'].unique()
        # Step 2: Identify all subjects with non-missing TAU_METAROI at baseline and sample
        #Make sure no overlaps with event subjects
        
        #Sample half of the tau_subjects_bl_no_event
        np.random.seed(42)
            
        # remaining_tau = tau_measurements[~tau_measurements['RID'].isin(np.concatenate((tau_ids, event_subjects), axis=0))]
        
        #Combine the subjects
        mci_ids = event_df[(event_df['Event']==0) & (event_df['DX']==1)]['RID'].unique()

        ids = np.unique(np.concatenate((event_subjects, mci_ids), axis=0))
        if verbose:
            have_ftp = event_df[(event_df['Event']==0) & (event_df['DX']==1)]['TAU_METAROI'].dropna().shape[0]
            print(f"{len(ids)} users selected, of which {len(event_subjects)} have events") 
            print(f"{len(mci_ids)} non-event subjects with MCI, of which {have_ftp} had tau meausrement")

        #Get the final dataset
        filtered_adni = adni3[adni3['RID'].isin(ids)]
        filtered_adni['REMOVE'] = 0

        filtered_adni = filtered_adni.dropna(subset=['DX'])

        if verbose:
            print("ADNI3 shape after reducing: ", filtered_adni.shape)
        pd.options.mode.chained_assignment = 'warn'

        return filtered_adni
    
    else:
        return adni3



def get_merge(cutoff=0.80, fill_dx_manually=True, remove_dx=True, 
              cohort=['ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO'], pure=True,
              drop_viscode=True, datapath=adni_merge_path,  prepare_sa=False,
              keep_cohort=False, normalize_volume=True):
    df = pd.read_csv(datapath)
    
    rm_ids = [ 135,  166,  416,  429,  467,  555,  566,  702, 2130, 2210, 2274, 2367,
       4005, 4114, 4293, 4426, 4430, 4434, 4506, 4706, 4741, 4746, 4899, 4947,
       6222, 6535]
    df = df[~df['RID'].isin(rm_ids)]

    if pure:
        df = df[(df['ORIGPROT'].isin(cohort)) & (df['COLPROT'].isin(cohort))]
    else:
        df = df[df['COLPROT'].isin(cohort)]


    df['DX'] = df['DX'].replace({'CN': 0, 'MCI': 1, 'Dementia': 2})
    if prepare_sa:
        df = get_events(df, only_baseline=True, filter=True)
        df.drop(['RID'], axis=1, inplace=True)


    else:
        if fill_dx_manually:
            df = df.groupby('RID').apply(fill_dx).reset_index(drop=True)
            # df['DX'] = df.sort_values(['RID', 'M']).groupby(['RID'])['DX'].transform(fill_dx).reset_index(drop=True)
        if(remove_dx):
            df = df.dropna(subset=['DX'])

    columns = ['TAU','PTAU', 'ABETA']
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')


    unrelated_columns = ['PTID','EXAMDATE', 'IMAGEUID', 'update_stamp']
    collinear_columns = ['Years_bl', 'Month_bl', 'Month']
    if(drop_viscode):
        unrelated_columns.append('VISCODE')
    
    baseline_columns = ['CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 
                        'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'LDELTOTAL_BL',
                        'DIGITSCOR_bl', 'TRABSCOR_bl', 'FAQ_bl', 'mPACCdigit_bl',
                        'mPACCtrailsB_bl', 'Ventricles_bl', 'Hippocampus_bl', 
                        'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl',
                        'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl',
                        'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl',
                        'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl',
                        'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'FDG_bl', 'PIB_bl','AV45_bl', 'FBB_bl',
                        'EXAMDATE_bl', 'FLDSTRENG_bl', 'FSVERSION_bl', 'IMAGEUID_bl', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 
                        'DX_bl']

    #Drop columns
    df = df.drop(unrelated_columns + collinear_columns + baseline_columns, axis=1)


    #Remove all columns with over % of missing values
    logger.info(f"Missing value feature cutoff: {cutoff*100} %")
    logger.info(f"Removing columns with cutoff: {df.loc[:, df.isnull().mean() > cutoff].columns}")
    df = df.loc[:, df.isnull().mean() < cutoff]


    #Reducing dimensionality where it makes sense (PTMARRY and PTRACCAT)
    #Marry new categories: Married, Never married. Change unknown (60 subjects, 116 rows) to Never Married.
    df['PTMARRY'] = df['PTMARRY'].replace({'Unknown': 'Never married', np.nan: 'Never married'})
    df['PTMARRY_Never_married'] = (df['PTMARRY'] == 'Never married').astype(bool)
    df['PTMARRY_married'] = (df['PTMARRY'] == 'Married').astype(bool)
    df = df.drop(['PTMARRY'], axis=1)

    df['PTRACCAT'] = df['PTRACCAT'].replace({'Asian': 'Other', 'More than one': 'Other', 'Am Indian/Alaskan': 'Other', 'Hawaiian/Other PI': 'Other', 'Unknown': 'Other'})


    #One hot encoding
    caterogial_values = df.select_dtypes(include=['category', 'object']).columns
    if keep_cohort:
        caterogial_values = caterogial_values.drop('COLPROT', errors='ignore')
    caterogial_values = caterogial_values.drop('VISCODE', errors='ignore')

    logger.info(f"Categorical columns: {caterogial_values}")
    df = pd.get_dummies(df, columns=caterogial_values, drop_first=True)
    
    min_max_scaler = preprocessing.MinMaxScaler()

    if normalize_volume:
        # Normalize brain measurements by whole brain volume
        brain_measures = ['Ventricles', 'Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']
        for column in brain_measures:
            df[column] = df[column]/df['WholeBrain']
            df[column] = min_max_scaler.fit_transform(df[[column]]) 
        df.drop(['WholeBrain'], axis=1, inplace=True)
    

    logger.info(f"Data shape = {df.shape}")
    return df


from pathlib import Path
import yaml


def post_imputation_adni(path, only_first=False, verbose=True, dxs=[0, 1]):
    conf_path = Path(f"{path}setup.yaml")                   
    print(f'Configuration file: {conf_path}\n')
    if not conf_path.exists():
        print('Configuration file not found')
        return
    # else:
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    
    drop_columns = ['DX','SITE', 'COLPROT_ADNI2', 'COLPROT_ADNI3', 'COLPROT_ADNIGO', 'ORIGPROT_ADNI2', 
                    'ORIGPROT_ADNI3', 'ORIGPROT_ADNIGO', 'FLDSTRENG_3 Tesla MRI', 
                    'FSVERSION_Cross-Sectional FreeSurfer (6.0)',
                    'FSVERSION_Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)', 
                    'UniqueID', 'RID', 'PIB', 'FBB']
    
    dfs_train = []
    dfs_test = []

    for i in range(config['num_datasets']):
        if(verbose):
            print(f"Preparing dataset {i}...\n")
        df_train = pd.read_csv(f"{path}/dataset_{i}/train.csv")
        df_test = pd.read_csv(f"{path}/dataset_{i}/test.csv")  

        df_train = df_train[df_train['DX'].isin(dxs)]
        df_test = df_test[df_test['DX'].isin(dxs)]

        df_train.drop(columns=[col for col in drop_columns if col in df_train.columns], inplace=True)
        df_test.drop(columns=[col for col in drop_columns if col in df_test.columns], inplace=True)
        
        dfs_train.append(df_train)
        dfs_test.append(df_test)
        if only_first:
            break

    return dfs_train, dfs_test, config
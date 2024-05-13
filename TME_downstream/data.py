import pandas as pd
from utils import choose_single_vs_multi


def load_dataset(name='tme_only'):
    df_multi = pd.read_csv('/home/air/chenah/UNI_TME_regression/predictions/suny_predictions_multitask_UNI.csv')

    df = df_multi

    clin_path = '/mnt/synology/ICB_Data_SUNY/clinical_metadata/SUNY_FINAL_clinical_data_with_PFS_corrected_050224.tsv'
    df_clin = pd.read_csv(clin_path, sep='\t')
    df_clin = df_clin.rename(columns={"overall_resp":'response_label', 'case_id':'ID'})

    if name == 'tme_only':
        df_clin = df_clin[~df_clin['response_label'].isna()]
        df_clin = df_clin[['ID', 'response_label']]
        df = df.merge(df_clin, on='ID', how='left')
        df = df[~df['response_label'].isna()]
    elif name == 'tme_clin':
        df_clin = df_clin[~df_clin['response_label'].isna()]
        df_clin = df_clin[['ID', 'response_label', 'histological_type_at_dx', 'abs_lymph_IO_start',
                           'abs_neut_IO_start','abs_mono_IO_start', 'IO_start_neut-mono_ratio', 'IO_start_neut-lymph_ratio', 'PDL1_scores']]

        df = df.merge(df_clin, on='ID', how='left')
        df = df[~df['response_label'].isna()]

    elif name == 'clin_only':
        df_clin = df_clin[~df_clin['response_label'].isna()]
        df = df_clin[['ID', 'response_label', 'histological_type_at_dx', 'abs_lymph_IO_start',
                      'abs_neut_IO_start','abs_mono_IO_start', 'IO_start_neut-mono_ratio', 'IO_start_neut-lymph_ratio', 'PDL1_scores']]
    elif name == 'io_no_response':
        df_clin = df_clin[df_clin['response_label'].isna()]
        df_clin = df_clin[df_clin['io_cohort'] == 'received io']
        df_clin = df_clin[['ID', 'io_cohort', 'response_label']]
        df = df.merge(df_clin, on='ID', how='left')
        df = df[~df['io_cohort'].isna()]
    
    if 'PDL1_scores' in df.columns:
        df['PDL1_scores'] = df['PDL1_scores'].apply(lambda x: 0 if x == '<50%' else 1)

        df = df.rename(columns={'histological_type_at_dx':'subtype'})
        df['subtype'] = df['subtype'].where(df['subtype'].isin(['NSCLC_adenoca', 'NSCLC_sqcc']), 'other')
        df = pd.get_dummies(df, columns=['subtype'], dtype=int)

        df["abs_lymph_IO_start"].fillna(df["abs_lymph_IO_start"].mean(), inplace = True)
        df["abs_neut_IO_start"].fillna(df["abs_neut_IO_start"].mean(), inplace = True)
        df["abs_mono_IO_start"].fillna(df["abs_mono_IO_start"].mean(), inplace = True)
        df["IO_start_neut-lymph_ratio"].fillna(df["IO_start_neut-lymph_ratio"].mean(), inplace = True)
        df["IO_start_neut-mono_ratio"].fillna(df["IO_start_neut-mono_ratio"].mean(), inplace = True)
        
    return df

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from copy import deepcopy
join = os.path.join

def get_datadict(df_dict, random_state_i):
    def split_tr_val_ts(X_df):
        X_df_D = X_df.loc[X_df['Death'] == 1].drop('DeathDate', axis=1)
        X_df_noD = X_df.loc[X_df['Death'] == 0].drop('DeathDate', axis=1)
        Xtr_D, Xval_D, Xts_D = np.split(X_df_D.sample(frac=1, random_state=random_state_i), [int(.7 * len(X_df_D)), int(.8 * len(X_df_D))])
        Xtr_noD, Xval_noD, Xts_noD = np.split(X_df_noD.sample(frac=1, random_state=random_state_i), [int(.7 * len(X_df_noD)), int(.8 * len(X_df_noD))])
        Xtr_df = pd.merge(Xtr_D, Xtr_noD, how='outer').sample(frac=1, random_state=random_state_i)
        Xts_df = pd.merge(Xts_D, Xts_noD, how='outer').sample(frac=1, random_state=random_state_i)
        Xval_df = pd.merge(Xval_D, Xval_noD, how='outer').sample(frac=1, random_state=random_state_i)
        return Xtr_df, Xts_df, Xval_df
    RIH_df = df_dict['RIH']
    RIHtr_df, RIHts_df, RIHval_df = split_tr_val_ts(RIH_df)
    TMH_df = df_dict['TMH']
    TMHtr_df, TMHts_df, TMHval_df = split_tr_val_ts(TMH_df)
    NPH_df = df_dict['NPH']
    NPHtr_df, NPHts_df, NPHval_df = split_tr_val_ts(NPH_df)
    
    tr_df = pd.merge(pd.merge(RIHtr_df, TMHtr_df, how='outer'), NPHtr_df, how='outer')
    ts_df = pd.merge(pd.merge(RIHts_df, TMHts_df, how='outer'), NPHts_df, how='outer')
    val_df = pd.merge(pd.merge(RIHval_df, TMHval_df, how='outer'), NPHval_df, how='outer')
    white_keys = ['White or Caucasian', 'White or Caucasian, Other', 'White or Caucasian, Unknown']
    color_keys = ['Black or African American', 'Asian', 'Other']
    latino_keys = ['Hispanic or Latino']
    nolatino_keys = ['Not Hispanic or Latino']
    def get_race_idx(x_df):
        white_idx = (x_df['Race'] == white_keys[0]) + (x_df['Race'] == white_keys[1]) + (x_df['Race'] == white_keys[2])
        color_idx = (x_df['Race'] == color_keys[0]) + (x_df['Race'] == color_keys[1]) + (x_df['Race'] == color_keys[2])
        return white_idx, color_idx
    def get_ethnicity_idx(x_df):
        latino_idx = (x_df['Ethnicity'] == latino_keys[0])
        nolatino_idx = (x_df['Ethnicity'] == nolatino_keys[0])
        return nolatino_idx, latino_idx
    def get_sex_idx(x_df):
        men_idx = (x_df['PatientSex'] == 1)
        women_idx = (x_df['PatientSex'] == 0)
        return men_idx, women_idx
    # sample ts
    white_idx, color_idx = get_race_idx(ts_df)
    ts_race_df = deepcopy(ts_df[white_idx + color_idx])
    ts_white_df = deepcopy(ts_df[white_idx])
    ts_color_df = deepcopy(ts_df[color_idx])
    nolatino_idx, latino_idx = get_ethnicity_idx(ts_df)
    ts_ethnicity_df = deepcopy(ts_df[nolatino_idx + latino_idx])
    ts_latino_df = deepcopy(ts_df[latino_idx])
    ts_nolatino_df = deepcopy(ts_df[nolatino_idx])
    men_idx, women_idx = get_sex_idx(ts_df)
    ts_sex_df = deepcopy(ts_df[men_idx + women_idx])
    ts_men_df = deepcopy(ts_df[men_idx])
    ts_women_df = deepcopy(ts_df[women_idx])
    
    # resample tr
    white_idx, color_idx = get_race_idx(tr_df)
    tr_race_nore_df = deepcopy(tr_df[white_idx + color_idx])
    tr_white_df = deepcopy(tr_df[white_idx])
    tr_color_df = deepcopy(tr_df[color_idx].sample(n=len(tr_white_df), replace=True, random_state=random_state_i))
    tr_race_re_df = deepcopy(pd.merge(tr_white_df, tr_color_df, how='outer'))
    white_idx, color_idx = get_race_idx(val_df)
    val_race_nore_df = deepcopy(val_df[white_idx + color_idx])
    val_white_df = deepcopy(val_df[white_idx])
    val_color_df = deepcopy(val_df[color_idx].sample(n=len(val_white_df), replace=True, random_state=random_state_i))
    val_race_re_df = deepcopy(pd.merge(val_white_df, val_color_df, how='outer'))
    
    nolatino_idx, latino_idx = get_ethnicity_idx(tr_df)
    tr_ethnicity_nore_df = deepcopy(tr_df[nolatino_idx + latino_idx])
    tr_nolatino_df = deepcopy(tr_df[nolatino_idx])
    tr_latino_df = deepcopy(tr_df[latino_idx].sample(n=len(tr_nolatino_df), replace=True, random_state=random_state_i))
    tr_ethnicity_re_df = deepcopy(pd.merge(tr_nolatino_df, tr_latino_df, how='outer'))
    nolatino_idx, latino_idx = get_ethnicity_idx(val_df)
    val_ethnicity_nore_df = deepcopy(val_df[nolatino_idx + latino_idx])
    val_nolatino_df = deepcopy(val_df[nolatino_idx])
    val_latino_df = deepcopy(val_df[latino_idx].sample(n=len(val_nolatino_df), replace=True, random_state=random_state_i))
    val_ethnicity_re_df = deepcopy(pd.merge(val_nolatino_df, val_latino_df, how='outer'))
    
    men_idx, women_idx = get_sex_idx(tr_df)   
    tr_sex_nore_df = deepcopy(tr_df[men_idx + women_idx])
    tr_men_df = deepcopy(tr_df[men_idx])
    tr_women_df = deepcopy(tr_df[women_idx])
    if np.sum(men_idx) > np.sum(women_idx):
        tr_women_df = deepcopy(tr_women_df.sample(n=np.sum(men_idx), replace=True, random_state=random_state_i))
    elif np.sum(men_idx) < np.sum(women_idx):
        tr_men_df = deepcopy(tr_men_df.sample(n=np.sum(women_idx), replace=True, random_state=random_state_i))
    tr_sex_re_df = deepcopy(pd.merge(tr_men_df, tr_women_df, how='outer'))
    
    men_idx, women_idx = get_sex_idx(val_df)
    val_sex_nore_df = deepcopy(val_df[men_idx + women_idx])
    val_men_df = deepcopy(val_df[men_idx])
    val_women_df = deepcopy(val_df[women_idx])
    if np.sum(men_idx) > np.sum(women_idx):
        val_women_df = deepcopy(val_women_df.sample(n=np.sum(men_idx), replace=True, random_state=random_state_i))
    elif np.sum(men_idx) < np.sum(women_idx):
        val_men_df = deepcopy(val_men_df.sample(n=np.sum(women_idx), replace=True, random_state=random_state_i))
    val_sex_re_df = deepcopy(pd.merge(val_men_df, val_women_df, how='outer'))
    
    df_dict = {
        "tr": tr_df,
        "val": val_df,
        "ts": ts_df,
        'tr_race_nore': tr_race_nore_df,
        'val_race_nore': val_race_nore_df,
        'tr_race_re': tr_race_re_df,
        'val_race_re': val_race_re_df,
        'ts_race': ts_race_df, 
        'ts_race_White': ts_white_df,
        'ts_race_Color': ts_color_df,
        
        'tr_ethnicity_nore': tr_ethnicity_nore_df,
        'val_ethnicity_nore': val_ethnicity_nore_df,
        'tr_ethnicity_re': tr_ethnicity_re_df,
        'val_ethnicity_re': val_ethnicity_re_df,
        'ts_ethnicity': ts_ethnicity_df,
        'ts_ethnicity_nolatino': ts_nolatino_df,
        'ts_ethnicity_latino': ts_latino_df,
        
        'tr_sex_nore': tr_sex_nore_df,
        'val_sex_nore': val_sex_nore_df,
        'tr_sex_re': tr_sex_re_df,
        'val_sex_re': val_sex_re_df,
        'ts_sex': ts_sex_df,
        'ts_sex_men': ts_men_df,
        'ts_sex_women': ts_women_df,
    }

    # Feature transforms
    cols_standardize = ['Age']
    cols_leave = ['PatientSex', 'CA', 'CHF', 'COPD', 'HR_gte110', 'SBP_lt100', 'RR', 'Temp', 'AMS', 'SpO2_lt90', ]
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)
    # get feature table
    img_feat_dict = {}
    text_feat_dict = {}
    event_dict = {}
    time_dict = {}
    race_dict = {}
    ethnicity_dict = {}
    sex_dict = {}
    AHA_dict = {}
    PESI_dict = {}
    PESI_caseid_dict = {}
    PESI_prams_dict = {}
    img_feat_root = './PE_data/feat_avgpool_out'
    text_feat_root = './PE_data/Report_Text_Features'
    for trts in list(df_dict.keys()):
        img_feat_dict[trts] = []
        text_feat_dict[trts] = []
        event_dict[trts] = []
        time_dict[trts] = []
        race_dict[trts] = []
        sex_dict[trts] = []
        ethnicity_dict[trts] = []
        AHA_dict[trts] = []
        PESI_caseid_dict[trts] = []
        PESI_dict[trts] = []
        PESI_prams_dict[trts] = []
        prams_data = x_mapper.fit_transform(df_dict[trts]).astype('float32')
        for i_0, i in enumerate(df_dict[trts].index):
            d = df_dict[trts].loc[i]
            c_name = d['AccessionNumber_md5']
            img_modal = d['img_modal']
            img_name = c_name + '_' + img_modal + '_.npy'
            img_npy = np.load(join(img_feat_root, img_name)).squeeze()
            img_feat_dict[trts].append(deepcopy(img_npy))

            text_name = c_name + '.npy'
            text_npy = np.load(join(text_feat_root, text_name)).squeeze().reshape(-1)
            text_feat_dict[trts].append(deepcopy(text_npy))
            event_dict[trts].append(d['Death'])
            time_dict[trts].append(d['follow_up_day'])
            AHA_dict[trts].append(d['AHA_PE_severity'])
            PESI_caseid_dict[trts].append(i_0)
            PESI_dict[trts].append(d['PESI'])
            PESI_prams_dict[trts].append(prams_data[i_0])
            
            if d['Race'] in white_keys:
                race_f = 0
            elif d['Race'] in color_keys:
                race_f = 1
            else:
                race_f = -1
            race_dict[trts].append(race_f)            
            if d['Ethnicity'] in latino_keys:
                ethnicity_f = 1
            elif d['Ethnicity'] in nolatino_keys:
                ethnicity_f = 0
            else:
                ethnicity_f = -1
            ethnicity_dict[trts].append(ethnicity_f)
            if d['PatientSex'] == 0:
                sex_f = 0 # female
            elif d['PatientSex'] == 1:
                sex_f = 1 # male
            else:
                sex_f = -1
            sex_dict[trts].append(sex_f)
        img_feat_dict[trts] = deepcopy(np.array(img_feat_dict[trts]).squeeze())
        # img_feat_dict[trts] = (img_feat_dict[trts] - img_feat_dict[trts].mean()) / img_feat_dict[trts].std()
        text_feat_dict[trts] = deepcopy(np.array(text_feat_dict[trts]).squeeze())
        # text_feat_dict[trts] = (text_feat_dict[trts] - text_feat_dict[trts].mean()) / text_feat_dict[trts].std()
        event_dict[trts] = deepcopy(np.array(event_dict[trts]).squeeze())
        time_dict[trts] = deepcopy(np.array(time_dict[trts]).squeeze())
        race_dict[trts] = deepcopy(np.array(race_dict[trts]).squeeze())
        ethnicity_dict[trts] = deepcopy(np.array(ethnicity_dict[trts]).squeeze())
        sex_dict[trts] = deepcopy(np.array(sex_dict[trts]).squeeze())
        AHA_dict[trts] = deepcopy(np.array(AHA_dict[trts]).squeeze())
        PESI_dict[trts] = deepcopy(np.array(PESI_dict[trts]).squeeze())
        PESI_prams_dict[trts] = np.array(PESI_prams_dict[trts]).astype(np.float32)

    labels_dict = {}
    # convert the train labels for CPH model
    for trts in list(df_dict.keys()):
        labels_dict[trts] = np.ndarray(shape=(event_dict[trts].shape[0],),
                                       dtype=[('status', '?'), ('survival_in_days', '<f8')])
        for ind_train in range(event_dict[trts].shape[0]):
            if event_dict[trts][ind_train] == 1:
                labels_dict[trts][ind_train] = (True, time_dict[trts][ind_train])
            else:
                labels_dict[trts][ind_train] = (False, time_dict[trts][ind_train])
    return df_dict, img_feat_dict, text_feat_dict, \
           event_dict, time_dict, labels_dict, race_dict, ethnicity_dict, sex_dict,\
           AHA_dict, PESI_dict, PESI_caseid_dict, PESI_prams_dict,
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
# from copy import deepcopy
from file_and_folder_operations import *
import torchtuples as tt
from loss_surv import *
from prepare_dict import get_datadict
from debias_model import debias_leaner
from tt_Model_save_feat import ttModel

def process(group = 'race'):
    random_state_i = 777
    global df_dict0
    df_dict = deepcopy(df_dict0)
    print(random_state_i, group)
    df_dict, img_feat_dict, text_feat_dict, \
    event_dict, time_dict, labels_dict, race_dict, ethnicity_dict, sex_dict, \
    AHA_dict, PESI_dict, PESI_caseid_dict, PESI_prams_dict, = get_datadict(df_dict, random_state_i)
    if group == 'race':
        group_dict = race_dict
        key_2 = ['ts_race_White', 'ts_race_Color', ]
    elif group == 'ethnicity':
        group_dict = ethnicity_dict
        key_2 = ['ts_ethnicity_nolatino', 'ts_ethnicity_latino', ]
    elif group == 'sex':
        group_dict = sex_dict
        key_2 = ['ts_sex_men', 'ts_sex_women']

    # save feats
    save_feat_root = f'./exps/{group}/result_{str(random_state_i)}/'
    save_dict_root = join(save_feat_root, 'KM_dict')
    maybe_mkdir_p(save_dict_root)
    for bias_key in ['id', 'surv']:
        for trts in list(df_dict.keys()):
            if trts in ['tr', 'val', 'ts', 'tr_' + group + '_nore', 'val_' + group + '_nore',
                        'tr_' + group + '_re', 'val_' + group + '_re', 'ts_' + group, ] + key_2:
                save_feat_dir = join(save_feat_root, 'img_feat', 'saved_feats_'+bias_key)
                maybe_mkdir_p(save_feat_dir)
                feat_np = img_feat_dict[trts]
                np.save(os.path.join(save_feat_dir, trts+'.npy'), feat_np)
                save_feat_dir = join(save_feat_root, 'text_feat', 'saved_feats_'+bias_key)
                maybe_mkdir_p(save_feat_dir)
                feat_np = text_feat_dict[trts]
                np.save(os.path.join(save_feat_dir, trts+'.npy'), feat_np)
                save_feat_dir = join(save_feat_root, 'clin_feat', 'saved_feats_'+bias_key)
                maybe_mkdir_p(save_feat_dir)
                feat_np = PESI_prams_dict[trts]
                np.save(os.path.join(save_feat_dir, trts+'.npy'), feat_np)
                save_feat_dir = join(save_feat_root, group+'_id', 'saved_feats_'+bias_key)
                maybe_mkdir_p(save_feat_dir)
                feat_np = group_dict[trts]
                np.save(os.path.join(save_feat_dir, trts+'.npy'), feat_np)

    # ---------deep
    def train_model(feat_dict, train_set='tr', val_set='val',  num_nodes=[32, 32], run_name='', device='cuda:0'):
        torch.manual_seed(0)
        y_dict = {}
        y_dict[train_set] = (time_dict[train_set], event_dict[train_set])
        y_dict[val_set] = (time_dict[val_set], event_dict[val_set])
        val = (feat_dict[val_set], y_dict[val_set])
        criterion = CoxPHLoss()
        net = tt.practical.MLPVanilla(in_features=feat_dict[train_set].shape[1], num_nodes=num_nodes, out_features=1, batch_norm=True,
                                      dropout=0.1, output_bias=False, output_activation=torch.nn.Sigmoid())
        optimizer = tt.optim.AdamW(lr=0.001, decoupled_weight_decay=0.0005)
        # model = tt.Model(net, criterion, optimizer, device=device,)
        model = ttModel(net, criterion, optimizer, device=device,) # rewrite to save features of baseline model
        model.init_save_feat(exp_dir=f'./exps/{group}/result_{str(random_state_i)}/{run_name}')
        batch_size = feat_dict[train_set].shape[0]
        # model.optimizer.set(0.01)
        epochs = 10000
        callbacks = [tt.cb.EarlyStopping(patience=600)]
        log = model.fit(feat_dict[train_set], y_dict[train_set], batch_size, epochs,
                        callbacks, verbose=False, val_data=val)
        return model


    def train_debias_model(img_feat_dict, id_dict, train_set='tr', val_set='val', num_nodes=[32, 32], swap_augment=False, run_name='img_feat', device='cuda:0'):
        torch.manual_seed(0)
        group = run_name.split('_')[1]
        val_with_id = id_dict[val_set]>-1
        val_data = (img_feat_dict[val_set][val_with_id],
                    id_dict[val_set][val_with_id], (time_dict[val_set][val_with_id], event_dict[val_set][val_with_id]))
        train_with_id = id_dict[train_set]>-1
        tarin_data = (img_feat_dict[train_set][train_with_id],
                    id_dict[train_set][train_with_id], (time_dict[train_set][train_with_id], event_dict[train_set][train_with_id]))
        
        in_features_a = img_feat_dict[train_set][train_with_id].shape[1]
        leaner = debias_leaner(in_features_a, num_nodes=num_nodes, swap_augment=swap_augment, exp_dir=f'./exps/{group}/result_{str(random_state_i)}/{run_name}', run_name=run_name, device=device, group=group)
        leaner.train(tarin_data, val_data)
        return leaner

    device = 'cuda:0'
    node_num_img = [1024, 512, 512]
    node_num_text = [1024, 512, 512]
    node_num_clin = [512, 512]
    # baseline deep
    model_img = train_model(img_feat_dict, num_nodes=node_num_img, run_name='img_base', device=device)
    model_text = train_model(text_feat_dict, num_nodes=node_num_text, run_name='text_base', device=device)
    model_clin = train_model(PESI_prams_dict, num_nodes=node_num_clin, run_name='clin_base', device=device)
    deep_cph_base = CoxPHSurvivalAnalysis()
    deep_fuse_PESI_cph_base = CoxPHSurvivalAnalysis()

    # swap, resample
    model_img_sw_re = train_debias_model(img_feat_dict, group_dict, train_set='tr_' + group + '_re', val_set='val_' + group + '_re', num_nodes=node_num_img, swap_augment=True,
                                         run_name='img_' + group + '_sw_re', )
    model_text_sw_re = train_debias_model(text_feat_dict, group_dict, train_set='tr_' + group + '_re', val_set='val_' + group + '_re', num_nodes=node_num_text, swap_augment=True,
                                          run_name='text_' + group + '_sw_re', )
    model_clin_sw_re = train_debias_model(PESI_prams_dict, group_dict, train_set='tr_' + group + '_re', val_set='val_' + group + '_re', num_nodes=node_num_clin, swap_augment=True,
                                          run_name='clin_' + group + '_sw_re', )
    deep_cph_sw_re = CoxPHSurvivalAnalysis()
    deep_fuse_PESI_cph_sw_re = CoxPHSurvivalAnalysis()

    # no_swap, no_resample
    model_img_nosw_nore = train_debias_model(img_feat_dict, group_dict, train_set='tr_'+group+'_nore', val_set='val_'+group+'_nore', num_nodes=node_num_img, swap_augment=False, run_name='img_'+group+'_nosw_nore',)
    model_text_nosw_nore = train_debias_model(text_feat_dict, group_dict, train_set='tr_'+group+'_nore', val_set='val_'+group+'_nore', num_nodes=node_num_text, swap_augment=False, run_name='text_'+group+'_nosw_nore',)
    model_clin_nosw_nore = train_debias_model(PESI_prams_dict, group_dict, train_set='tr_'+group+'_nore', val_set='val_'+group+'_nore', num_nodes=node_num_clin, swap_augment=False, run_name='clin_'+group+'_nosw_nore',)
    deep_cph_nosw_nore = CoxPHSurvivalAnalysis()
    deep_fuse_PESI_cph_nosw_nore = CoxPHSurvivalAnalysis()

    # swap, no_resample
    model_img_sw_nore = train_debias_model(img_feat_dict, group_dict, train_set='tr_'+group+'_nore', val_set='val_'+group+'_nore', num_nodes=node_num_img, swap_augment=True, run_name='img_'+group+'_sw_nore',)
    model_text_sw_nore = train_debias_model(text_feat_dict, group_dict, train_set='tr_'+group+'_nore', val_set='val_'+group+'_nore', num_nodes=node_num_text, swap_augment=True, run_name='text_'+group+'_sw_nore',)
    model_clin_sw_nore = train_debias_model(PESI_prams_dict, group_dict, train_set='tr_'+group+'_nore', val_set='val_'+group+'_nore', num_nodes=node_num_clin, swap_augment=True, run_name='clin_'+group+'_sw_nore',)
    deep_cph_sw_nore = CoxPHSurvivalAnalysis()
    deep_fuse_PESI_cph_sw_nore = CoxPHSurvivalAnalysis()

    # no_swap, resample
    model_img_nosw_re = train_debias_model(img_feat_dict, group_dict,  train_set='tr_'+group+'_re', val_set='val_'+group+'_re', num_nodes=node_num_img, swap_augment=False, run_name='img_'+group+'_nosw_re',)
    model_text_nosw_re = train_debias_model(text_feat_dict, group_dict, train_set='tr_'+group+'_re', val_set='val_'+group+'_re', num_nodes=node_num_text, swap_augment=False, run_name='text_'+group+'_nosw_re',)
    model_clin_nosw_re = train_debias_model(PESI_prams_dict, group_dict, train_set='tr_'+group+'_re', val_set='val_'+group+'_re', num_nodes=node_num_clin, swap_augment=False, run_name='clin_'+group+'_nosw_re',)
    deep_cph_nosw_re = CoxPHSurvivalAnalysis()
    deep_fuse_PESI_cph_nosw_re = CoxPHSurvivalAnalysis()

    c_d = {}
    # test the performance
    c_d['PESI_c_ind_dict'] = {}
    c_d['rsf_img_dict'] = {}
    c_d['rsf_text_dict'] = {}
    c_d['rsf_clin_dict'] = {}
    c_d['rsf_fuse_dict'] = {}
    c_d['rsf_fuse_PESI_dict'] = {}
    
    c_d['deep_img_dict'] = {}
    c_d['deep_text_dict'] = {}
    c_d['deep_clin_dict'] = {}
    c_d['deep_fuse_dict'] = {}
    c_d['deep_fuse_PESI_dict'] = {}
    
    for exp_name in [group+'_nosw_nore', group+'_sw_nore', group+'_nosw_re', group+'_sw_re',]:
        c_d['deep_img_'+exp_name+'_dict'] = {}
        c_d['deep_text_'+exp_name+'_dict'] = {}
        c_d['deep_clin_'+exp_name+'_dict'] = {}
        c_d['deep_fuse_'+exp_name+'_dict'] = {}
        c_d['deep_fuse_PESI_'+exp_name+'_dict'] = {}

    def evaluate_model(model_img, model_text, model_clin, deep_cph, deep_fuse_PESI_cph, trts, exp_name_a,
                        deep_img_dict, deep_text_dict, deep_clin_dict, deep_fuse_dict, deep_fuse_PESI_dict):
        risk_img = model_img.predict(img_feat_dict[trts], save_feat_name=trts)
        risk_text = model_text.predict(text_feat_dict[trts], save_feat_name=trts)
        risk_clin = model_clin.predict(PESI_prams_dict[trts], save_feat_name=trts)
        deep_img_dict[trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], risk_img[:,0])[0]
        deep_text_dict[trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], risk_text[:,0])[0]
        deep_clin_dict[trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], risk_clin[:,0])[0]
        risk_imgclin = np.concatenate((risk_img, risk_text, risk_clin), axis=1)
        if 'tr' in trts:
            deep_cph = deep_cph.fit(risk_imgclin, labels_dict[trts])
        deep_fuse_dict[trts] = deep_cph.score(risk_imgclin, labels_dict[trts])
        risk_deep_fused = deep_cph.predict(risk_imgclin, )
        
        risk_concatPESI = np.concatenate([risk_deep_fused[:,None], PESI_dict[trts][:,None]], axis=1)
        if 'tr' in trts:
            deep_fuse_PESI_cph = deep_fuse_PESI_cph.fit(risk_concatPESI, labels_dict[trts])
        deep_fuse_PESI_dict[trts] = deep_fuse_PESI_cph.score(risk_concatPESI, labels_dict[trts])
        risk_deep_fused_PESI = deep_fuse_PESI_cph.predict(risk_concatPESI, )
        
        deep_fused_curves = deep_cph.predict_survival_function(risk_imgclin)
        deep_fused_PESI_curves = deep_fuse_PESI_cph.predict_survival_function(risk_concatPESI)

        KM_dict = {'risk_img':risk_img, 'risk_text':risk_text, 'risk_clin':risk_clin, 
                   'risk_deep_fused': risk_deep_fused, 'risk_deep_fused_PESI':risk_deep_fused_PESI,
                   'deep_fused_curves': deep_fused_curves, 'deep_fused_PESI_curves':deep_fused_PESI_curves,
                   'id_list':group_dict[trts],
                   'PESI_scores':PESI_dict[trts], 'PESI_caseid':PESI_caseid_dict[trts], 'PESI_variables':PESI_prams_dict[trts],
                   'time': time_dict[trts], 'event':event_dict[trts],}
        KM_dict_npy_path = join(save_dict_root, exp_name_a+'@'+trts+'.npy')
        np.save(KM_dict_npy_path, KM_dict)
        return deep_img_dict, deep_text_dict, deep_clin_dict, deep_fuse_dict, deep_fuse_PESI_dict
                
    def evaluate_idmodel(exp_name='race_nosw_nore'):  
        model_name = exp_name.split('_')[1:]
        model_name = model_name[0] + '_' + model_name[1]
        phase = "c_d['deep_img_"+exp_name+"_dict'], c_d['deep_text_"+exp_name+"_dict'], c_d['deep_clin_"+exp_name+"_dict'], c_d['deep_fuse_"+exp_name+"_dict'], c_d['deep_fuse_PESI_"+exp_name+"_dict']\
            = evaluate_model(model_img_"+model_name+", model_text_"+model_name+", model_clin_"+model_name+", deep_cph_"+model_name+", deep_fuse_PESI_cph_"+model_name+", trts, '"+exp_name+"', \
                c_d['deep_img_"+exp_name+"_dict'], c_d['deep_text_"+exp_name+"_dict'], c_d['deep_clin_"+exp_name+"_dict'], c_d['deep_fuse_"+exp_name+"_dict'], c_d['deep_fuse_PESI_"+exp_name+"_dict'])"
        return phase
    # evaluate
    for trts in list(df_dict.keys()):
        if trts in ['tr', 'val', 'ts', 'ts_'+group, ] + key_2:
            # ----------PESI c index
            c_d['PESI_c_ind_dict'][trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], PESI_dict[trts])[0]
            c_d['deep_img_dict'], c_d['deep_text_dict'], c_d['deep_clin_dict'], c_d['deep_fuse_dict'], c_d['deep_fuse_PESI_dict'] = \
                    evaluate_model(model_img, model_text, model_clin, deep_cph_base, deep_fuse_PESI_cph_base, trts, 'base',
                                c_d['deep_img_dict'], c_d['deep_text_dict'], c_d['deep_clin_dict'], c_d['deep_fuse_dict'], c_d['deep_fuse_PESI_dict'])

            print(group+'_nosw_nore', group+'_sw_nore', trts)
            exec(evaluate_idmodel(exp_name=group+'_nosw_nore' ))
            exec(evaluate_idmodel(exp_name=group+'_sw_nore' ))
            print(group+'_nosw_re', group+'_sw_re', trts)
            exec(evaluate_idmodel(exp_name=group+'_nosw_re' ))
            exec(evaluate_idmodel(exp_name=group+'_sw_re' ))

    return c_d

if __name__ == '__main__':
    label_path = './PE_data/pe_label_search.xls'

    labels_df_s = pd.read_excel(label_path)
    labels_df_s['Age'] = (labels_df_s['Age'] / labels_df_s['Age'].max())
    labels_df_s['follow_up_day'] = labels_df_s['follow_up_day'] / labels_df_s['follow_up_day'].max()
    RIH_df = labels_df_s.loc[labels_df_s['Var22'] == 'RIH']
    TMH_df = labels_df_s.loc[labels_df_s['Var22'] == 'TMH']
    NPH_df = labels_df_s.loc[labels_df_s['Var22'] == 'NPH']
    df_dict0 = {'RIH': RIH_df,
                "TMH": TMH_df,
                "NPH": NPH_df,
                }

    out_xlsx_folder = './exps/out_xlsx'
    maybe_mkdir_p(out_xlsx_folder)

    for group in ['race', 'ethnicity', 'sex']:
        out_c_ind_dict = process( group)
        out_df = pd.DataFrame.from_dict(out_c_ind_dict).transpose()
        out_filename = 'unbias_'+group+'.xlsx'
        out_xlsx_path = join(out_xlsx_folder, out_filename)
        out_df.to_excel(out_xlsx_path)





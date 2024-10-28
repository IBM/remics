import pandas as pd 
import numpy as np 
import os, sys, time

import matplotlib.pyplot as plt 
import seaborn as sns
import argparse 

from sklearn.feature_selection import RFE
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sys.path.append('/dccstor/boseukb/CuNA/Geno4SD')
from geno4sd.topology.CuNA import cumulants, cuna, cures


def read_data(data_path, dx): 
    
    methy = pd.read_csv(data_path+'methy', sep=" ")
    print("Methylation: ", methy.shape)

    exp = pd.read_csv(data_path+'exp', sep=" ")
    print("Expression: ", exp.shape)

    mirna = pd.read_csv(data_path + 'mirna', sep=" ")
    print("miRNA: ", mirna.shape)

    clinical = pd.read_csv(data_path + 'survival', sep="\t")
    print("clinical: ", clinical.shape)

    clinical_all = pd.read_csv(data_path+dx, sep="\t")
    print("all clinical: ", clinical_all.shape)
    clinical.dropna(inplace=True)
    covars = clinical_all[['sampleID', 'gender', 'age_at_initial_pathologic_diagnosis']]
    covars['gender'] = covars['gender'].map({'MALE': 0, 'FEMALE': 1})
    
    return methy, exp, mirna, clinical, covars

def replace_delim(cols):
    return [x.replace('.','-') for x in cols]

def find_overlapping(methy, exp, mirna, clinical):
    methy.columns = replace_delim(methy.columns)
    exp.columns = replace_delim(exp.columns)
    mirna.columns = replace_delim(mirna.columns)
    clinical_samples = set(clinical['PatientID'])

    overlapping_cols = set(methy.columns).intersection(set(exp.columns))
    overlapping_cols = set(mirna.columns).intersection(overlapping_cols)
    overlapping_cols = list(clinical_samples.intersection(overlapping_cols))
    print("Number of overlapping IDs: ", len(overlapping_cols))
    
    return overlapping_cols

def split_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y, 
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=123
                                                    )
    
    return x_train, x_test, y_train, y_test

def get_best_model(X,y): 
    estimator = LR()
    parameters = [{'penalty':['l1','l2']},
                {'C': [0.1, 1, 10, 100]}]
    
    grid_search = GridSearchCV(
                            estimator=estimator, 
                            param_grid=parameters, 
                            scoring='f1_weighted', 
                            cv=5, 
                            verbose=0
                            )
    grid_search.fit(X,y)
    best_model = grid_search.best_estimator_
    
    return best_model 
    
def get_highvar(df, q=5, threshold=3): 
    var_exp = pd.DataFrame(df.var())
    var_exp.columns = ['var']
    var_exp['decile'] = pd.qcut(var_exp['var'], q, labels=False)
    low_var_features = var_exp[var_exp['decile'] < threshold].index 
    df_highvar = df.drop(columns=low_var_features)
    
    return df_highvar

def get_f1(model, x_test, y_test):
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1
    
def select_features(x_train, y_train, num_features):
    # rfe = RFE(model, n_features_to_select=num_features)
    # rfe.fit(x_train, y_train)

    omp = OMP(n_nonzero_coefs=num_features)
    omp.fit(x_train, y_train)
    selected_features = np.where(omp.coef_ != 0)[0]
    
    return selected_features

def scale_data(x_train, x_test):
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test) 
    
    return x_train_scaled, x_test_scaled   

def check_sample(df):
    if 'sampleID' in list(df.columns): 
        df.set_index('sampleID', inplace=True)
    return df 

def get_features(df, y, num_features):
    beg_time = time.time()
    features = df.columns
    X = df.values
    x_train, x_test, y_train, y_test = split_data(X,y)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    
    mdl = get_best_model(x_train_scaled,y_train)
    mdl.fit(x_train_scaled, y_train)
    pre_f1 = get_f1(mdl, x_test_scaled, y_test)
    print("Before Feature Selection, F1 score: ", pre_f1)

    feature_mask = select_features(
                                x_train_scaled, 
                                y_train, 
                                num_features
                                )
    selected_features = features[feature_mask]
    
    X_sel = df[selected_features].values
    x_train, x_test, y_train, y_test = split_data(X_sel,y)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    mdl_sel = get_best_model(x_train_scaled, y_train)
    mdl_sel.fit(x_train_scaled, y_train)
    post_f1 = get_f1(mdl_sel, x_test_scaled, y_test)
    print("Before Feature Selection, F1 score: ", post_f1)
    
    covar_feats = ['gender', 'age_at_initial_pathologic_diagnosis']
    selected_features = [x for x in selected_features if x not in covar_feats]
    
    print("Time taken to train and select features (mins): ", (time.time()-beg_time)/60)
    
    return selected_features

def merge_covar(df, covars): 
    merged_df = pd.merge(
                    df, 
                    covars, 
                    left_index=True, 
                    right_on='sampleID',
                    how='inner'
                    )
    merged_df.drop(['sampleID'], axis=1, inplace=True)
    return merged_df

def merge_omics(df1, df2, df3): 
    df12 = pd.merge(
                df1, 
                df2, 
                left_index=True, 
                right_index=True, 
                how='inner'
                )
    df123 = pd.merge(
                df12, 
                df3, 
                left_index=True, 
                right_index=True, 
                how='inner'
                )
    
    return df123


def main():
    parser = argparse.ArgumentParser(description="Reads a file path.", 
                                    exit_on_error=True)
    parser.add_argument(
                    "-f", 
                    "--file_path",
                    action='store', 
                    help="The path to the file."
                    )
    parser.add_argument(
                    "-e", 
                    "--num_meth", 
                    action='store',
                    default=200,
                    help="Number of methylation features"  
                    )
    parser.add_argument(
                    "-g", 
                    "--num_genes", 
                    action='store',
                    default=400, 
                    help="Number of genes"  
                    )
    parser.add_argument(
                    "-m", 
                    "--num_mirna", 
                    action='store',
                    default=50, 
                    help="Number of microRNAs"  
                    )
    parser.add_argument(
                    "-p", 
                    "--pval", 
                    action='store', 
                    default=1e-5, 
                    help="p-value threshold to filter cumulants"  
                    )
    parser.add_argument(
                    "-c", 
                    "--cancer", 
                    action='store',
                    choices=['aml', 'breast', 'liver', 'lung', 'kidney', 'sarcoma', 'ovarian', 'gbm', 'colon', 'melanoma'],
                    help="Name of cancer"  
                    )
    parser.add_argument(
                    "-i", 
                    "--integrate", 
                    action=argparse.BooleanOptionalAction, 
                    default=True, 
                    help="Flag indicating whether to integrate"  
                    )
    
    args = parser.parse_args()

    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
        
    if args.integrate: 
        print("******************")
        print("  Reading and Processing Data  ")
        print("******************")
        
        methy, exp, mirna, clinical, covars = read_data(args.file_path, args.cancer)
        overlapping_ids = find_overlapping(methy, exp, mirna, clinical)
        methy_ov = methy[overlapping_ids].T
        exp_ov = exp[overlapping_ids].T
        mirna_ov = mirna[overlapping_ids].T
        
        print(methy_ov.shape)
        print(exp_ov.shape)
        print(mirna_ov.shape)
        
        clinical_ov = pd.merge(
                            clinical, 
                            methy_ov, 
                            left_on='PatientID', 
                            right_index=True, 
                            how='inner'
                            )
        
        clinical_ov.drop_duplicates('PatientID', keep='last', inplace=True)
        outcome = list(clinical_ov['Death'])
        print(clinical_ov.groupby('Death').size())
        
        merged_methy_ov = merge_covar(methy_ov, covars)
        merged_exp_ov = merge_covar(exp_ov, covars)
        merged_mirna_ov = merge_covar(mirna_ov, covars)
    
    
    print("******************")
    print("  Model Fitting & Feature Selection  ")
    print("******************")
    
    if args.integrate is False: 
        merged_methy_ov = check_sample(pd.read_csv(args.file_path + 'merged_methy.csv'))
        merged_exp_ov = check_sample(pd.read_csv(args.file_path + 'merged_exp.csv'))
        merged_mirna_ov = check_sample(pd.read_csv(args.file_path + 'merged_mirna.csv'))
        merged_clinical_ov = check_sample(pd.read_csv(args.file_path + 'merged_clinical.csv'))
        outcome = list(merged_clinical_ov['Death'])
    
    print(len(outcome))
    print(merged_methy_ov.shape)
    print("Getting methylation features")    
    sel_methy = get_features(merged_methy_ov, outcome, num_features=int(args.num_meth))
    print("Getting expression features")    
    sel_exp = get_features(merged_exp_ov, outcome, num_features=int(args.num_genes))
    print("Getting microRNA features")    
    sel_mirna = get_features(merged_mirna_ov, outcome, num_features=int(args.num_mirna))
    
    sel_methy_df = merged_methy_ov[sel_methy]
    sel_exp_df = merged_exp_ov[sel_exp]
    sel_mirna_df = merged_mirna_ov[sel_mirna]
    
    methy_exp_mirna_df = merge_omics(sel_methy_df, sel_exp_df, sel_mirna_df)
    print("Dimension of early omics integration: ", methy_exp_mirna_df.shape)
    
    methy_exp_mirna_df.to_csv(args.file_path + 'omics_integrated.csv')
    
    print("******************")
    print("  Cumulant calculation  ")
    print("******************")
    
    
    ## get cumulants 
    beg_time = time.time()
    cumulants_df, vec_df = cumulants.get_cumulants(
                                                methy_exp_mirna_df, 
                                                verbose = 1, 
                                                julia = 1, 
                                                order = 3) 
    print("Time spent computing cumulants (mins): ", (time.time() - beg_time)/60)
    
    filt_cumulants_df = cumulants_df[cumulants_df['P'] < float(args.pval)]  
    filt_vec_df = vec_df[vec_df.k.isin(list(filt_cumulants_df.index))]
    print("Number of significant redescription groups: ", filt_vec_df.shape[0])
    
    filt_vec_df.to_csv(args.file_path + "filt_vec_df.csv", sep='\t')
    filt_cumulants_df.to_csv(args.file_path + "filt_cumulants_df.csv", sep='\t')
    
    
if __name__ == "__main__":
    main()
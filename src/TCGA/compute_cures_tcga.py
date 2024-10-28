import pandas as pd 
import numpy as np 
import os, sys, time 
import copy 
sys.path.append('/dccstor/boseukb/CuNA/Geno4SD')
from geno4sd.topology.CuNA import cumulants, cuna, cures

import matplotlib.pyplot as plt 
import seaborn as sns
import argparse

def read_data(data_path): 
    
    cumulants_df = pd.read_csv(data_path+"/filt_cumulants_df.csv", sep='\t')
    vec_df = pd.read_csv(data_path + '/filt_vec_df.csv', sep='\t')
    clinical_df = pd.read_csv(data_path+'/merged_clinical.csv')
    integrated_df = pd.read_csv(data_path+'/omics_integrated.csv')
    covars_df = covars_df = pd.read_csv(data_path+'/merged_covars.csv')
    
    newvec_df = vec_df.T
    newvec_df.columns = newvec_df.loc['k']
    newvec_df.drop(['k'], axis=0, inplace=True)
    newvec_df.reset_index(drop=True, inplace=True)
    
    merged_vec_df = pd.merge(covars_df,
                            newvec_df, 
                            left_index=True,
                            right_index=True, 
                            how='inner')
    merged_vec_df.set_index('sampleID', inplace=True)
    
    merged_integrated_df = pd.merge(
                            covars_df, 
                            integrated_df,
                            on='sampleID',
                            how='inner'
                            )
    
    merged_clinical = pd.merge( 
                        covars_df,
                        clinical_df,
                        right_on='PatientID', 
                        left_on='sampleID', 
                        how='inner')
    merged_clinical_df = merged_clinical[['PatientID', 'Death']]
    
    return cumulants_df, merged_vec_df, merged_clinical_df, merged_integrated_df

def pval_iter_cures(cumulants_df, vec_df, integrated_df, outcome, pvalues, num_iters):
    
    cures_p = {}
    cures_res_dict = {}
    beg_time = time.time()
    old_cf1 = 0
    old_croc = 0
    final_pval = 0
    
    for p in pvalues: 
        print("p-value", p)
    
        filt_cumulants_df = cumulants_df[cumulants_df['P'] < p]
        print(filt_cumulants_df.shape)
        filt_vec_df = vec_df[list(vec_df.columns[:2]) + list(filt_cumulants_df['index'])]
        
        cures_vec, _ = cures.get_cures(
                                        filt_vec_df, 
                                        outcome, 
                                        verbose=0,               
                                        # fit_cures=True,
                                        # append=True,
                                        # multi_class=False,
                                        # get_distance=True
                                        )
            
        cures_p[p] = cures_vec
        res_dict = {}
        for i in range(0, num_iters):
            print("==== Iteration "+ str(i+1) + " ====")
            print("Prediction without CuRES")
            full_df = copy.deepcopy(integrated_df)
            #print(full_df.shape)
            no_cures_res = cures.get_prediction(
                                            full_df.iloc[:,1:], 
                                            outcome,
                                            cross_val=True,
                                            verbose=1
                                            )
            
            print("Prediction with CuRES")
            full_df['cures'] = cures_vec
            cures_res = cures.get_prediction(
                                            full_df.iloc[:,1:], 
                                            outcome,
                                            cross_val=True,
                                            verbose=1
                                            )
            res_dict[i] = {
                        'Baseline F1': no_cures_res['f1'], 
                        'CuRES F1': cures_res['f1'], 
                        'Baseline ROC AUC': no_cures_res['ROC AUC'], 
                        'CuRES ROC AUC': cures_res['ROC AUC']
                        }
            full_df = pd.DataFrame()
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')    
        cures_res_dict[p] = res_df
    
    print("Time spent computing cures (mins): ", (time.time() - beg_time)/60)
    
    for k,v in cures_res_dict.items():
        _, cf1, _ ,croc = v.mean()
        if (cf1 + croc > old_cf1 + old_croc): 
            old_cf1 = cf1 
            old_croc = croc
            final_pval = k

    final_res_df = cures_res_dict[final_pval]
    
    return final_res_df, final_pval

def pval_cures(cumulants_df, vec_df, outcome, pvalues):
    best_res = []
    f1 = 0
    beg_time = time.time()

    for p in pvalues: 
        print("p-value", p)
        filt_cumulants_df = cumulants_df[cumulants_df['P'] < p]
        print(filt_cumulants_df.shape)
        filt_vec_df = vec_df[vec_df.k.isin(list(filt_cumulants_df['index']))]
        df = filt_vec_df.iloc[:,2:].T

        _, res, _ , _ , _ = cures.get_cures(
                                            df, 
                                            outcome, 
                                            verbose=1,               
                                            fit_cures=True,
                                            multi_class=False,
                                            get_distance=True
                                            )
        
        if (res['f1'] + res['ROC AUC']) > f1:
            best_res = [p,res]
            f1 = res['f1'] + res['ROC AUC']

    print("Time spent computing cures over " + str(len(pvalues)) + " p-values is " + str((time.time() - beg_time)/60) + " (mins)" )
        
    return best_res

def iter_cures(cumulants_df, vec_df, outcome, num_iters, best_pval): 

    filt_cumulants_df = cumulants_df[cumulants_df['P'] < best_pval]
    res_iters = {}
    beg_time = time.time()
    for i in range(0,num_iters): 
        filt_vec_df = vec_df[vec_df.k.isin(list(filt_cumulants_df['index']))]
        df = filt_vec_df.iloc[:,2:].T
    
        _, res, _ , _ , _  = cures.get_cures(
                                            df, 
                                            outcome, 
                                            verbose=1,               
                                            fit_cures=True,
                                            multi_class=False,
                                            get_distance=True
                                            )
        res_iters[i] = res
        
    print("Time spent computing cures over " + str(num_iters) + " iterations is " + str((time.time() - beg_time)/60) + " (mins)" )
    
    return res_iters 

def baseline_cures(integrated_df, outcome, num_iters):
    res_iters_baseline = {}
    beg_time = time.time()
    for i in range(0,num_iters): 
        
        _, res, _ , _ , _  = cures.get_cures(
                                            integrated_df.iloc[:, 1:], 
                                            outcome, 
                                            verbose=1,               
                                            fit_cures=True,
                                            multi_class=False,
                                            get_distance=True
                                            )
        res_iters_baseline[i] = res
        
    return res_iters_baseline

def plot_metrics(df_scores, data_path, cancer):
    fig, ax = plt.subplots()
    sns.boxplot(data=df_scores, 
                x='metric', 
                y='value',
                hue='method')
    plt.title(cancer)
    plt.ylabel('Value')
    plt.xlabel('Metric')
    ax.set_xticklabels(['$F_1$', 'ROC-AUC'])
    fig.savefig(data_path + '/' + cancer + '_cures_metrics.png',
                dpi=600,
                bbox_inches='tight')

def make_results(iter_df, baseline_iter_df):
    
    df_scores = pd.DataFrame()
    df_scores['cures_f1'] = iter_df['f1']
    df_scores['cures_rocauc'] = iter_df['ROC AUC']
    df_scores['baseline_f1'] = baseline_iter_df['f1']
    df_scores['baseline_rocauc'] = baseline_iter_df['ROC AUC']
    df_scores = df_scores.melt()
    df_scores['method'] = [x.split('_')[0] for x in df_scores['variable']]
    df_scores['metric'] =  [x.split('_')[1] for x in df_scores['variable']]
    df_scores.drop('variable', axis=1, inplace=True)
    
    return df_scores

def main():
    "Computing CuRES"
    parser = argparse.ArgumentParser(description="Reads a file path.", 
                                    exit_on_error=True)
    parser.add_argument(
                    "-f", 
                    "--file_path",
                    action='store', 
                    help="The path to the file."
                    )
    parser.add_argument(
                    "-c", 
                    "--cancer", 
                    action='store',
                    choices=["AML", "BIC", "SKCM", "SARC", "LUSC", "KIRC", "LIHC", "OV", "GBM", "COAD"],
                    help="Name of cancer"  
                    )
    parser.add_argument(
                    "-i", 
                    "--num_iter", 
                    action='store',
                    default=20, 
                    help="Number of iterations"  
                    )
    
    args = parser.parse_args()

    pval = np.logspace(-18,-2,12)

    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
        
    data_path = args.file_path + args.cancer
    print(data_path)
    
    cumulants_df, vec_df, clinical_df, integrated_df = read_data(data_path)
    outcome = list(clinical_df['Death'])
    
    
    print("****************")
    print("   CuRES vs. Baseline   ")
    print("****************")
    
    res_iters_df, p_val = pval_iter_cures(cumulants_df,
                                vec_df, t
                                integrated_df, 
                                outcome, 
                                pval, 
                                args.num_iter)

    print("Best results obtained for p-value: ", p_val )
    
    df_scores = res_iters_df.melt()
    df_scores['method'] = [x.split(' ')[0] for x in df_scores['variable']]
    df_scores['metric'] =  [x.split(' ')[1] for x in df_scores['variable']]
    df_scores.drop('variable', axis=1, inplace=True)
    
    df_scores.to_csv(data_path+'/accuracy_metrics.csv')
    plot_metrics(df_scores, data_path, args.cancer)
    

if __name__ == "__main__":
    main()
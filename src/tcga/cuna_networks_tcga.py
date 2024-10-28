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
    
    return cumulants_df

def main():
    "Computing CuNA"
    parser = argparse.ArgumentParser(description="Reads a file path.", 
                                    exit_on_error=True)
    parser.add_argument(
                    "-f", 
                    "--file_path",
                    action='store', 
                    required=True,
                    help="The path to the file."
                    )
    parser.add_argument(
                    "-c", 
                    "--cancer",
                    required=True, 
                    action='store',
                    choices=["AML", "BIC", "SKCM", "SARC", "LUSC", "KIRC", "LIHC", "OV", "GBM", "COAD"],
                    help="Name of cancer"  
                    )
    parser.add_argument(
                    "-p", 
                    "--p_value", 
                    nargs="+",
                    action='store',
                    default=1e-2, 
                    required=True,
                    help="P-value to filter"  
                    )

    args = parser.parse_args()

    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
        
    data_path = args.file_path + args.cancer
    print(data_path)
    
    cumulants_df= read_data(data_path)
    pvalue = list(args.p_value)
    pvalue = [float(x) for x in pvalue]
    beg_time = time.time()
    interactions, _, communities, noderank = cuna.get_network(cumulants_df.copy(),
                                                            plot_flag=0, 
                                                            pvalues=pvalue, 
                                                            #community_flag=1,
                                                            verbose=1)
    print("Time spent computing CuNA network (mins): ", (time.time() - beg_time)/60)
    
    interactions.to_csv(data_path+"/interactions_all.csv", index=False)
    noderank.to_csv(data_path+"/noderank_.csv", index=False)
    communities.to_csv(data_path+"/communities.csv", index=False)
    
if __name__ == "__main__":
    main()
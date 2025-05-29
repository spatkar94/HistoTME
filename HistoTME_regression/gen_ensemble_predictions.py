import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default=None, help='name of cohort')
    args = parser.parse_args()
    embeddings = ['uni','uni2','virchow','virchow2','hoptimus0','gigapath']

    dfs = []
    for embed in embeddings:
        dfs.append(pd.read_csv(f'predictions/{args.cohort}_predictions_{embed}.csv', index_col=0))

    df_multi = pd.concat(dfs,axis=0).groupby(level=0).mean()
    df_multi.to_csv(f'predictions/{args.cohort}_ensemble.csv')
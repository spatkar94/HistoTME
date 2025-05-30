import os
import pandas as pd
import numpy as np
import argparse

def compute_ensemble_spatial(fm_preds):
    dfs = []
    for fm in fm_preds.keys():
        histoTME = fm_preds[fm]
        histoTME['barcodes'] = histoTME.apply(lambda row: row['x'].astype(int).astype(str) + '_' + row['y'].astype(int).astype(str), axis=1)
        histoTME.index = histoTME['barcodes']
        histoTME = histoTME.drop(columns=['x','y','barcodes'],axis=1)
        dfs.append(histoTME)
    
    dfs = pd.concat(dfs)
    dfs = dfs.groupby(dfs.index).mean()

    # Split index and insert as new columns
    dfs['barcodes'] = dfs.index
    dfs[['x','y']] = dfs['barcodes'].str.split('_',n=1, expand=True)
    dfs.drop(columns=['barcodes'],axis=1, inplace=True)
    dfs = dfs.reset_index(drop=True)

    # Reorder columns to put x and y at the beginning
    cols = ['x', 'y'] + [col for col in dfs.columns if col not in ['x', 'y']]
    dfs = dfs[cols]
    return dfs

def compute_ensemble_bulk(fm_preds):
    dfs = []
    for fm in fm_preds.keys():
        dfs.append(fm_preds[fm])

    dfs = pd.concat(dfs,axis=0).groupby(level=0).mean()
    return dfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='bulk for bulk predictions, spatial for spatial predictions', default='bulk')
    parser.add_argument('--cohort', type=str, default=None, help='name of cohort (if bulk mode)')
    parser.add_argument('--filename', type=str, default=None, help='WSI slide ID (if spatial mode)')
    parser.add_argument('--save_loc', type=str, help="location where foundation model-specific predictions are stored", default="predictions")

    args = parser.parse_args()
    embeddings = ['uni','uni2','virchow','virchow2','hoptimus0','gigapath']

    
    if args.mode == 'bulk':
        if args.cohort is None:
            raise ValueError("Please provide cohort name for bulk mode")
        fm_preds = {}
        for embed in embeddings:
            fm_preds[embed] = pd.read_csv(f'{args.save_loc}/{args.cohort}_predictions_{embed}.csv', index_col=0)
        ensemble = compute_ensemble_bulk(fm_preds)
        ensemble.to_csv(f'{args.save_loc}/{args.cohort}_ensemble.csv')
        
    elif args.mode == 'spatial':
        if args.filename is None:
            raise ValueError("Please provide correct WSI filename")
        
        if not os.path.exists(os.path.join(args.save_loc,'ensemble')):
            os.mkdir(os.path.join(args.save_loc,'ensemble'))

        fm_preds = {}
        suffix = '_5fold.csv'
        for embed in embeddings:
            fm_preds[embed] = pd.read_csv(os.path.join(args.save_loc,embed,args.filename+suffix), index_col=0)
        ensemble = compute_ensemble_spatial(fm_preds)
        ensemble.to_csv(os.path.join(args.save_loc,'ensemble',args.filename+'_ensemble.csv'))
        
    else:
        raise ValueError("mode only accepts the following options: [bulk, spatial]")
    

        
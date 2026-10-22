"""Run deterministic, validation-selected recovery ablations using NumPy only."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from core import dataset, fit, evaluate, save

def experiment(out, seeds=(11,22,33), epochs=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train,val,test=dataset()
    report={'status':'new_recovery_rebuild_not_original_reproduction',
            'dataset_seed':20261022,'seeds':list(seeds),'epochs':epochs,
            'python':platform.python_version(),'numpy':np.__version__,
            'unique_records':{'train':96,'validation':32,'test':32},
            'selection':'minimum validation joint cross-entropy',
            'runs':[]}
    split_keys={name:split['keys'].tolist() for name,split in zip(('train','validation','test'),(train,val,test))}
    (out/'split_keys.json').write_text(json.dumps(split_keys,separators=(',',':'))+'\n')
    for name,conditioned,aux,consistency in [('unconditioned',False,0.,0.),('conditioned',True,0.,0.),('auxiliary',True,.3,0.),('auxiliary_consistency',True,.3,2.)]:
        for seed in seeds:
            p,h=fit(train,val,seed,epochs,aux,consistency,conditioned)
            save(p,out/f'{name}_{seed}.npz')
            (out/f'{name}_{seed}_history.json').write_text(json.dumps(h,separators=(',',':'))+'\n')
            report['runs'].append(dict(objective=name,seed=seed,aux_weight=aux,consistency_weight=consistency,**evaluate(p,test,conditioned)))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args()
    if args.epochs<1:parser.error('epochs must be positive')
    experiment(args.out,epochs=args.epochs)

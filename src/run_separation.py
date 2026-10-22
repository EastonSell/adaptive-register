"""Fixed multi-split content/formatting comparison. No remote services."""
import argparse, hashlib, json
from pathlib import Path
import core
import separation

DATASET_SEEDS=(20261022,20261023,20261024)
MODEL_SEEDS=(11,22,33)

def experiment(dataset_seeds=DATASET_SEEDS,model_seeds=MODEL_SEEDS,epochs=100):
    protocol=Path(__file__).resolve().parents[1]/'milestone2/PREREGISTRATION.md'
    report={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'dataset_seeds':list(dataset_seeds),'model_seeds':list(model_seeds),'epochs':epochs,
            'runs':[],'paired_accuracy_differences':[]}
    for ds in dataset_seeds:
        train,val,test=core.dataset(ds)
        for seed in model_seeds:
            batch=[]
            for name,conditioned,aux,cons in [('unconditioned',False,0.,0.),('conditioned',True,0.,0.),('auxiliary',True,.3,0.),('auxiliary_consistency',True,.3,2.)]:
                p,_=core.fit(train,val,seed,epochs,aux,cons,conditioned)
                batch.append(dict(dataset_seed=ds,model_seed=seed,objective=name,
                                  structural_parameters=1263,
                                  structural_nonaux_parameters=1164,
                                  active_parameters=(1263 if aux else (1164 if conditioned else 1068)),
                                  **core.evaluate(p,test,conditioned)))
            for width in (32,46):
                p,_=separation.fit(train,val,seed,epochs,width)
                row=dict(dataset_seed=ds,model_seed=seed,objective=f'separated_{width}',
                         structural_parameters=sum(v.size for v in p.values()),
                         active_parameters=sum(v.size for v in p.values()),**separation.evaluate(p,test))
                batch.append(row)
                baseline=next(r for r in batch if r['objective']=='conditioned')
                report['paired_accuracy_differences'].append(dict(dataset_seed=ds,model_seed=seed,
                    objective=row['objective'],difference=row['answer_accuracy']-baseline['answer_accuracy']))
            report['runs'].extend(batch)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=experiment();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')

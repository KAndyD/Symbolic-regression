"""30 запусков GP с мягким штрафом и без него; тест не участвует в отборе."""
import csv
import time
import numpy as np
from common import ROOT,setup,write_json,write_csv,summarize,convergence,save_histories,style,plt
from model import generate,run,size,depth,expression,mse,predict,target


def main():
    cfg,out=setup(); path=ROOT/'data/regression.csv'
    if not path.exists(): write_csv(path,generate(cfg['data_seed']))
    with path.open(encoding='utf-8') as f: data=list(csv.DictReader(f))
    splits={s:(np.array([float(r['x']) for r in data if r['split']==s]),np.array([float(r['y']) for r in data if r['split']==s])) for s in ['train','validation','test']}
    train,val,test=(splits[s] for s in ['train','validation','test']); rows=[]; histories={}; sizes={}; candidates=[]; champions={}
    for method,penalty in [('GP_plain',0.),('GP_parsimony',cfg['penalty'])]:
        histories[method]=[]; sizes[method]=[]; best_validation=float('inf')
        for seed in range(cfg['seed'],cfg['seed']+cfg['runs']):
            start=time.perf_counter(); tree,h,z,calls,vcalls,pop=run(cfg,seed,train,val,penalty)
            row=dict(method=method,seed=seed,train_mse=mse(tree,*train),validation_mse=mse(tree,*val),test_mse=mse(tree,*test),nodes=size(tree),depth=depth(tree),evaluations=calls,validation_evaluations=vcalls,seconds=time.perf_counter()-start,expression=expression(tree))
            rows.append(row); histories[method].append(h); sizes[method].append(z)
            if row['validation_mse']<best_validation:
                best_validation=row['validation_mse']; champions[method]=(tree,row)
            # Кандидаты разного размера, отбор представителей позже только по validation.
            candidates.extend((t,seed,method) for t in pop)
        print(method,'finished',flush=True)
    write_csv(out/'runs.csv',rows); write_csv(out/'summary.csv',summarize(rows,['train_mse','validation_mse','test_mse','nodes','depth','seconds','evaluations']))
    save_histories(out/'histories.csv',histories); save_histories(out/'size_histories.csv',sizes)
    convergence(out/'convergence.png',histories,ylabel='Минимальная train MSE в популяции'); convergence(out/'bloat.png',sizes,ylabel='Среднее число узлов',title='Разрастание деревьев: среднее и min–max по запускам')
    unique={}
    for t,seed,method in candidates:
        key=size(t); error=mse(t,*val)
        if key not in unique or error<unique[key][0]: unique[key]=(error,t,seed,method)
    keys=sorted(unique)
    # Малое, среднее, большое из доступных размеров; никакого выбора по тесту.
    pick=[keys[0],keys[len(keys)//2],keys[-1]]
    representatives=[]
    for k in dict.fromkeys(pick):
        _,t,seed,method=unique[k]; representatives.append(dict(method=method,seed=seed,nodes=k,tree=t,expression=expression(t),train_mse=mse(t,*train),validation_mse=mse(t,*val),test_mse=mse(t,*test)))
    write_json(out/'representatives.json',representatives)
    write_json(out/'best.json',{m:dict(**r,tree=t) for m,(t,r) in champions.items()})
    tx,ty=train; design=np.column_stack([np.ones(len(tx)),tx]); coeff=np.linalg.lstsq(design,ty,rcond=None)[0]
    baseline=[]
    for name in ['constant','linear']:
        row={'method':name}
        for split,(xx,yy) in splits.items():
            prediction=np.full_like(xx,ty.mean()) if name=='constant' else coeff[0]+coeff[1]*xx
            row[split+'_mse']=float(np.mean((prediction-yy)**2))
        baseline.append(row)
    write_csv(out/'baselines.csv',baseline)
    style(); fig,ax=plt.subplots(figsize=(10,5)); xx=np.linspace(-2,2,400); ax.scatter(*test,s=18,alpha=.45,label='Test'); ax.plot(xx,target(xx),'k--',label='Истинная функция')
    for m,(t,_) in champions.items(): ax.plot(xx,predict(t,xx),label=m)
    ax.set(xlabel='x',ylabel='y',title='Представители выбраны по validation MSE'); ax.legend(); fig.tight_layout(); fig.savefig(out/'predictions.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for m in histories:
        rr=[r for r in rows if r['method']==m]; ax.scatter([r['nodes'] for r in rr],[r['test_mse'] for r in rr],label=m)
    ax.set(xlabel='Число узлов',ylabel='Test MSE',title='Качество и сложность: все независимые запуски'); ax.legend(); fig.tight_layout(); fig.savefig(out/'quality_complexity.png'); plt.close(fig)

if __name__=='__main__': main()

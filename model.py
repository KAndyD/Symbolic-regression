"""Генетическое программирование над неизменяемыми деревьями-кортежами."""
import numpy as np

ARITY={'+':2,'-':2,'*':2,'/':2,'sin':1}


def size(tree):
    return 1+sum(size(child) for child in tree[1:]) if isinstance(tree,tuple) else 1


def depth(tree):
    return 1+max(depth(child) for child in tree[1:]) if isinstance(tree,tuple) else 1


def expression(tree):
    if not isinstance(tree,tuple): return str(tree) if isinstance(tree,str) else f'{tree:.5g}'
    if tree[0]=='sin': return f'sin({expression(tree[1])})'
    if tree[0]=='/': return f'pdiv({expression(tree[1])}, {expression(tree[2])})'
    return f'({expression(tree[1])} {tree[0]} {expression(tree[2])})'


def predict(tree,x):
    if not isinstance(tree,tuple): return x if tree=='x' else np.full_like(x,float(tree))
    a=predict(tree[1],x)
    if tree[0]=='sin': return np.sin(a)
    b=predict(tree[2],x)
    if tree[0]=='+': result=a+b
    elif tree[0]=='-': result=a-b
    elif tree[0]=='*': result=a*b
    else:
        result=np.ones_like(a); np.divide(a,b,out=result,where=np.abs(b)>1e-6)
    return np.clip(result,-1e6,1e6)


def random_tree(rng,limit=4):
    if limit<=1 or rng.random()<.25:
        return 'x' if rng.random()<.65 else round(float(rng.uniform(-2,2)),3)
    op=rng.choice(list(ARITY)); return (str(op),*(random_tree(rng,limit-1) for _ in range(ARITY[op])))


def paths(tree,prefix=()):
    result=[prefix]
    if isinstance(tree,tuple):
        for i,child in enumerate(tree[1:],1): result.extend(paths(child,prefix+(i,)))
    return result


def subtree(tree,path):
    for i in path: tree=tree[i]
    return tree


def replace(tree,path,new):
    if not path: return new
    values=list(tree); values[path[0]]=replace(values[path[0]],path[1:],new); return tuple(values)


def valid(tree,cfg):
    return depth(tree)<=cfg['max_depth'] and size(tree)<=cfg['max_nodes']


def variation(a,b,rng,cfg):
    child=a
    if rng.random()<cfg['crossover_probability']:
        pa,pb=paths(a),paths(b)
        proposal=replace(a,pa[int(rng.integers(len(pa)))],subtree(b,pb[int(rng.integers(len(pb)))]))
        if valid(proposal,cfg): child=proposal
    if rng.random()<cfg['mutation_probability']:
        pp=paths(child); proposal=replace(child,pp[int(rng.integers(len(pp)))],random_tree(rng,3))
        if valid(proposal,cfg): child=proposal
    return child


def target(x): return x*x+np.sin(3*x)


def generate(seed):
    rng=np.random.default_rng(seed); x=rng.uniform(-2,2,240); y=target(x)+rng.normal(0,.05,len(x))
    order=rng.permutation(len(x)); split=np.empty(len(x),dtype=object)
    split[order[:120]]='train'; split[order[120:180]]='validation'; split[order[180:]]='test'
    return [dict(x=float(a),y=float(b),split=str(s)) for a,b,s in zip(x,y,split)]


def mse(tree,x,y): return float(np.mean((predict(tree,x)-y)**2))


def run(cfg,seed,train,validation,penalty):
    rng=np.random.default_rng(seed); n=cfg['population']; x,y=train; vx,vy=validation
    pop=[random_tree(rng,3+i%3) for i in range(n)]
    errors=np.array([mse(t,x,y) for t in pop]); scores=errors+penalty*np.array([size(t) for t in pop]); calls=n
    history=[float(errors.min())]; sizes=[float(np.mean([size(t) for t in pop]))]
    for _ in range(cfg['generations']):
        ids=rng.integers(n,size=(2,n,3)); wins=ids[np.arange(2)[:,None],np.arange(n)[None,:],scores[ids].argmin(2)]
        children=[variation(pop[a],pop[b],rng,cfg) for a,b in zip(*wins)]
        ce=np.array([mse(t,x,y) for t in children]); calls+=n
        merged=pop+children; all_errors=np.r_[errors,ce]; all_scores=all_errors+penalty*np.array([size(t) for t in merged])
        order=np.argsort(all_scores,kind='stable')[:n]
        pop=[merged[i] for i in order]; errors=all_errors[order]; scores=all_scores[order]
        history.append(float(errors.min())); sizes.append(float(np.mean([size(t) for t in pop])))
    # Ровно n validation-оценок, одинаково для обоих методов. Тест сюда не передаётся.
    validation_errors=np.array([mse(t,vx,vy) for t in pop]); best=min(range(n),key=lambda i:(validation_errors[i],size(pop[i])))
    return pop[best],history,sizes,calls,n,pop

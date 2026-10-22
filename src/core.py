"""Independent recovery rebuild: structured synthetic objective study, not an LLM."""
import itertools
from pathlib import Path
import numpy as np

REGISTERS = ('introductory', 'working', 'technical')

def dataset(seed=20261022):
    """Unique records split before query and register expansion."""
    rng = np.random.default_rng(seed)
    records = np.array(list(itertools.product(range(4), repeat=4)), dtype=int)
    records = records[rng.permutation(len(records))[:160]]
    result = []
    for part in (records[:96], records[96:128], records[128:]):
        subset = np.array([list(record) + [q] for record in part for q in range(4)])
        values, query = subset[:, :4], subset[:, 4]
        base = np.concatenate((np.eye(4)[values].reshape(-1, 16), np.eye(4)[query]), axis=1)
        x = np.concatenate((np.repeat(base, 3, axis=0), np.tile(np.eye(3), (len(base), 1))), axis=1)
        answer = np.repeat(values[np.arange(len(values)), query], 3)
        result.append(dict(keys=subset, x=x, answer=answer, register=np.tile(np.arange(3), len(base))))
    return result

def initialize(seed, hidden=32):
    rng = np.random.default_rng(seed)
    return {'W': rng.normal(0, .15, (23, hidden)), 'b': np.zeros(hidden),
            'J': rng.normal(0, .15, (hidden, 12)), 'c': np.zeros(12),
            'A': rng.normal(0, .15, (hidden, 3)), 'd': np.zeros(3)}

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(p, x):
    h = np.tanh(x @ p['W'] + p['b'])
    return h, softmax(h @ p['J'] + p['c']), softmax(h @ p['A'] + p['d'])

def objective(p, x, answer, register, aux=0., consistency=0.):
    h, joint, extra = forward(p, x)
    n = len(x); target = answer * 3 + register
    loss = -np.log(joint[np.arange(n), target]).mean()
    dj = joint.copy(); dj[np.arange(n), target] -= 1; dj /= n
    da = extra.copy(); da[np.arange(n), register] -= 1; da *= aux / n
    loss -= aux * np.log(extra[np.arange(n), register]).mean()
    if consistency:
        # Mean squared discrepancy across the three explicit register requests.
        marginal = joint.reshape(n, 4, 3).sum(axis=2).reshape(-1, 3, 4)
        delta = marginal - marginal.mean(axis=1, keepdims=True)
        loss += consistency * np.mean(delta ** 2)
        dp = np.repeat((2 * consistency * delta / delta.size).reshape(n, 4), 3, axis=1)
        dj += joint * (dp - (dp * joint).sum(axis=1, keepdims=True))
    dh = (dj @ p['J'].T + da @ p['A'].T) * (1 - h ** 2)
    return float(loss), {'W': x.T @ dh, 'b': dh.sum(axis=0), 'J': h.T @ dj,
                         'c': dj.sum(axis=0), 'A': h.T @ da, 'd': da.sum(axis=0)}

def fit(train, validation, seed, epochs=100, aux=0., consistency=0., conditioned=True):
    p = initialize(seed); m = {k: np.zeros_like(v) for k,v in p.items()}; v = {k: a.copy() for k,a in m.items()}
    x = train['x'].copy(); vx = validation['x'].copy()
    if not conditioned: x[:, -3:] = 0; vx[:, -3:] = 0
    best = float('inf'); chosen = None; history = []
    for step in range(1, epochs + 1):
        loss, grad = objective(p,x,train['answer'],train['register'],aux,consistency)
        for k in p:
            m[k] = .9*m[k] + .1*grad[k]; v[k] = .999*v[k] + .001*grad[k]**2
            p[k] -= .01 * (m[k]/(1-.9**step)) / (np.sqrt(v[k]/(1-.999**step))+1e-8)
        vl = objective(p,vx,validation['answer'],validation['register'])[0]
        history.append({'epoch':step,'train_loss':loss,'validation_joint_loss':vl})
        if vl < best: best=vl; chosen={k:a.copy() for k,a in p.items()}
    return chosen, history

def predict(p,x):
    joint = forward(p,x)[1].reshape(-1,4,3)
    return joint.sum(axis=2).argmax(axis=1), joint.sum(axis=1).argmax(axis=1)

def evaluate(p, data, conditioned=True):
    x=data['x'].copy()
    if not conditioned:x[:,-3:]=0
    a,r=predict(p,x);groups=a.reshape(-1,3)
    return {'answer_accuracy':float(np.mean(a==data['answer'])),
            'requested_register_agreement':float(np.mean(r==data['register'])),
            'answer_change_rate':float(np.mean(np.any(groups!=groups[:,:1],axis=1))),
            'all_registers_answer_correct':float(np.mean(np.all((a==data['answer']).reshape(-1,3),axis=1)))}

def save(p,path):
    np.savez(path,**p)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def render(answer,register='working'):
    if register not in REGISTERS:raise ValueError('Use an explicit supported register, not an inferred personal trait.')
    if answer not in range(4):raise ValueError('Answer must be a value from the record vocabulary.')
    return {'introductory':f'The requested value is {answer}. A slot is a labeled place in the record.',
            'working':f'The requested value is {answer}.',
            'technical':f'Record lookup result: {answer}.'}[register]

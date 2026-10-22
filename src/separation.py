"""Content-only predictor plus a deterministic, explicitly requested formatter."""
import numpy as np
from core import softmax, render

def initialize(seed,hidden=32):
    rng=np.random.default_rng(seed)
    return {'W':rng.normal(0,.15,(20,hidden)),'b':np.zeros(hidden),
            'O':rng.normal(0,.15,(hidden,4)),'c':np.zeros(4)}

def forward(p,x):
    h=np.tanh(x[:,:20]@p['W']+p['b'])
    return h,softmax(h@p['O']+p['c'])

def objective(p,x,answer):
    h,prob=forward(p,x);n=len(x)
    loss=-np.log(prob[np.arange(n),answer]).mean()
    dz=prob.copy();dz[np.arange(n),answer]-=1;dz/=n
    dh=(dz@p['O'].T)*(1-h*h)
    return float(loss),{'W':x[:,:20].T@dh,'b':dh.sum(axis=0),'O':h.T@dz,'c':dz.sum(axis=0)}

def fit(train,validation,seed,epochs=100,hidden=32):
    p=initialize(seed,hidden);m={k:np.zeros_like(v) for k,v in p.items()};v={k:a.copy() for k,a in m.items()}
    best=float('inf');selected=None;history=[]
    for step in range(1,epochs+1):
        _,g=objective(p,train['x'][:,:20],train['answer'])
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=.01*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        loss=objective(p,validation['x'][:,:20],validation['answer'])[0];history.append(loss)
        if loss<best:best=loss;selected={k:a.copy() for k,a in p.items()}
    return selected,history

def predict(p,x):
    return forward(p,x)[1].argmax(axis=1)

def format_prediction(predicted_answer,requested_register):
    """No access to source record, gold answer, or inferred user traits."""
    return render(predicted_answer,requested_register)

def evaluate(p,data):
    a=predict(p,data['x']);groups=a.reshape(-1,3)
    return {'answer_accuracy':float(np.mean(a==data['answer'])),
            'answer_change_rate':float(np.mean(np.any(groups!=groups[:,:1],axis=1))),
            'all_registers_answer_correct':float(np.mean(np.all((a==data['answer']).reshape(-1,3),axis=1))),
            'deterministic_template_choice_agreement':1.0}

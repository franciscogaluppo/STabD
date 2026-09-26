#!/usr/bin/env python3
"""Stochastic Tabular Descent (STabD): a TabPFN v3.5 learned optimizer.

A validation-guided beam search trains small networks on y=sin(x). Its chosen
steps become a tabular dataset, and TabPFN predicts, in context, how to correct
a fresh minibatch gradient toward the beam's choice. See README.md.

Install (Python >=3.10):
    pip install -U tabpfn numpy matplotlib scikit-learn tqdm
TabPFN v3.5 weights require accepting the Prior Labs licence once (browser
login, or export TABPFN_TOKEN=<API key from https://ux.priorlabs.ai>).

Run (defaults reproduce the reported experiment; about 2 h of CPU for the
teacher and about 18 h on one A100 for the learned rollout):
    python STabD.py --device cuda
Reuse a saved teacher and change only the learned policy:
    python STabD.py --device cuda --context-file stabd_run/contexts.npz \
        --features grad --out stabd_grad
Check the mechanics without TabPFN:
    python STabD.py --self-test
    python STabD.py --backend ridge --meta-train 8 --meta-test 4 --steps 20 \
        --beam-width 4 --candidates 8 --out smoke

Outputs: comparison.png/pdf, metrics.json, contexts.npz, baselines.npz,
trajectories.npz. Test data never select beam children or hyperparameters.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    import torch
    from tabpfn import TabPFNRegressor
    try:
        # Older releases export ModelSpecs as a non-callable union alias.
        from tabpfn.base import RegressorModelSpecs as RegressionModelSpecs
    except ImportError:
        # Newer releases unify classifier/regressor specs into one dataclass.
        from tabpfn.base import ModelSpecs as RegressionModelSpecs
    from tabpfn.constants import ModelVersion
except ImportError as exc:
    TABPFN_IMPORT_ERROR = exc
else:
    TABPFN_IMPORT_ERROR = None

matplotlib.use('Agg')

MODEL_VERSIONS = {'3': 'V3', '3.5': 'V3_5', '3.5-fast': 'V3_5_FAST'}

D = 141
SLICES = [(0,10,(1,10)), (10,20,(10,)), (20,120,(10,10)),
          (120,130,(10,)), (130,140,(10,1)), (140,141,(1,))]


def unpack(theta):
    return [theta[..., a:b].reshape(theta.shape[:-1] + shape)
            for a,b,shape in SLICES]


def initialize(rng):
    # Same bounds as torch.nn.Linear.reset_parameters, independent RNG stream.
    return np.concatenate([rng.uniform(-1/np.sqrt(fan), 1/np.sqrt(fan), n)
                           for fan,n in [(1,10),(1,10),(10,100),
                                         (10,10),(10,10),(10,1)]])


def forward(theta, x):
    """Batched networks (P,D), common (N,1) or per-network (P,N,1) x."""
    w1,b1,w2,b2,w3,b3 = unpack(theta)
    z1 = np.matmul(x,w1) + b1[:,None,:]
    h1 = np.maximum(z1,0)
    z2 = h1 @ w2 + b2[:,None,:]
    h2 = np.maximum(z2,0)
    return h2 @ w3 + b3[:,None,:], (z1,h1,z2,h2)


def gradient(theta, x, y):
    """Exact reverse-mode derivatives of minibatch MSE, implemented in NumPy."""
    pred,(z1,h1,z2,h2) = forward(theta,x)
    w1,b1,w2,b2,w3,b3 = unpack(theta)
    e = 2*(pred-y)/pred.shape[1]
    g3 = h2.swapaxes(-1,-2) @ e
    gb3 = e.sum(1)
    d2 = (e @ w3.swapaxes(-1,-2))*(z2>0)
    g2 = h1.swapaxes(-1,-2) @ d2
    gb2 = d2.sum(1)
    d1 = (d2 @ w2.swapaxes(-1,-2))*(z1>0)
    xx = np.broadcast_to(x,(len(theta),pred.shape[1],1))
    g1 = xx.swapaxes(-1,-2) @ d1
    return np.concatenate([v.reshape(len(theta),-1)
                           for v in (g1,d1.sum(1),g2,gb2,g3,gb3)],axis=1)


def losses(theta, x, y):
    with np.errstate(over='ignore', invalid='ignore'):
        out = ((forward(theta,x)[0]-y)**2).mean((1,2))
    return np.where(np.isfinite(out),out,np.inf)


def sample_data(rng,n):
    x = rng.uniform(-5,5,(n,1))
    return x,np.sin(x)


def beam(theta, train, val, a, rng):
    """Return coherent winning path, selected gradients, and gradient-call count."""
    current = theta[None].copy()
    velocity = np.zeros_like(current)  # one momentum buffer per beam node
    layers, parents, edges = [current], [], []
    cost = 0
    scales = np.asarray(getattr(a,'grad_scales',(1.0,)))
    for _ in range(a.steps):
        pk = np.repeat(np.arange(len(current)),a.candidates)
        ix = rng.integers(len(train[0]),size=(len(pk),a.batch_size))
        gk = gradient(current[pk],train[0][ix],train[1][ix])
        # Each minibatch gradient is tried at every step scale; the scaled
        # gradient is what enters the momentum buffer and the context target.
        p = np.repeat(pk,len(scales)); base = current[p]
        g = (gk[:,None,:]*scales[None,:,None]).reshape(-1,D)
        with np.errstate(over='ignore',invalid='ignore'):
            children, child_v = momentum_step(base, g, velocity[p], a.lr, a.momentum)
        score = losses(children,*val)
        keep = np.argsort(score,kind='stable')[:a.beam_width]
        keep = keep[np.isfinite(score[keep])]
        if not len(keep):
            raise RuntimeError('All beam children diverged; reduce --lr.')
        cost += len(pk)
        parents.append(p[keep]); edges.append(g[keep])
        current = children[keep]; velocity = child_v[keep]; layers.append(current)
    # Layers are sorted by validation score; leaf zero is final winner.
    j = 0
    states, grads = [layers[-1][0]], []
    for t in range(a.steps-1,-1,-1):
        grads.append(edges[t][j]); j = parents[t][j]
        states.append(layers[t][j])
    return np.array(states[::-1]),np.array(grads[::-1]),cost


def momentum_step(theta, grad, velocity, lr, momentum):
    """Classical momentum, no dampening/Nesterov; one buffer per network."""
    velocity = momentum*velocity + grad
    return theta-lr*velocity, velocity


def velocities(grads, momentum):
    """Momentum buffers BEFORE each step of a replayed gradient sequence."""
    out = np.zeros_like(grads); v = np.zeros_like(grads[...,0,:])
    for t in range(grads.shape[-2]):
        out[...,t,:] = v; v = momentum*v + grads[...,t,:]
    return out


def minibatch_gradient(theta, train, batch_size, rng):
    """One fresh minibatch gradient per row of theta (P,D)."""
    ix = rng.integers(len(train[0]),size=(len(theta),batch_size))
    with np.errstate(over='ignore',invalid='ignore'):
        return gradient(theta,train[0][ix],train[1][ix])


def policy_inputs(theta, observed, velocity, features):
    """Policy features; gradient modes predict a residual on the observed gradient."""
    return np.concatenate({'theta':[theta],'grad':[theta,observed],
                           'grad_vel':[theta,observed,velocity]}[features],axis=1)


def sgd(theta, train, a, rng):
    states = [theta.copy()]
    velocity = np.zeros_like(theta)
    for _ in range(a.steps):
        ix = rng.integers(len(train[0]),size=a.batch_size)
        with np.errstate(over='ignore',invalid='ignore'):
            g = gradient(theta[None],train[0][ix],train[1][ix])[0]
            theta, velocity = momentum_step(theta, g, velocity, a.lr, a.momentum)
        states.append(theta.copy())
    return np.array(states)


def fit_tabpfn_context(model, X, y):
    """Suppress only TabPFN's advisory about large CPU contexts."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Running on CPU with more than [0-9,]+ samples may be slow\.',
            category=UserWarning,
        )
        return model.fit(X, y)


class IndependentTabPFN:
    def __init__(self,a):
        if TABPFN_IMPORT_ERROR is not None:
            raise ImportError('Install or upgrade tabpfn and torch to use this backend.') from TABPFN_IMPORT_ERROR
        self.a = a; self.cls = TabPFNRegressor
        # Pin to one device: auto can otherwise select multiple GPUs and clone weights.
        device = a.device
        if device == 'auto':
            device = ('cuda:0' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
        kw = dict(device=device,n_estimators=a.n_estimators,
                  random_state=a.seed,fit_mode=('fit_preprocessors'
                  if a.fit_mode=='refit' else a.fit_mode),
                  keep_cache_on_device=a.cache_device=='gpu')
        self.kw = kw
        if a.model_path:
            self.first = self.cls(model_path=a.model_path,**kw)
        else:
            name = MODEL_VERSIONS[a.version]
            if not hasattr(ModelVersion,name):
                raise RuntimeError(f'Upgrade tabpfn: requested version {a.version} ({name}) is unavailable.')
            self.first = self.cls.create_default_for_version(getattr(ModelVersion,name),**kw)

    def fit(self,X,Y):
        self.X = X.astype(np.float32); self.Y = Y.astype(np.float32)
        total = 1 if self.a.fit_mode == 'refit' else Y.shape[1]
        with tqdm(total=total, desc='Fitting TabPFN contexts', unit='target') as progress:
            fit_tabpfn_context(self.first,self.X,self.Y[:,0])
            progress.update(1)
            self.spec = RegressionModelSpecs(model=self.first.models_[0],
                architecture_config=self.first.configs_[0],
                inference_config=self.first.inference_config_,
                norm_criterion=self.first.znorm_space_bardist_)
            if len(self.first.models_) != 1:
                raise RuntimeError('Expected a single checkpoint backbone.')
            self.heads = [self.first]
            # Probe first context before/after fitting other targets to catch shared
            # mutable cache state in an incompatible package release.
            probe = self.X[:min(3,len(X))]
            before = np.asarray(self.first.predict(probe))
            if self.a.fit_mode=='refit':
                self.first.model_path = self.spec
                return self
            for j in range(1,Y.shape[1]):
                m = self.cls(model_path=self.spec,**self.kw)
                fit_tabpfn_context(m,self.X,self.Y[:,j])
                if m.models_[0] is not self.first.models_[0]:
                    raise RuntimeError('Installed TabPFN copied the shared model.')
                # Verify the actual inference engine too, not just estimator metadata.
                if hasattr(m.executor_, 'model_caches'):
                    for device in m.devices_:
                        if m.executor_.model_caches[0].get(device) is not self.spec.model:
                            raise RuntimeError('Inference engine duplicated the backbone.')
                self.heads.append(m)
                progress.update(1)
            np.testing.assert_allclose(self.first.predict(probe),before,rtol=2e-4,atol=2e-5,
                err_msg='Fitting another target changed the first target cache.')
            return self

    def predict(self,X):
        X = X.astype(np.float32)
        if self.a.fit_mode=='refit':
            out = []
            for j in range(self.Y.shape[1]):
                fit_tabpfn_context(self.first,self.X,self.Y[:,j])
                out.append(self.first.predict(X))
            return np.column_stack(out)
        return np.column_stack([m.predict(X) for m in self.heads])


def self_test():
    rng = np.random.default_rng(91)
    theta = initialize(rng); x,y = sample_data(rng,13)
    g = gradient(theta[None],x,y)[0]
    numeric = np.empty(D); eps=1e-6
    for j in range(D):
        d=np.zeros(D); d[j]=eps
        numeric[j]=(losses((theta+d)[None],x,y)[0]-
                    losses((theta-d)[None],x,y)[0])/(2*eps)
    np.testing.assert_allclose(g,numeric,rtol=1e-4,atol=1e-7)
    val=sample_data(rng,31)
    for momentum in (0.0,0.9):
        a=argparse.Namespace(steps=5,candidates=4,beam_width=3,batch_size=8,lr=.01,momentum=momentum,
                             grad_scales=(0.5,1.0,2.0))
        path,gs,cost=beam(theta,(x,y),val,a,np.random.default_rng(1))
        # Replaying the selected gradients through momentum reproduces the winning path.
        replay=[theta]; v=np.zeros_like(theta)
        for g in gs:
            nxt,v=momentum_step(replay[-1],g,v,a.lr,a.momentum); replay.append(nxt)
        np.testing.assert_allclose(path,np.array(replay),atol=1e-14)
        assert cost==4+4*3*4
        # Width one, one candidate must exactly reproduce ordinary (momentum) SGD.
        a.beam_width=a.candidates=1; a.grad_scales=(1.0,)
        path,_,_=beam(theta,(x,y),(x,y),a,np.random.default_rng(2))
        plain=sgd(theta,(x,y),a,np.random.default_rng(2))
        np.testing.assert_allclose(path,plain,atol=1e-14)
    print('PASS: all 141 finite-difference gradients, beam ancestry, budget, SGD equivalence '
          '(momentum 0 and 0.9, step scales).')


def mean_cosine(pred, truth):
    return float(np.mean(np.sum(pred*truth,axis=1)/
        np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1),1e-15)))


def plot_results(out, curves, a):
    """SGD vs the learned optimizer only; beam curves stay in trajectories.npz."""
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    colors={'sgd':'#d58936','meta':'#344d9b'}
    names={'sgd':f'SGD (momentum={a.momentum:g})',
           'meta':'Stochastic Tabular Descent' if a.backend=='tabpfn' else 'Ridge smoke test'}
    t=np.arange(a.steps+1)
    for split,ax in zip(('val','test'),axes):
        for method in ('sgd','meta'):
            ys=curves[method+'_'+split]
            # Keep finite catastrophic losses; nonfinite values cannot be plotted.
            for row in ys:
                ax.plot(t,np.where(np.isfinite(row),np.maximum(row,1e-12),np.nan),
                        color=colors[method],alpha=.10,lw=.7)
            med=np.median(ys,axis=0)
            lo,hi=np.quantile(ys,[.25,.75],axis=0,method='nearest')
            ax.plot(t,np.where(np.isfinite(med),np.maximum(med,1e-12),np.nan),
                    color=colors[method],lw=2,label=names[method])
            ok=np.isfinite(lo)&np.isfinite(hi)
            ax.fill_between(t,np.maximum(lo,1e-12),np.maximum(hi,1e-12),
                            where=ok,color=colors[method],alpha=.13)
        ax.set(title=f'Held-out initializations: {split} MSE',xlabel='Update',
               ylabel='MSE',yscale='log')
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    failed={m:int(np.sum(~np.isfinite(curves[m+'_test'][:,-1]))) for m in ('sgd','meta')}
    fig.suptitle(f'Sine regression | {a.meta_test} held-out starts | '
                 f'median / IQR; faint individual paths | nonfinite failures {failed}',fontsize=10)
    fig.savefig(out/'comparison.png',dpi=180)
    fig.savefig(out/'comparison.pdf')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out',type=Path,default=Path('stabd_run'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--meta-train',type=int,default=512)
    p.add_argument('--meta-test',type=int,default=16)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--beam-width',type=int,default=16)
    p.add_argument('--candidates',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--momentum', type=float, default=0.9,
                   help='Momentum for beam, SGD and learned updates (default: 0.9)')
    p.add_argument('--grad-scales',type=lambda s:tuple(float(v) for v in s.split(',')),default=(0.5,1.0,2.0),
                   help='Comma-separated step-size multipliers searched by the beam (default: 0.5,1,2)')
    p.add_argument('--features',choices=['theta','grad','grad_vel'],default='grad_vel',
                   help='Policy inputs (default: grad_vel)')
    p.add_argument('--feature-draws',type=int,default=1,
                   help='Observed-gradient draws per teacher state (default: 1)')
    p.add_argument('--context-trajectories',type=int,
                   help='Fit on only the first N teacher trajectories (default: all)')
    p.add_argument('--n-train',type=int,default=256)
    p.add_argument('--n-val',type=int,default=512)
    p.add_argument('--n-test',type=int,default=2048)
    p.add_argument('--backend',choices=['tabpfn','ridge'],default='tabpfn')
    p.add_argument('--version', choices=list(MODEL_VERSIONS), default='3.5',
                   help='TabPFN checkpoint version (default: 3.5; overridden by --model-path)')
    p.add_argument('--model-path')
    p.add_argument('--device',default='auto')
    p.add_argument('--n-estimators',type=int,default=1)
    p.add_argument('--fit-mode',choices=['fit_with_cache','fit_preprocessors','refit'],
                   default='fit_preprocessors',
                   help='fit_with_cache stores a KV cache per coordinate (fast, very memory hungry); '
                        'fit_preprocessors recomputes the context each call (default); '
                        'refit keeps one estimator and refits every call (slowest)')
    p.add_argument('--cache-device',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--context-file',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test(); return
    for name in ('meta_train','meta_test','steps','beam_width','candidates','batch_size',
                 'n_train','n_val','n_test','n_estimators','feature_draws'):
        if getattr(a,name)<1: p.error(name+' must be positive')
    if not np.isfinite(a.lr) or a.lr<=0: p.error('--lr must be finite and positive')
    if not np.isfinite(a.momentum) or not 0 <= a.momentum < 1:
        p.error('--momentum must be finite and in [0, 1)')
    if not all(np.isfinite(a.grad_scales)) or min(a.grad_scales)<=0:
        p.error('--grad-scales must be finite and positive')
    if a.context_trajectories is not None and not 1<=a.context_trajectories<=a.meta_train:
        p.error('--context-trajectories must be in [1, meta_train]')
    a.out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (a.out/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(np.random.SeedSequence([a.seed,0]))
    train=sample_data(rng,a.n_train); val=sample_data(rng,a.n_val); test=sample_data(rng,a.n_test)
    context_keys=('seed','meta_train','steps','beam_width','candidates','batch_size',
                  'lr','momentum','grad_scales','feature_draws','n_train','n_val','n_test')
    signature=json.dumps({k:config[k] for k in context_keys},sort_keys=True)
    timers={}; start=time.perf_counter()
    if a.context_file:
        with np.load(a.context_file,allow_pickle=False) as z:
            if str(z['signature'])!=signature:
                raise ValueError('Context configuration differs; reuse original teacher/data flags.')
            X=z['X']; Y=z['Y']; G=z['G']; V=z['V']
            teacher_paths=z['teacher_paths']; teacher_cost=int(z['teacher_cost'])
    else:
        teacher_paths=[]; targets=[]; teacher_cost=0
        progress = tqdm(range(a.meta_train), desc="Collecting beam trajectories", unit="trajectory")
        for i in progress:
            r=np.random.default_rng(np.random.SeedSequence([a.seed,1,i]))
            path,g,cost=beam(initialize(r),train,val,a,r)
            teacher_paths.append(path); targets.append(g); teacher_cost+=cost
            progress.set_postfix(val_mse=f'{losses(path[-1:],*val)[0]:.5f}')
        teacher_paths=np.array(teacher_paths); targets=np.array(targets)
        # Rows are trajectory-major (trajectory, step, draw) so prefixes are whole trajectories.
        k=a.feature_draws
        X=np.repeat(teacher_paths[:,:-1].reshape(-1,D),k,axis=0)
        Y=np.repeat(targets.reshape(-1,D),k,axis=0)
        V=np.repeat(velocities(targets,a.momentum).reshape(-1,D),k,axis=0)
        G=minibatch_gradient(X,train,a.batch_size,np.random.default_rng(np.random.SeedSequence([a.seed,5])))
    timers['teacher_generation_or_load_s']=time.perf_counter()-start
    np.savez_compressed(a.out/'contexts.npz',X=X,Y=Y,G=G,V=V,teacher_paths=teacher_paths,
        teacher_cost=teacher_cost,signature=signature,
        train_x=train[0],train_y=train[1],val_x=val[0],val_y=val[1],test_x=test[0],test_y=test[1])
    if a.context_trajectories is not None:
        n=a.context_trajectories*a.steps*a.feature_draws
        X,Y,G,V=X[:n],Y[:n],G[:n],V[:n]
    grad_features=a.features!='theta'
    CX=policy_inputs(X,G,V,a.features); CY=Y-G if grad_features else Y
    print(f'Context: {len(CX)} rows, {CX.shape[1]} features ({a.features}), {D} independent targets',flush=True)
    starts=[]; beams=[]; beam_grads=[]; sgds=[]; held_cost=0; beam_s=sgd_s=0.
    for i in tqdm(range(a.meta_test), desc="Collecting evaluation trajectories", unit="trajectory"):
        init_rng=np.random.default_rng(np.random.SeedSequence([a.seed,2,i]))
        theta=initialize(init_rng); starts.append(theta)
        start=time.perf_counter()
        path,g,cost=beam(theta,train,val,a,np.random.default_rng(np.random.SeedSequence([a.seed,3,i])))
        beam_s+=time.perf_counter()-start; beams.append(path); beam_grads.append(g); held_cost+=cost
        start=time.perf_counter()
        sgds.append(sgd(theta,train,a,np.random.default_rng(np.random.SeedSequence([a.seed,4,i]))))
        sgd_s+=time.perf_counter()-start
    timers.update(heldout_beam_s=beam_s,heldout_sgd_s=sgd_s)
    np.savez_compressed(a.out/'baselines.npz',beam=np.array(beams),sgd=np.array(sgds),starts=starts)
    start=time.perf_counter()
    if a.backend=='tabpfn':
        model=IndependentTabPFN(a).fit(CX,CY)
    else:
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(CX,CY)

    def policy(theta, velocity, rng):
        observed=(minibatch_gradient(theta,train,a.batch_size,rng) if grad_features
                  else np.zeros_like(theta))
        inputs=policy_inputs(theta,observed,velocity,a.features)
        # Diverged networks can have nonfinite gradients; mark them failed.
        ok=np.isfinite(inputs).all(1); pred=np.full_like(theta,np.nan)
        if ok.any(): pred[ok]=model.predict(inputs[ok])
        return observed+pred if grad_features else pred
    timers['meta_fit_s']=time.perf_counter()-start
    theta=np.array(starts); meta=[theta.copy()]; start=time.perf_counter()
    velocity=np.zeros_like(theta)
    rollout_rng=np.random.default_rng(np.random.SeedSequence([a.seed,6]))
    for t in tqdm(range(a.steps), desc="Learned rollout", unit="step"):
        alive=np.isfinite(theta).all(1)&np.isfinite(velocity).all(1)
        g=np.full_like(theta,np.nan)
        if alive.any(): g[alive]=policy(theta[alive],velocity[alive],rollout_rng)
        with np.errstate(over='ignore',invalid='ignore'):
            theta, velocity = momentum_step(theta, g, velocity, a.lr, a.momentum)
        meta.append(theta.copy())
    timers['meta_rollout_s']=time.perf_counter()-start
    paths={'beam':np.array(beams),'sgd':np.array(sgds),'meta':np.stack(meta,axis=1)}
    curves={'teacher_val':np.array([losses(path,*val) for path in teacher_paths])}
    for method,ps in paths.items():
        for name,data in [('val',val),('test',test)]:
            curves[method+'_'+name]=np.array([losses(path,*data) for path in ps])
    # Diagnose one-step imitation on held-out beam states, without fitting them.
    beam_grads=np.array(beam_grads)
    bx=paths['beam'][:,:-1].reshape(-1,D); truth=beam_grads.reshape(-1,D)
    bv=velocities(beam_grads,a.momentum).reshape(-1,D)
    # Same seed: the policy sees exactly the gradient the SGD reference uses.
    diag_rng=np.random.default_rng(np.random.SeedSequence([a.seed,7]))
    sgd_guess=minibatch_gradient(bx,train,a.batch_size,np.random.default_rng(np.random.SeedSequence([a.seed,7])))
    start=time.perf_counter(); pred=policy(bx,bv,diag_rng)
    timers['one_step_diagnostic_s']=time.perf_counter()-start
    versions={}
    for pkg in ('numpy','tabpfn','torch','scikit-learn','matplotlib'):
        try: versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: pass
    metrics={'config':config,'versions':versions,'timings':timers,
             'budget':{'context_gradient_evaluations':teacher_cost,
                       'heldout_beam_gradient_evaluations':held_cost,
                       'heldout_sgd_gradient_evaluations':a.meta_test*a.steps,
                       'learned_rollout_gradient_evaluations':a.meta_test*a.steps*grad_features,
                       'context_rows':len(CX),'context_features':CX.shape[1],
                       'learned_scalar_predictions':a.meta_test*a.steps*D},
             'one_step_gradient_mse':float(np.mean((pred-truth)**2)),
             'zero_gradient_mse':float(np.mean(truth**2)),
             'one_step_mean_cosine':mean_cosine(pred,truth),
             # Reference: a fresh minibatch gradient (what SGD would use) vs the beam's choice.
             'sgd_one_step_gradient_mse':float(np.mean((sgd_guess-truth)**2)),
             'sgd_one_step_mean_cosine':mean_cosine(sgd_guess,truth),
             'final':{}}
    for method in paths:
        v=curves[method+'_test'][:,-1]
        metrics['final'][method]={'test_median':float(np.median(v)),
            'test_mean':float(np.mean(v)),'nonfinite_runs':int(np.sum(~np.isfinite(v))),
            'test_per_run':v.tolist()}
    metrics['meta_beats_sgd_fraction']=float(np.mean(curves['meta_test'][:,-1]<curves['sgd_test'][:,-1]))
    # Strict JSON: record nonfinite quantities as strings, not invalid Infinity tokens.
    def clean(obj):
        if isinstance(obj,float) and not np.isfinite(obj): return str(obj)
        if isinstance(obj,dict): return {k:clean(v) for k,v in obj.items()}
        if isinstance(obj,list): return [clean(v) for v in obj]
        return obj
    (a.out/'metrics.json').write_text(json.dumps(clean(metrics),indent=2,allow_nan=False))
    np.savez_compressed(a.out/'trajectories.npz',**paths,**curves,
                        one_step_truth=truth,one_step_pred=pred)
    plot_results(a.out,curves,a)
    print(json.dumps(clean(metrics['final']),indent=2))
    print(f'Wrote plots, raw trajectories, contexts, and metrics to {a.out.resolve()}')


if __name__=='__main__':
    main()

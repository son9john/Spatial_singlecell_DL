"""Null checks under a known original-graph sign-symmetry model."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.stats import norm

from experiment import ExperimentGraph, FIXED, opposition, ratio, spatial_grid, two_fields

N_PAIRS, N_DRAWS = 240, 199


def p_mc(v):
    return (1+np.count_nonzero(np.abs(v[1:])>=abs(v[0])-1e-13))/len(v)


def calibration(out,jx,seed_module):
    rng = np.random.default_rng([20261123,17])
    graph, xx, yy = spatial_grid(12)
    # Break exact repeated eigenvalues while retaining a regular tissue layout.
    up = sparse.triu(graph,k=1,format="coo")
    w = rng.uniform(.9,1.1,len(up.data))
    graph = sparse.csr_matrix((np.tile(w,2),(np.r_[up.row,up.col],np.r_[up.col,up.row])),shape=graph.shape)
    heat = jx.smoother(graph)
    left, right = (xx<6).astype(float),(xx>=6).astype(float)
    panel = two_fields(np.column_stack([left]*16+[right]*16),heat,rng)
    model = ExperimentGraph(graph,panel,jx,seed_module)
    mu,U = eigh(model.lap.toarray())
    assert np.min(np.diff(mu)) > 1e-10
    mu = np.maximum(mu,0.)
    transfer = 1-model.dt*mu
    filters = {"J1":mu*transfer**2,
               "J4":mu*transfer**8,
               "wavelet_J":(transfer-transfer**16)**2,
               "resolvent_J":mu/(1+4*model.dt*mu)**2,
               "phase_J":4*np.sin(np.pi*model.dt*mu)**2}
    phase_model = ExperimentGraph(graph,panel,jx,seed_module,phase_only=True)
    gradient_basis = model.t.gradient(U)*transfer
    # A frozen edge mask is a fixed operator, but generally not diagonal in
    # the original graph basis. Test the correct linear numerator separately.
    diagnostic = []
    for key,w in model.fixed_weights.items():
        projected = gradient_basis.T @ ((model.w*w)[:,None]*gradient_basis)
        offdiag = projected-np.diag(np.diag(projected))
        diagnostic.append(dict(method=key, seeds=model.counts[key],
                               relative_off_diagonal=np.linalg.norm(offdiag)/np.linalg.norm(projected)))
    pd.DataFrame(diagnostic).to_csv(out/"operator_diagnostics.csv",index=False)
    rows = []
    for family,strength in [("white",0),("smooth",1),("very_smooth",2)]:
        amplitude = (1+mu/.6)**(-strength)
        for replicate in range(N_PAIRS):
            bx, by = rng.normal(size=(2,len(mu))) * amplitude
            bx[0] = by[0] = 0.
            bx *= np.sqrt(len(bx))/np.linalg.norm(bx)
            by *= np.sqrt(len(by))/np.linalg.norm(by)
            signs = np.vstack((np.ones_like(by),rng.choice((-1,1),size=(N_DRAWS,len(by)))))
            null_y = signs*by
            values, pvals = {}, {}
            for key,f in filters.items():
                # This filtered cross-energy has an invariant denominator in this basis.
                ff = np.sqrt(f)
                evaluated = opposition((ff*bx)[:,None],(null_y*ff).T)
                values[key] = float(evaluated[0])
                pvals[key,"conditional_MC"] = p_mc(evaluated)
                a = f*bx*by
                pvals[key,"normal_numerator"] = 2*norm.sf(abs(a.sum())/np.linalg.norm(a))
            x1 = U @ (bx*transfer)
            y1 = U @ (null_y*transfer).T
            dx,dy = model.t.gradient(x1)[:,None],model.t.gradient(y1)
            cross = -2*model.w[:,None]*dx*dy
            energy = model.w[:,None]*(dx*dx+dy*dy)
            local = model.local_from_terms(cross,energy)
            seed_scores = ratio(model.t.incidence@local,model.t.degree[:,None])

            for key in FIXED:
                w = model.fixed_weights[key]
                evaluated = opposition(dx,dy,(model.w*w)[:,None])
                values[key] = float(evaluated[0])
                pvals[key,"conditional_MC"] = p_mc(evaluated)
                a = by * (gradient_basis.T @ (model.w*w*dx[:,0]))
                pvals[key,"normal_numerator"] = 2*norm.sf(abs(a.sum())/np.linalg.norm(a))

            adaptive = {"elbow_J":np.empty(len(signs)), "cluster_J":np.empty(len(signs))}
            frozen = {}
            for draw in range(len(signs)):
                seeds = seed_module.first_elbow(seed_scores[:,draw])
                for key, selection in [("elbow_J",seeds),("cluster_J",model.clustered(seeds))]:
                    w = model.weights(selection)
                    v = ratio(w@cross[:,draw],w@energy[:,draw])
                    adaptive[key][draw] = v
                    if draw==0:
                        frozen[key] = opposition(dx,dy,(model.w*w)[:,None])
            for key, evaluated in adaptive.items():
                values[key] = float(evaluated[0])
                pvals[key,"reselected_MC"] = p_mc(evaluated)
                pvals[key,"freeze_same_pair_MC"] = p_mc(frozen[key])

            za = U @ np.column_stack((bx,null_y.T))
            for _ in range(4):
                za = model.anisotropic @ za
            e = model.t.gradient(za)
            evaluated = opposition(e[:,0,None], e[:,1:],model.w[:,None])
            values["anisotropic_J"] = float(evaluated[0])
            pvals["anisotropic_J","conditional_MC"] = p_mc(evaluated)

            assert np.isclose(values["J1"],opposition(dx[:,0],dy[:,0],model.w),atol=1e-12)
            if replicate==0:
                # Check the fixed spectral shortcuts against the actual scoring routes.
                x = U @ np.column_stack((bx,by))
                scores,_ = model.score(model.prepare(x),0,1)
                scores.update(phase_model.score(phase_model.prepare(x),0,1)[0])
                for key,val in values.items():
                    np.testing.assert_allclose(val,scores[key],atol=1e-10,rtol=0)
            rows.extend(dict(family=family,replicate=replicate,method=key,inference=inf,p=p,score=values[key],
                             fixed_signed_seeds=model.counts["fixed_signed_J"],
                             fixed_positive_seeds=model.counts["fixed_positive_J"])
                        for (key,inf),p in pvals.items())
        print(f"Known null {family}: {N_PAIRS} pairs",flush=True)
    pd.DataFrame(rows).to_csv(out/"known_null.csv.gz",index=False)

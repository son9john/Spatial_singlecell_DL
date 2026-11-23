import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crc_mc.py')).read(); src=src[:src.index('B = 4000')]
exec(src)
rng=np.random.default_rng(3); i=rng.integers(0,300,3000); j=rng.integers(0,300,3000); k=i!=j; i,j=i[k],j[k]
B=2000; E=rng.choice(np.array([-1.,1.],dtype=np.float32),size=(B,n))
for s in (1,2,4,8,16):
    h=mu*(1-mu/(2*deg.max()))**(2*s); A=h[:,None]*xh[:,i]*xh[:,j]
    eff=(A*A).sum(0)**2/(A**4).sum(0); obs=A.sum(0); v=(A*A).sum(0)
    pn=2*stats.norm.sf(np.abs(obs)/np.sqrt(v)); nul=E@A.astype(np.float32); pm=(1+(np.abs(nul)>=np.abs(obs)-1e-9).sum(0))/(B+1)
    print(f's={s:2d} median eff modes {np.median(eff):6.1f}  normal p<.05 {100*(pn<.05).mean():5.1f}%  MC p<.05 {100*(pm<.05).mean():5.1f}%')

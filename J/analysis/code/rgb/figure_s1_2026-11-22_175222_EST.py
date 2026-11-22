"""Observed tissue benchmark, including the archived contact-rate orientation audit."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAMP='2026-11-22_104348_EST'
METHODS={'J_region':'Regional J','smooth_pearson':'Matched-scale Pearson','hotspot_Z':'Hotspot Z',
         'giotto_spatial_cor':'Giotto network','lee_l':'Lee’s L','liana_morans_R':'LIANA+ Moran',
         'spatialdm_R':'SpatialDM R','meringue_moran':'MERINGUE','pearson':'Pearson','spearman':'Spearman',
         'CrossExpression_r':'CrossExpression r','CrossExpression_p':'CrossExpression raw P','CrossExpression_q':'CrossExpression adj. P'}
HIST={'Pearson':'Pearson','Hotspot_Cov':'Hotspot covariance',
      'SpatialDM_Moran':'Bivariate Moran','Lee_L':'Lee’s L','LIANA_Cosine':'LIANA+ cosine'}

def build(results, figures):
    data=pd.read_csv(next(results.glob('metrics_by_unit_*.csv')))
    data=data[data.task.eq('touch_apart')]
    extra=pd.read_csv(next(results.glob('seqfish_cross_expression_metrics*.csv'))).rename(columns={'AUROC':'auroc'}).assign(design='C1')
    data=pd.concat([data,extra],ignore_index=True)
    historical=list(HIST.items())
    fig,axs=plt.subplots(1,3,figsize=(11.0,4.75),layout='constrained')
    plt.rcParams['pdf.fonttype']=42
    for i,(key,ax,title) in enumerate(zip(['B1','C1','C2'],axs,['Cortical layers','Embryo contact z','Embryo contact rate'])):
        sub=data[data.design.eq(key)]
        entries=[(k,v,False) for k,v in METHODS.items() if k in set(sub.method)]
        if i==2:
            entries=[('J_region','Regional J',False),('smooth_pearson','Matched Pearson',False),('hotspot_Z','Hotspot Z',False)]+[(k,v,True) for k,v in historical]
        for row,(k,name,rev) in enumerate(entries):
            vals=sub.loc[sub.method.eq(k),'auroc'].to_numpy()
            col='#00BA38' if k=='J_region' else '#F8766D' if k in ('J1','J_unnorm') else '#619CFF'
            if rev:
                ax.plot([1-vals.mean(),vals.mean()],[row,row],color='#9caab8',lw=1)
                ax.scatter(1-vals.mean(),row,s=22,facecolor='white',edgecolor=col,zorder=3)
            else:
                ax.scatter(vals,row+np.linspace(-.12,.12,len(vals)),s=13,alpha=.55,facecolor='white',edgecolor=col,lw=.65)
            ax.scatter(vals.mean(),row,marker='D',s=24,color=col,zorder=4)
        ax.set(yticks=np.arange(len(entries)),yticklabels=[e[1] for e in entries],xlim=(0,1),ylim=(len(entries)-.45,-.55),
               xlabel='AUROC',xticks=[0,.25,.5,.75,1],xticklabels=['0','','0.5','','1'])
        ax.tick_params(labelsize=8,length=2);ax.axvline(.5,color='#9cabb5',ls='--',lw=.7)
        ax.set_title(f'{"ABC"[i]}   {title}',fontsize=11,loc='left',pad=11,fontweight='bold')
        ax.spines[['top','left','right']].set_visible(False);ax.yaxis.grid(color='#e2e7ea',lw=.6);ax.set_axisbelow(True)
    for fmt in ('pdf','png'):
        fig.savefig(figures/f'figure_s1_observed_pair_accuracy_{STAMP}.{fmt}',dpi=320,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('figures',type=Path);a=p.parse_args()
    build(a.results,a.figures)

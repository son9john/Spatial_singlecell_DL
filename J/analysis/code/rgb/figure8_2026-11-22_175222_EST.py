"""Observed gene-pair AUROC against annotation-derived proximity in measured tissue."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

STAMP = '2026-11-22_135104_EST'
REGION = '#00BA38'
CONTROL = '#619CFF'


def draw(data, destination):
    plots = [
        ('B1', 'Cortical layers', '12 sections · 3 donors'),
        ('C1', 'Cell-type contact', '3 embryos · permutation z'),
        ('C2', 'Cell-type contact', '3 embryos · contact rate'),
    ]
    row_titles = ['Regional J', 'Matched-scale Pearson', 'Hotspot Z',
                  'Giotto spatial correlation', 'Bivariate Moran (MERINGUE)', 'Lee’s L', 'Pearson']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.53), sharey=True)
    fig.subplots_adjust(left=.22, right=.955, top=.78, bottom=.23, wspace=.39)
    for ax,(panel,title,sub),letter in zip(axes,plots,'ABC'):
        subset = data.query("design==@panel and task=='touch_apart'")
        keys = ['J_region','smooth_pearson','hotspot_Z','giotto_spatial_cor',
                'meringue_moran','lee_l','pearson']
        for iy, method in enumerate(keys):
            per_unit = subset[subset.method == method].sort_values('unit').auroc.values
            color = REGION if method == 'J_region' else CONTROL
            jitter = np.linspace(-.20,.20,len(per_unit))
            ax.scatter(per_unit, iy+jitter, color='white', edgecolor=color, alpha=.85,
                       linewidth=.68, s=10, zorder=3)
            ax.scatter(per_unit.mean(), iy, color=color, edgecolor='white', marker='D',
                       linewidth=.38, s=30, zorder=4)
            ax.text(1.025, iy, f'{per_unit.mean():.4f}', transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=7, color=color)
        ax.set(xlim=(0,1), xticks=(0,.25,.5,.75,1), xticklabels=['0','','0.5','','1'],
               yticks=np.arange(len(keys)), yticklabels=row_titles, ylim=(len(keys)-.35,-.85))
        ax.axvline(.5, color='#89929B', lw=.7, ls=(0,(3,3)))
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E6EAED',lw=.65)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#B3BEC5')
        ax.tick_params(axis='y',which='both',length=0,pad=5)
        ax.tick_params(axis='x',length=2.5,color='#A6B0B8',labelsize=7)
        ax.text(0,1.24,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
        ax.text(.13,1.24,title,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top')
        ax.text(0,1.13,sub,transform=ax.transAxes,fontsize=7.2,color='#475A69',va='top')
        ax.text(1.025,-.74,'Mean',transform=ax.get_yaxis_transform(),ha='left',va='center',
                color='#475A69',fontsize=6.5)
        ax.set_xlabel('AUROC',fontsize=7.4,labelpad=3)
    legend = [
        Line2D([],[],ls='',marker='o',mfc='white',mec=CONTROL,ms=4,label='Section or embryo'),
        Line2D([],[],ls='',marker='D',mfc=CONTROL,mec='white',mew=.3,ms=5.5,label='Mean'),
        Line2D([],[],ls='--',color='#89929B',lw=.7,label='Chance'),
    ]
    fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.56,.006),
               handlelength=1.4,columnspacing=2.6,fontsize=7.5)
    destination.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(destination/f'figure8_observed_accuracy_{STAMP}.{ext}',dpi=350,
                    facecolor='white',bbox_inches='tight',pad_inches=.075)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    draw(pd.read_csv(args.metrics),args.output)

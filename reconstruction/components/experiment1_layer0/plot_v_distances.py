"""Plot target/incorrect-token layer-0 V distance distributions from evaluate_generic_vocabulary.py output."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def panel(root,backend,output,label):
    
    summary=json.loads((root/'summary.json').read_text());stats=summary['tables'][backend]
    with np.load(root/(backend+'_distribution.npz')) as d:edges=d['edges'];counts=d['other_counts'];target=d['target_distances']
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.5,2.9));right=ax.twinx()
    right.stairs(counts/counts.sum()/np.diff(edges),edges,fill=True,color='#2479b5',alpha=.4)
    if np.ptp(target)==0:ax.axvline(target[0],color='#ed7d20',linewidth=1.8);ax.set_ylim(0,1)
    else:ax.hist(target,bins=40,density=True,color='#ed7d20',alpha=.5);ax.set_ylim(bottom=0)
    mu,sigma=stats['other_mean'],stats['other_std']
    if sigma>0:
        x=np.linspace(max(0,mu-5*sigma),mu+5*sigma,500)
        right.plot(x,np.exp(-.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)),'--',color='#2479b5',linewidth=1.5)
    ax.set_zorder(right.get_zorder()+1);ax.patch.set_visible(False)
    xmax=stats['other_max']*1.45;ax.set_xlim(-xmax*.015,xmax);right.set_ylim(bottom=0)
    ax.set_title(f'{label} / {backend.upper()}',fontsize=10);ax.set_xlabel('RMSE',fontsize=9)
    ax.set_ylabel('Target density',color='#d96712',fontsize=8);right.set_ylabel('Other density',color='#2479b5',fontsize=8)
    ax.tick_params(labelsize=7);ax.tick_params(axis='y',colors='#d96712');right.tick_params(labelsize=7,axis='y',colors='#2479b5');ax.grid(alpha=.2,linestyle=':')
    for kind,y,color in [('target',.97,'#ffe5cf'),('other',.66,'#daebf5')]:
        ax.text(.98,y,f"{kind.title()}:\n"+r'$\mu$: '+f"{stats[kind+'_mean']:.4g}\n"+r'$\sigma$: '+f"{stats[kind+'_std']:.4g}",transform=ax.transAxes,ha='right',va='top',fontsize=7,bbox=dict(boxstyle='round,pad=.3',facecolor=color,edgecolor='#bbbbbb'))
    fig.tight_layout();fig.savefig(output.with_suffix('.pdf'));fig.savefig(output.with_suffix('.png'),dpi=220);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Output directory of evaluate_generic_vocabulary.py')
    p.add_argument('--output',type=Path,help='Figure directory (default: INPUT)')
    p.add_argument('--label',help='Panel title (default: model name in summary.json)')
    args=p.parse_args()
    summary=json.loads((args.input/'summary.json').read_text());out=args.output or args.input;out.mkdir(parents=True,exist_ok=True)
    for backend in summary['tables']:
        target=out/f"{summary['model']}_{backend}";panel(args.input,backend,target,args.label or summary['model'])
        print('FIGURE',target.with_suffix('.pdf'),flush=True)

if __name__=='__main__':main()

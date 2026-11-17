
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
L0=ROOT/'audio_recovery/run01/token_duration_eval'
L4=ROOT/'audio_chat_layer4/test_token_duration_eval/run01'
OUT=ROOT.parents[1]/'sp2026_template/figures/audio'
ids=json.loads((ROOT/'audio_recovery/run01/fixed_case_ids.json').read_text(encoding='utf8'))
assert len(ids)==6
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,'pdf.fonttype':42})
fig=plt.figure(figsize=(7.1,8.2))
outer=fig.add_gridspec(6,2,left=.105,right=.985,bottom=.045,top=.925,wspace=.24,hspace=.48)
fig.suptitle('Gemma 4 E2B · NPU',fontsize=12,fontweight='bold',y=.989)
fig.text(.30,.953,'Layer 0 · V',ha='center',fontsize=11,fontweight='bold')
fig.text(.79,.953,'Layer 4 · K/V',ha='center',fontsize=11,fontweight='bold')
provenance=[]
for row,sid in enumerate(ids):
    with np.load(L0/'examples'/f'{sid}.npz') as z:
        original=z['original'];p0=z['recovered'];t0=float(z['original_duration']);t1=float(z['recovered_duration'])
    with np.load(L4/'examples'/f'{sid}.npz') as z:
        assert np.array_equal(original,z['original'])
        assert t0==float(z['original_duration']) and t1==float(z['recovered_duration'])
        p4=z['recovered']
    assert p0.shape==p4.shape
    vmax=float(np.quantile(original,.99));end=max(t0,t1)
    for col,pred in enumerate([p0,p4]):
        grid=outer[row,col].subgridspec(2,1,hspace=.32)
        for j,(label,arr,duration) in enumerate([('Original',original,t0),('Recovered',pred,t1)]):
            ax=fig.add_subplot(grid[j])
            ax.imshow(arr,origin='lower',aspect='auto',extent=[0,duration,0,8],vmin=0,vmax=vmax,cmap='magma',interpolation='nearest',rasterized=True)
            ax.set_xlim(0,end);ax.set_yticks([0,8]);ax.tick_params(labelsize=6,pad=1,length=2)
            ax.set_ylabel(label+'\n(kHz)',fontsize=6.5,labelpad=3)
            if j==0:
                ax.tick_params(labelbottom=False)
                ax.set_title(f'({chr(97+row)}) {sid}',fontsize=8,pad=3)
            elif row==5:ax.set_xlabel('Time (s)',fontsize=8,labelpad=1)
    provenance.append({'id':sid,'original_duration':t0,'estimated_duration_both_layers':t1,'shared_vmin':0,'shared_vmax':vmax,'layer0_file':str(L0/'examples'/f'{sid}.npz'),'layer4_file':str(L4/'examples'/f'{sid}.npz')})
fig.savefig(OUT/'spectrogram_examples.pdf',dpi=240)
fig.savefig(OUT/'spectrogram_examples.png',dpi=160)
plt.close(fig)
(OUT/'spectrogram_layers_provenance.json').write_text(json.dumps({'selection':'Same six preselected cases as the original appendix; no quality reselection','layout':'Layer0 left; layer4 right; original above recovered in every cell; shared axes and colour limits across all four plots of each case','duration':'Both recoveries use valid-token-count estimates; neither uses original duration','cases':provenance},indent=2),encoding='utf8')
print(OUT/'spectrogram_examples.pdf')

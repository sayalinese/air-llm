from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from paper_plot_style import FIG_DIR, COLORS, save

ROOT = FIG_DIR.parent / '模型'
RUNS = {
    '5k': ROOT / 'lstm_temp_filt_5000_s42' / 'history.csv',
    '20k': ROOT / 'lstm_temp_filt_20000_s42' / 'history.csv',
}

def load_hist():
    out={}
    for name,path in RUNS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        out[name]=pd.read_csv(path)
    return out

def fig_accuracy_loss(hist):
    fig, ax = plt.subplots(1,2,figsize=(7.2,3.0))
    for name,df in hist.items():
        ax[0].plot(df.epoch, df.train_loss, marker='o', lw=1.8, ms=3.5, label=name, color=COLORS[name])
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss'); ax[0].legend(frameon=False)
    for name,df in hist.items():
        ax[1].plot(df.epoch, df.acc, marker='o', lw=1.8, ms=3.5, label=name, color=COLORS[name])
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Validation accuracy'); ax[1].legend(frameon=False)
    fig.tight_layout(w_pad=2.4); save(fig,'fig1_loss_and_accuracy_iterations')

def fig_auc_f1_pr(hist):
    fig, axes = plt.subplots(1,3,figsize=(10.0,3.0))
    specs=[('auc','Validation AUC'),('f1','Validation F1'),('pr_auc','Validation PR-AUC')]
    for k,(col,label) in enumerate(specs):
        for name,df in hist.items():
            axes[k].plot(df.epoch, df[col], marker='o', lw=1.8, ms=3.5, label=name, color=COLORS[name])
        axes[k].set_xlabel('Epoch'); axes[k].set_ylabel(label)
        axes[k].set_ylim(0,1); axes[k].legend(frameon=False)
    fig.tight_layout(w_pad=2.0); save(fig,'fig2_auc_f1_pr_auc_iterations')

def fig_model_bars():
    df=pd.read_csv(FIG_DIR/'comparison_results.csv')
    scales=['5k','20k','100k']; modes=['lstm_only','gate','crossattn','catgate']
    labels=['LSTM-only','Gate','CrossAttn','CatGate']
    x=np.arange(len(scales)); width=0.18
    fig,ax=plt.subplots(figsize=(6.6,3.4))
    for i,(mode,label) in enumerate(zip(modes,labels)):
        vals=[float(df[(df['scale']==s)&(df['mode']==mode)]['test_auc'].iloc[0]) for s in scales]
        bars=ax.bar(x+(i-1.5)*width, vals, width, label=label)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.0015,f'{v:.4f}',ha='center',va='bottom',fontsize=7,rotation=90)
    ax.set_xticks(x,scales); ax.set_xlabel('Training scale'); ax.set_ylabel('Test ROC-AUC'); ax.set_ylim(0.58,0.66); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); save(fig,'fig3_fusion_model_comparison')

def fig_scale_lines():
    df=pd.read_csv(FIG_DIR/'comparison_results.csv')
    fig,ax=plt.subplots(figsize=(5.6,3.2))
    for mode,label in [('lstm_only','LSTM-only'),('gate','Gate'),('crossattn','CrossAttn'),('catgate','CatGate')]:
        z=df[df['mode']==mode].set_index('scale').loc[['5k','20k','100k']]
        ax.plot(['5k','20k','100k'],z.test_auc,marker='o',lw=1.8,label=label)
    ax.set_xlabel('Training scale'); ax.set_ylabel('Test ROC-AUC'); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); save(fig,'fig4_fusion_scaling_trend')

if __name__=='__main__':
    h=load_hist(); fig_accuracy_loss(h); fig_auc_f1_pr(h); fig_model_bars(); fig_scale_lines(); print('figures saved to',FIG_DIR)

"""Plot saved source-internal support/query curves without new model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output-stem',type=Path,required=True)
    args=parser.parse_args()
    for suffix in ['.png','.svg']:
        if args.output_stem.with_suffix(suffix).exists():raise FileExistsError(args.output_stem.with_suffix(suffix))
    result=json.loads(args.result.read_text())
    if result['status']!='complete' or result['formal_result'] is not False:raise ValueError('expected completed diagnostic')
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams['axes.unicode_minus']=False
    updates=result['config']['curve_updates']
    arms=[('A_original','A：原始 SPT','#377eb8'),('B_fomaml','B：FOMAML 候选','#e68613'),('C_pooled_bc','C：普通预训练','#359458')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey='row')
    for column,fold in enumerate(result['fold_results']):
        for row,(role,label) in enumerate([('meta_support','support 拟合'),('meta_query','query 评价')]):
            ax=axes[row,column]
            for arm,name,color in arms:
                values=np.array([[s['arms'][arm]['curve'][str(u)][role]['mse'] for u in updates] for s in fold['seed_results']])
                ax.fill_between(updates,values.min(0),values.max(0),color=color,alpha=.12)
                ax.plot(updates,values.mean(0),marker='o',markersize=4,color=color,label=name)
            ax.set_title(f"source 内部探针 task {fold['probe']} · {label}",fontproperties=font)
            ax.set_ylabel('七维动作 MSE（越低越好）',fontproperties=font)
            ax.grid(alpha=.25);ax.set_xticks(updates);ax.set_ylim(bottom=0)
            if row==1:ax.set_xlabel('探针 Policy 适应 updates',fontproperties=font)
    axes[0,0].legend(prop=font,fontsize=9)
    fig.suptitle('SPT/SPI 离线可行性诊断：三 seed 均值与范围',fontproperties=font,fontsize=16)
    fig.text(.5,.012,'阴影为三 seed 的最小—最大范围；已暴露的 source 任务；无环境交互，不是正式迁移结果。',
             fontproperties=font,ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ['.png','.svg']:fig.savefig(args.output_stem.with_suffix(suffix),dpi=160)
    print(json.dumps({'png':str(args.output_stem.with_suffix('.png')),'svg':str(args.output_stem.with_suffix('.svg')),'new_model_calls':0},ensure_ascii=False))


if __name__=='__main__':main()

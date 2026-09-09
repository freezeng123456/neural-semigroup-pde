#!/usr/bin/env python3
"""Build a complete interpretation from frozen first-screen and extension artifacts."""
import argparse,csv,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gm(values):
    values=list(values);return math.exp(statistics.mean(math.log(v) for v in values))
def primary(s,nu=.02):
    rows=[r['mse'] for r in s['rows'] if r['split']=='test' and r['nu']==nu and r['horizon'] in (1.2,2.4) and r['lag'] in (.06,.12)]
    assert len(rows)==4
    return gm(rows)
def gate(rs):return gm(rs)<=.9 and sum(x<=.9 for x in rs)>=2 and max(rs)<=1.05

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
    r=b/'recovered';base=r/'semigroup-discovery-20260909-r1/run';ext=r/'semigroup-clock-control-20260909-r1/run'
    orig=[json.loads(p.read_text()) for p in sorted((base/'cells').glob('*/summary.json'))]
    extra=[json.loads(p.read_text()) for p in sorted((ext/'cells').glob('*/summary.json'))]
    assert len(orig)==33 and len(extra)==3
    hf=json.loads((r/'semigroup-high-frequency-20260909-r1/run/summary.json').read_text())
    assert len(hf['rows'])==108
    summaries=orig+extra;out=b/'analysis-final';out.mkdir(exist_ok=True)
    groups=json.loads((base/'analysis/comparison.json').read_text())['groups']
    clock=[]
    for e in extra:
        s=next(d for d in orig if d['config']['model']=='clock' and d['config']['seed']==e['config']['seed'])
        clock.append(dict(seed=e['config']['seed'],clock_mse=primary(s),clock_query_mse=primary(e),ratio=primary(s)/primary(e)))
    high=[]
    for noise in (0.,.01):
        for nu in (.005,.02,.08):
            records=[]
            for seed in (42,123,2026):
                vals=[gm(q['mse'] for q in hf['rows'] if q['noise']==noise and q['nu']==nu and q['seed']==seed and q['model']==model and q['horizon'] in (1.2,2.4)) for model in ('autonomous','query')]
                records.append(dict(seed=seed,autonomous_mse=vals[0],query_mse=vals[1],ratio=vals[0]/vals[1]))
            high.append(dict(noise=noise,nu=nu,autonomous_mse=gm(d['autonomous_mse'] for d in records),query_mse=gm(d['query_mse'] for d in records),
                ratio=gm(d['ratio'] for d in records),passed=gate([d['ratio'] for d in records]),seeds=records))
    model_rows=[]
    for d in summaries:
        c=d['config'];model_rows.append(dict(track=c['track'],seed=c['seed'],n=c['n'],noise=c['noise'],model=c['model'],
            primary_mse=primary(d),reaction_grid_mse=d['diagnostics']['reaction_grid_mse'],best_update=d['best_update'],
            training_seconds=d['training_seconds'],elapsed_seconds=d['elapsed_seconds'],parameters=c['parameters'],effective_parameters=c['effective_parameters']))
    with (out/'all_models.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(model_rows[0]));w.writeheader();w.writerows(model_rows)
    analysis=dict(first_screen_groups=groups,clock_information_control=clock,clock_information_control_passed=gate([d['ratio'] for d in clock]),high_frequency=high,
        training_cells=36,optimizer_updates=72000,training_seconds=sum(d['training_seconds'] for d in summaries),
        selected_at_final_update=sum(d['best_update']==2000 for d in summaries),recorded_test_endpoints=sum(len(d['rows']) for d in summaries)+len(hf['rows']),
        max_observed_abs_state=max(q['max_state'] for d in summaries for q in d['rows']),
        max_fraction_abs_gt_1_5=max(q['fraction_abs_gt_1_5'] for d in summaries for q in d['rows']),
        high_frequency_diffusion_reaction_initial_norm=hf['diffusion_reaction_initial_norm'])
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    lines=['# SCNet 半群机制探索：36个训练实验与高频迁移核查','',
        '本轮完成33个预注册训练实验、3个看过首批结果后追加的信息匹配对照，以及12份冻结权重的高频初始条件评估。全部在SCNet RTX 3080上完成；原始文件已回收，训练权重与主端点已独立重放。所有比较仍属于三种子的探索性证据。','',
        '## 结论','',
        '1. **在128个样本和固定2000更新预算下，自主模型出现约15%的长期预测优势候选。** 16样本组不支持稳定优势；因此不能把它概括成普遍的小样本或抗噪优势。',
        '2. **已知扩散算子可以替换并复用原权重，但收益有分布边界。** 平滑初始条件下，更换扩散系数后A/B改善保持；高频、扩散更强的状态下，多数设置未通过逐种子门槛。',
        '3. **显含时间的外力问题首先需要正确的时钟信息。** 时钟模型相对两个缺少物理时钟的模型降低约90%主MSE；同样提供物理时钟后，自主结构的额外改善为约13.6%，但未通过预设逐种子门槛。大幅收益主要支持正确的状态表示，尚不能归因于半群结构本身。',
        '4. **本轮不证明充分收敛后的精度优势。** 36个模型中有34个按验证集选中了预算终点的权重；多数学习曲线仍在下降。模型选择没有使用测试集，但固定预算的优化行为仍可能影响对照。','',
        '## 问题与对照','',
        '空间网格N=64，周期区间长度2π。自主问题为 u_t=νΔ_hu+u−u³+0.2sin(3u)。所有模型共享已知扩散算子与负三次项，只从状态快照学习其余反应规律。外力问题额外加入0.3sin(2πt)。','',
        'A（autonomous）只用状态；B（query）允许反应规律依赖请求时长。clock使用状态与真实时钟相位sin(2πt)、cos(2πt)，积分和重启都累计物理时间；追加的clock_query获得同样的时钟相位，并额外使用请求时长。已知周期是显式先验，未向模型提供真实外力值或反应项标签。','',
        '每个模型名义参数193；A、B、clock、clock_query的有效参数分别97、129、161、193，不能声称有效容量完全相同。每个映射4个分裂子步，每步2次中点反应项调用；同一设置的成对模型匹配样本、噪声、初始化、优化更新及调用预算。','',
        '训练时长为0.03/0.07/0.13/0.23；输入与输出的测量噪声标准差为0或0.01。Adam学习率0.003，2000更新。按独立验证集t=0.6/1.2、步长0.12的误差选择权重。主测试指标为t=1.2/2.4、步长0.06/0.12的4个端点MSE几何平均，再对3个seed取几何平均。额外0.24步长外推和压力集端点完整保留。','',
        '候选门槛：汇总A/B≤0.90、至少2个seed≤0.90、没有seed>1.05。该规则不是统计显著性检验；追加对照和高频评估使用了已查看的首批结果，明确标记为事后探索。','',
        '## 不规则观测：训练扩散系数ν=0.02','',
        '| 样本数 | 噪声标准差 | A主MSE | B主MSE | A/B | 三seed A/B（42/123/2026） | 候选门槛 |',
        '|---:|---:|---:|---:|---:|---|---|']
    for n in (16,128):
        for noise in (0.,.01):
            g=next(g for g in groups if g['track']=='irregular' and g['n']==n and g['noise']==noise and g['nu']==.02)
            vals=[gm(primary(d) for d in orig if d['config']['track']=='irregular' and d['config']['n']==n and d['config']['noise']==noise and d['config']['model']==m) for m in ('autonomous','query')]
            seeds='/'.join(f"{g['ratios_by_seed'][str(s)]:.3f}" for s in (42,123,2026))
            lines.append(f"| {n} | {noise:g} | {vals[0]:.6g} | {vals[1]:.6g} | {g['pooled_ratio']:.4f} | {seeds} | {'通过' if g['exploratory_candidate'] else '未通过'} |")
    lines+=['','128样本的统一状态网格反应项误差A/B约为0.906（无噪声）和0.909（含噪声），均没有达到汇总10%改善。因此预测候选不能升级为全局反应项识别优势。噪声组的绝对误差有时反而更小，但只测试一个噪声水平、三个seed，不能据此认定加噪声是可靠改进方法。','',
        '## 冻结权重的扩散参数迁移','',
        '所有权重只在ν=0.02训练，评估时将双方相同的已知热算子换为ν=0.005或0.08，没有重训或重新选权重。这里迁移的是已知扩散系数，未知反应规律保持不变；不等于任意跨PDE迁移。','',
        '| n=128训练噪声 | 测试ν | A主MSE | B主MSE | A/B | 候选门槛 |','|---:|---:|---:|---:|---:|---|']
    for noise in (0.,.01):
        for nu in (.005,.08):
            g=next(g for g in groups if g['track']=='irregular' and g['n']==128 and g['noise']==noise and g['nu']==nu)
            vals=[gm(primary(d,nu) for d in orig if d['config']['track']=='irregular' and d['config']['n']==128 and d['config']['noise']==noise and d['config']['model']==m) for m in ('autonomous','query')]
            lines.append(f"| {noise:g} | {nu:g} | {vals[0]:.6g} | {vals[1]:.6g} | {g['pooled_ratio']:.4f} | {'通过' if g['exploratory_candidate'] else '未通过'} |")
    lines+=['','原测试初始状态较平滑。在ν=0.005/0.02/0.08，扩散项与反应项初始范数比中位数分别约0.0083/0.0334/0.1335。因此追加高频8–16的独立初始条件，检验扩散真正变强时的边界。','',
        '## 高频迁移：事后不训练评估','',
        '新增32个初始状态，seed=2026090931，频率8–16，振幅与均值范围保持原设定。12份原选中权重在3个扩散系数与3个物理终点上评估，共108个端点；下表为t=1.2/2.4、步长0.12的MSE几何平均。','',
        '| 训练噪声 | ν | 扩散/反应范数中位比 | A MSE | B MSE | A/B | 三seed A/B | 候选门槛 |','|---:|---:|---:|---:|---:|---:|---|---|']
    for h in high:
        seeds='/'.join(f"{s['ratio']:.3f}" for s in h['seeds'])
        lines.append(f"| {h['noise']:g} | {h['nu']:g} | {hf['diffusion_reaction_initial_norm'][str(h['nu'])]['median']:.3f} | {h['autonomous_mse']:.6g} | {h['query_mse']:.6g} | {h['ratio']:.4f} | {seeds} | {'通过' if h['passed'] else '未通过'} |")
    lines+=['','仅无噪声、ν=0.005的高频组通过门槛。更强扩散的无噪声组虽有较好的汇总比值，但逐seed改善不足；含噪声组出现明显反向seed，全部未通过。这限制了平滑状态下迁移优势的外推。高频绝对误差不能直接与平滑数据作收益比较，因为初始条件与参考轨迹不同。','',
        '## 周期外力与时钟信息','',
        '| 模型 | 物理时钟信息 | 请求时长条件 | 主MSE |','|---|---|---|---:|']
    for m,label in [('autonomous','autonomous'),('query','query'),('clock','clock'),('clock_query','clock_query（追加）')]:
        ds=[d for d in summaries if d['config']['track']=='forced' and d['config']['model']==m]
        lines.append(f"| {label} | {'有' if 'clock' in m else '无'} | {'有' if 'query' in m else '无'} | {gm(primary(d) for d in ds):.6g} |")
    lines+=['',f"clock/autonomous为0.0942，clock/query为0.0963。信息匹配后clock/clock_query为**{gm(d['ratio'] for d in clock):.4f}**，三个seed为"+'/'.join(f"{d['ratio']:.3f}" for d in clock)+'；只有一个seed达到10%改善，未通过候选门槛。','',
        '这支持把真实物理时钟纳入状态的用途：将显含时间问题写成增广状态(u,t)，推进时同步更新t。它没有证明本模型比所有非自主模型更好，也没有证明未知周期或任意外力可恢复。','',
        '## 数值与证据边界','',
        '- 参考解为固定空间网格FP64 RK4，dt≤0.002，并与dt≤0.001比较。高频检查最大相对L2差约2.40e-11。没有进行空间网格收敛证明。',
        '- 组合诊断直接0.24和两段0.12使用相同基本步长和相同总工作；自主/正确时钟模型在严格对齐路径上可仅剩舍入差。这是实现机制检查，不能单独证明预测精度或任意时间步长泛化。',
        '- |u|>1.5只是诊断阈值。全部普通评估中的观察越界比例为0；不是已证明不变区间。周期外力下原自由能不预期单调，未用它筛选模型。',
        '- 34/36个主选中权重位于2000更新预算终点，比较不代表充分收敛。每seed报告的是不同训练数据和初始化的配对结果；仅3个seed，不作强统计结论。',
        '- 原始训练只用状态快照；真实反应项只用于生成合成参考和事后诊断。追加实验未重新选择原权重。','',
        '## 完成与复现','',
        '| 检查 | 结果 |','|---|---|',
        '| 正式训练 | 36/36，72,000次优化更新 |',
        '| GPU冒烟 | 4个模型小测试；2个Slurm冒烟作业 |',
        '| 原训练权重 | 108份（初始/最终/验证最优），全部有限、最终权重确有变化 |',
        '| 训练模型主端点CPU重放 | 144个，通过 |',
        '| 高频端点CPU重放 | 108个，通过；所用权重哈希一致 |',
        '| 跨设备重放容差 | MSE绝对差≤max(1e-8, GPU MSE的2%) |',
        '| 原始回收文件 | 353个内容文件逐哈希核验，另含3份原清单；文件集合核验通过 |',
        '| 调度退出 | 所有相关Slurm条目COMPLETED、0:0 |',
        '| 单任务资源 | 1张RTX 3080、4 CPU，原批次最多4任务并发，新增训练3任务并发 |',
        '| 记录的测试端点 | 1620个（1512普通评估+108高频） |','',
        '训练源码：首批21bc488e479e47b74ae57e886ad9b7ceb6f2567c；时钟信息对照87f2f49f721c0098d6f104a5343894684c25beef；高频评估574ac25258494caafc16318bbd861b7c220e1f58。SCNet旧Git使用经SHA-256核验的源码归档，PyTorch为1.12.1。','',
        '作业：首批冒烟23778345、正式阵列23778411；新增冒烟23778831、对照阵列23778885；高频评估23778926。服务器Git发布因未登录而未完成，随后通过SCNet文件管理下载原始归档，在本地核验并发布完整结果。','',
        f"36个正式模型累计纯训练时间约{analysis['training_seconds']:.1f}秒；该值不包含参考生成、验证、测试、共享存储与排队开销，不能当作整轮墙钟时间。",'',
        '完整机器可读记录见all_models.csv与analysis.json；原端点、所有seed、权重、数据、日志、配置和回放证据均保留。','',
        '## 后续研究判断','',
        '优先保留两条有明确用途的方向：一是把可学习局部规律与可替换的已知算子分开，围绕真实参数变化与跨频率覆盖继续验证；二是依据问题的真实状态（含必要的时钟）构造可组合演化。当前应先做信息匹配、状态覆盖和独立确认，不能仅凭小组合缺陷或本轮15%左右的固定预算改善升级为普遍半群精度优势。']
    report='\n'.join(lines)+'\n'
    report=report.replace('## 周期外力与时钟信息', '图中圆点为各seed，菱形为几何平均；虚线表示0.9门槛。\n\n![高频迁移对照](high_frequency_transfer.png)\n\n## 周期外力与时钟信息')
    report=report.replace('## 数值与证据边界', '下图柱为三个seed的几何平均，黑点为各seed。\n\n![时钟信息对照](clock_information.png)\n\n## 数值与证据边界')
    (out/'REPORT.zh.md').write_text(report)
    # Information-matched clock effects are explicitly separated from blind controls.
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    names=['autonomous','query','clock','clock_query'];colors=['#777777','#a3a3a3','#2b728a','#ca6b43']
    for i,m in enumerate(names):
        vals=[primary(d) for d in summaries if d['config']['track']=='forced' and d['config']['model']==m]
        ax.bar(i,gm(vals),color=colors[i],alpha=.8,width=.65)
        ax.scatter([i]*3,vals,c='#222222',s=26,zorder=3)
    ax.set_xticks(range(4),['No clock\nautonomous','No clock\nquery duration','Clock\nautonomous','Clock + query\npost-screen control'])
    ax.set_yscale('log');ax.set_ylabel('Primary prediction MSE');ax.set_title('Physical clock information vs. additional autonomy effect')
    ax.grid(axis='y',alpha=.2);fig.savefig(out/'clock_information.png',dpi=180);fig.savefig(out/'clock_information.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,noise in zip(axes,(0.,.01)):
        hh=[h for h in high if h['noise']==noise]
        for i,h in enumerate(hh):
            ax.scatter([i]*3,[s['ratio'] for s in h['seeds']],c='#315d8a',s=35)
            ax.scatter(i,h['ratio'],marker='D',s=60,c='#ca6b43')
        ax.axhline(1,color='#666');ax.axhline(.9,ls='--',color='#aaa')
        ax.set_xticks(range(3),['nu=.005\nD/R=0.43','nu=.02\nD/R=1.71','nu=.08\nD/R=6.84'])
        ax.set_ylabel('Autonomy / query MSE');ax.set_title(f'High-frequency OOD, training noise={noise:g}');ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'high_frequency_transfer.png',dpi=180);fig.savefig(out/'high_frequency_transfer.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in analysis.items() if k not in ['first_screen_groups','clock_information_control','high_frequency']},indent=2))
if __name__=='__main__':main()

# 紧凑结果与复核入口

最终结论见 [实验分析](../FINAL_REPORT.zh.md)

原始聚合 [AGGREGATE_REPORT.json](analysis-r1/AGGREGATE_REPORT.json) 保留六次固定最终回放指标、每条轨迹与每个快照的标量误差，以及三种子配对结果

[development_curves.csv](development_curves.csv) 收录六次训练每25轮的同口径开发指标，共120行；各训练的500轮优化曲线、随机排列摘要、配置及最终回执位于 `training/`

`epochs.csv` 为已核验 `epochs.jsonl` 的逐行无损字段转换，原始 JSONL 的 SHA 仍由远端产物清单与紧凑归档回执保留，CSV 与原始 JSONL 分别记录字节身份，发布 CSV 统一使用 LF 换行

[真实调度终态](FINAL_SLURM_ACCOUNTING.txt)、[空队列](FINAL_SQUEUE.txt) 与 [传输核验](FINAL_SLURM_VERIFICATION.json) 证明六次训练和回收作业均 COMPLETED 0:0，终态文本通过终端独立远端哈希匹配

[回收执行回执](RECOVERY_EXECUTION_RECEIPT.json) 绑定实际回收作业2420820、零优化更新、最终聚合和归档哈希；[来源证明](analysis-r1/RECOVERY_PROVENANCE.json) 保留主训练真实调度快照及回收代码身份

[原始紧凑包回执](remote_archive_receipt.json) 与 [逐文件回收核验](COMPACT_RECOVERY_VERIFICATION.json) 对应远端231成员归档，GitHub 采用本目录明确选择的紧凑记录，完整原始归档清单与本目录发布清单分别保留身份

[独立复核](INDEPENDENT_FINAL_REVIEW.zh.md) 核对最终回放、完整更新数、配对初始化和批次顺序、有限指标及三种子聚合

本目录仅包含代码关联的指标、曲线、回执及必要记录，模型、优化器、原始预测与数据缓存均未收录

运行发布记录复核 `python3 ../operations/verify_published_results.py --results .`

运行绘图 `python3 ../operations/plot_final_results.py --evidence . --output figures`，需要 matplotlib 与 numpy，绘图使用标量 CSV 与聚合报告，不加载模型

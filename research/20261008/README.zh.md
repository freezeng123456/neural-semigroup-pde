# 半群 PDE：核心代码与结果记录

整理日期：2026-10-08。保留代码、协议、配置、结果表和结论；模型、优化器状态、预测数组、逐更新日志与实验归档按用户要求舍弃。负面结果保留记录。

## 已有结论

| 实验 | 可确认的结果 | 范围 |
|---|---|---|
| Burgers reference-v2 | 小模型和大模型的自主/时间条件 MSE 比分别为 1.0750、1.0154，均未通过收益门槛 | 3 种子、12 个模型；修正参考解后的探索实验 |
| Stage A 局部 OSG | 33/1,313 参数对照及一致性训练结果已记录 | 不用于推断全局 FNO 的容量需求 |
| DUE 官方涡度基线 | GDSG 联合相对 L2 为 1.0683%，SG0 为 70.2999% | 单训练种子，公开测试 100 条轨迹；计算成本不同 |
| Stage B 固定中点 | 相对 GDSG-500 耗时减少约 32.9%，开发误差比 1.0240；没有支持额外精度收益 | 4 臂 × 3 种子，12 次训练完整回收 |
| 两步映射 | seed0 四臂开发误差约 0.96%–1.06% | 只回收部分矩阵，不能作三种子成功判断 |
| 生成元 | 自主臂最终轨迹溢出；时间条件臂主误差约 1.02e27，积分细化失败 | 局部速度拟合下降不保证轨迹稳定；完整矩阵未完成 |

负面结果及停止原因见 [NEGATIVE_RESULTS.zh.md](NEGATIVE_RESULTS.zh.md)。详细指标以原结果表为准。

## 文件入口

- [文献综述](../../docs/research/LITERATURE_REVIEW_20261003.zh.md)
- [Burgers reference-v2 协议](../../docs/research/BURGERS_REFERENCE_V2_PROTOCOL.md)及[结果](../../docs/research/BURGERS_REFERENCE_V2_RESULTS.md)
- [官方基线报告](outputs/due-official-vorticity-20261004/BASELINE_REPORT.zh.md)及[比较指标](outputs/due-official-vorticity-20261004/comparison/metrics.json)
- [Stage B 报告](outputs/semigroup-midpoint-stage-b-20261004/RESULTS_REPORT.zh.md)、[逐种子结果](outputs/semigroup-midpoint-stage-b-20261004/analysis-r1/per_seed_results.csv)、[配对比值](outputs/semigroup-midpoint-stage-b-20261004/analysis-r1/paired_ratios.csv)
- [Stage A 完整结果](outputs/beijing-stage-a-20261004/FULL_RESULTS.zh.md)
- [后续实验的部分结果与失败报告](outputs/semigroup-next-methods-20261004/web-execution-20261005-r1/RECOVERY_AND_FAILURE_REPORT.zh.md)
- [理论及其边界](outputs/semigroup-next-methods-20261004/THEORY_AND_LIMITS.zh.md)
- [官方基线实现](work/due_official_vorticity/operations/run_official_observed.py)、[Stage B 训练](work/semigroup_midpoint_stage_b/train.py)、[两路线训练](work/semigroup_next_methods_r1/train.py)
- `records/`：由原回执提取的配置、更新数、标量指标和每轮曲线；省略预测数组和权重
- [SOURCE_MAP.json](SOURCE_MAP.json)：原始文件路径、SHA-256、保留方式
- [RETENTION_POLICY.zh.md](RETENTION_POLICY.zh.md)：当前保留边界
- [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)：本次补齐文件的字节数、SHA-256 和 Git blob 标识

## 代码来源与复现

本快照整理自本地四个提交，原末端为 `7948527673eb46dd3b9049f495df77acda8f68f9`。新源代码和协议被保留，未把原提交中的二进制产物重新上传。后续源码的原 SHA-256 记录在 SOURCE_MAP.json；仅 Markdown 中的本地绝对路径链接作了可移植处理。

上游 DUE 为 `AI4Equations/due@1e629104a12a9c3e3064996cbd4e301cec641ad8`，75 个文件的哈希清单位于 `work/due_official_vorticity/operations/official_source_manifest.json`。上游源码按该版本自行取回，不在这里重复存放。原始公开数据的下载地址、大小和 SHA-256 位于 `work/due_official_vorticity/data/download_manifest.json`。

运行环境以原协议和锁定依赖为准：主要 GPU 实验使用 Python 3.11、PyTorch 2.5.1、NumPy 1.26.4、V100。官方基线锁定的其他依赖见 `work/due_official_vorticity/operations/requirements-linux-cp311.lock`。Burgers CPU 实验的环境不同，见其 run card。

恢复公开源码与原始数据（这两条命令不会启动训练）：

```sh
python research/20261008/restore_public_inputs.py --source
python research/20261008/restore_public_inputs.py --data
```

可在删除本地数据前，用 `--verify-source /path/to/due --verify-data /path/to/data` 只校验现有副本。程序逐项检查记录的 SHA-256，拒绝覆盖来源或已有下载文件。

派生数据的生成顺序为 `prepare_data.py` → `prepare_two_step_data.py` → `prepare_flow_data.py`，分别位于 Stage B 和后续实验的 work 目录。必须按各自命令行传入源码清单、原始数据和新的输出路径；原始训练器仍检查固定数据哈希、环境和最终训练端点。

NPZ/Torch 容器重建后的文件哈希可能受运行环境或容器写入元数据影响；各数据报告同时保存数组内容哈希。不能跳过训练器的冻结校验，或声称只凭代码即可恢复已舍弃的原模型与 GPU 预测。精确回放原检查点需要重新训练或另有原始副本；本次没有重新训练。

## 证据边界

各历史报告中的“保留了模型/完整包”“可回放”等表述描述当时的验证状态。2026-10-08 的保留决策覆盖这些历史存储状态；当前只承诺代码和结果记录存在。报告中的已执行哈希校验、指标复算、CPU 回放属于历史验证记录，删除原始产物后无法仅凭这些记录再次执行完整独立复核。

后续实验状态截至 2026-10-05 最后一次远端实查。本次整理没有查询 Slurm、回收剩余作业或启动新任务；不把部分结果改写为完整的 24 次训练结论。

# SCNet 半群探索完整交付

先阅读 [完整报告](analysis-final/REPORT.zh.md)。本包包含36个训练实验、4个GPU小测试与108个高频评估端点的原始记录；主训练和高频评估均在SCNet RTX 3080完成。全部252个选定核验端点已在CPU重放通过。

- `runs/`：三个原始运行根及本地新增的重放、绘图产物。
- `analysis-final/`：中文报告、全部36模型CSV、机器可读比较及PNG/PDF图。
- `verification/`：回收核验、调度与版本收据、重放日志。
- `original_archives/`：从SCNet下载并核对远端SHA-256的原始归档，保留未添加本地分析前的字节。
- `source_snapshots/`：按固定commit提取的运行、分析源码与协议。`COMMITS.json`给出每组完整版本。
- `DELIVERY_MANIFEST.json`：除清单本身外每个文件的大小和SHA-256。

原始运行根中的`artifacts.sha256`对应服务器归档时的文件；在本地添加的审计和图表由总清单覆盖。因此原始字节核验可用`original_archives/`，最终交付完整性核验应使用`DELIVERY_MANIFEST.json`。

首批来源commit为21bc488，时钟信息对照87f2f49，高频评估574ac25。服务器Git因没有可用写入认证而未发布；结果通过SCNet文件管理回收，本地完成审计后发布到结果分支。

## 复现检查

环境实际版本在各config.json中：SCNet Python3.10.18/PyTorch1.12.1，CPU核验使用Python3.12/PyTorch2.14。分析依赖PyTorch、NumPy、Matplotlib。

从仓库对应源码版本或本包`source_snapshots/analysis/experiments`运行审计，给`audit_semigroup_discovery.py --root`传入首批`runs/semigroup-discovery-20260909-r1/run`。追加时钟对照使用相同审计函数和`run_semigroup_clock_control.matrix`；详细CPU执行记录见verification。高频评估入口支持`--device cpu`，必须使用新的输出目录，避免覆盖原结果。

这是一轮三种子的固定预算探索，包含明确标记的事后对照。它不是充分收敛、统计显著性或普遍PDE精度优势证明。没有合入主分支。

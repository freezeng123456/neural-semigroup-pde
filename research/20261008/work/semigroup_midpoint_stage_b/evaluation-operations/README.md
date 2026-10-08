# Stage B 最终评估 operations helper

这里只准备脚本。没有提交评估作业、运行模型或更改冻结训练 payload。

## 独立目录

训练根目录为现有 `20261004-midpoint-stage-b-r1`。评估使用相邻目录：

```text
/data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-evaluation-r1/
  evaluation-operations/
    evaluate_matrix.py
    slurm_evaluate_matrix.sh
    evaluate_final.py                         # 原冻结文件，必须保持 SHA
    EVALUATOR_FREEZE.json                     # 原冻结收据
    evaluation-preparation-r2/AST_AND_MATH_REVIEW.json
    evidence/<attempt-id>/                   # 每次独占；不可覆盖
  results/<run_name>/                         # 12 个原始模型各自独占输出
```

必须保留此独立布局：冻结评估器不允许输出位于训练源码目录内部。`--training-code-root` 仍直接指向原始训练根目录，没有复制或修改训练代码。

## CLI

在上述目录中的 helper 运行；下面只是待执行命令示例。`gate` 仅访问文件及真实 `sacct`，不加载模型，不要求 GPU allocation。

```bash
/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u \
  /data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-evaluation-r1/evaluation-operations/evaluate_matrix.py \
  --training-root /data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-r1 \
  --evaluation-root /data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-evaluation-r1 \
  --training-array-job 2349407 --attempt-id gate-final-r1 --mode gate
```

`evaluate` 模式必须在单卡 V100 Slurm allocation 内运行。入口 shell 的四个位置参数为：训练根目录、评估根目录、训练 array master job ID、全新的 attempt ID。例如 allocation 内调用：

```bash
bash /data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-evaluation-r1/evaluation-operations/slurm_evaluate_matrix.sh \
  /data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-r1 \
  /data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-evaluation-r1 \
  2349407 evaluate-final-r1
```

资源建议沿用已核验的北京环境：`partition=gpu`、`account=scv7tsq`、1 GPU、1 task、10 CPUs、`02:00:00` 时限，单个作业、无 array。时限是保守的运维预算，尚无本轮 12 模型完整评估的实测耗时。实际提交前由主任务核验当前配额。脚本不含任何 `sbatch` 调用，不修改 scheduler 提供的 `CUDA_VISIBLE_DEVICES`。

## 开始推理前的门槛

1. 冻结训练 payload、评估器和 freeze 收据全部 SHA 匹配；12 个 matrix cell 完整。
2. 每次尝试重新查询 `sacct --jobs=2349407 --parsable2 --format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,AllocTRES`，保留表头、原始 stdout/stderr 和真实命令。证据文件写入后设为只读。
3. 每个 `2349407_<task>` 必须为 `COMPLETED / 0:0`，其数值 `JobIDRaw` 必须与训练 `environment.json` 的 `SLURM_JOB_ID` 一致。12 个 raw ID 必须唯一。
4. 每个完整 `artifact_manifest.json` 校验通过，`RUN_RECEIPT`、配置、最终开发预测及实际预算均正确：50000 或 33300 updates；不得接受 preflight、早停或中间网络。
5. **全部 12 个通过后**才生成最终模型评估进程；真正 checkpoint strict load 由冻结评估器再次执行。

## 部分失败与结果

严格串行运行，任何时刻只有一个模型评估进程。每个模型的 stdout、stderr、退出码和目录均保留；一次普通进程失败会记录后继续其他模型。运行成功但诊断未定义或数值不佳，同样保留原结果，不剔除种子。工程失败与科学质量状态分开。

`run_index.json` 在第一个模型启动前保存全部 12 行，schema 为 `stage_b_matrix_analysis_inputs_v1`。若作业中断，没有输出的模型仍在索引中。回收后应另写一份使用本地实际路径的新索引，保留原远端索引；分析器不会猜测路径。

每次尝试使用新的 evidence attempt ID。已有模型输出会阻止再次评估，脚本不覆盖、不自动复跑，也不把已有目录当作成功。需要重试时，先核查失败原因，再由主任务安排新的独立评估根目录并保留原失败产物。`OPERATIONS_REPORT.json` 和逐模型 `EVALUATION_REPORT.json` 才是执行证据；这份说明不证明任何评估已经完成。

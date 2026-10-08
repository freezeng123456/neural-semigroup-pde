# Stage B：本地恢复与 CPU 回放

## 输入与范围

- 使用现有 `RECOVERY_PLAN.json` 的 12 个任务与固定归档名。
- 下载目录：`outputs/semigroup-midpoint-stage-b-20261004/archives`。
- 需要下载真实 `MATRIX_CAPTURE_REPORT_R2.json`，并提供从远端独立读到的 SHA-256。该报告内嵌每个归档的原始回执；生成器同时核对冻结 packer 的 JSON 序列化哈希。
- 恢复仍调用冻结 `recover_evidence.py`；回放仍调用冻结 `replay_final_checkpoint.py`。生成器只写命令计划，不执行命令、不创建训练、恢复或回放输出。

## 生成计划

在 `work/semigroup_midpoint_stage_b/evaluation-operations` 中运行：

```sh
python3 -B prepare_local_recovery_replay.py \
  --capture-report /绝对路径/MATRIX_CAPTURE_REPORT_R2.json \
  --capture-sha256 从远端独立观察到的报告SHA256 \
  --output-plan /绝对路径/LOCAL_RECOVERY_REPLAY_PLAN_R1.json
```

输出计划必须使用新的文件名。每条 `commands[].argv` 是完整参数列表；`commands[].shell` 使用 shell 引号序列化同一命令。按每个任务的 phase 顺序执行：先恢复成功，再回放。每次保存实际返回码、stdout、stderr；日志放在独立目录，例如 `BOUT/local-recovery-execution/r1/<run_name>`。某项失败则保留日志与部分输出，停止该项后续阶段，继续处理其他明确可执行的任务。不要自动重试失败命令。

生成命令在实际执行时核对冻结脚本 SHA，并用已有 Python 3.12 解释器。回放命令要求 Torch `2.5.1`、NumPy `1.26.4`，设置已有依赖 overlay，禁用字节码与 Python 优化，使用 4 个 CPU 线程。通过 `runpy` 执行原封不动的冻结脚本。

## 输出与状态

| 内容 | 目录或文件 |
|---|---|
| 恢复文件 | `BOUT/training-recovery/<run_name>/<remote_root_name>` |
| 恢复报告 | `BOUT/training-recovery-reports/<run_name>.json` |
| 回放输出 | `BOUT/replay/<run_name>` |
| CPU 两步预测 | `development_first_two_steps_cpu.npy` |
| CPU 回放报告 | `FINAL_REPLAY_REPORT.json` |

计划始终保留 12 行：

- `AWAITING_ARCHIVE_DOWNLOAD`：归档尚未下载，不生成命令。
- `COMMANDS_READY_EXECUTE_SEQUENTIALLY`：外部回执与归档 SHA、大小一致，可按顺序执行列出的新建阶段。
- `RECOVERY_AND_REPLAY_REVERIFIED`：完整已有输出已重新检查；无需重做。
- `BLOCKED_INSPECT_RETAIN_OUTPUTS`：存在不匹配或部分输出；保留全部文件并人工检查。

已有恢复仅在报告身份、原始归档清单、全部恢复文件哈希、冻结源文件及完整训练回执重新通过后复用。`.partial` 目录、缺失报告或只有部分回放文件均阻断重试。已有回放需匹配任务、最终检查点 SHA、冻结验证器 SHA、20 条轨迹的 `2e-5` 容差、实际完整训练预算，以及保存预测的哈希。

执行后再次运行生成器，另存新计划以检查完整输出。计划中的 `completion_verified` 仅描述恢复与所定义回放范围，不能代替最终科学比较和结论。

## 数据与诊断命名

CPU 回放只使用 `BWORK/data-r1/development_data.npz`，即 20 条 development 轨迹。它重新计算全部 51 个快照的开发集指标，并对每条轨迹回放前 2 步。它不读 `VorticityOSG_test.mat`，不执行 600 点组合缺陷诊断，不重复完整 99 步公开测试。这些属于单独冻结的最终评估及其结果回收流程。

已有 PASS 回放的复用依赖原报告与保存输出的身份和哈希；重新检查不会再次做数值推理，也不能独立证明外部产生的历史报告所用运行环境。当前生成的执行命令会在推理前检查实际运行环境。

# 官方 GDSG 固定配点的确定性重建：审阅版

状态：本计划在执行前完成并经独立审阅；随后获准执行。实际重建已通过，703 项检查全部成立，耗时 18.21 秒。脚本位于 `work/due_official_vorticity/operations/reconstruct_official_collocation.py`。未调用训练、前向、反向、预测或优化器更新。

## 可验证路径

1. 原入口先导入 `due`、示例 `network`，再按作者 loader 处理原 MAT、构造 FNO 和 `PDE_osg`。脚本保持这一顺序，核对 75 个冻结源码文件、两份 MAT 以及原 YAML。
2. `due/models/pde.py:55` 在模型初始化之后重新设置 seed=0。其后优化器、调度器及 DataLoader 的构造没有抽取训练随机数。`due/models/pde_osg.py:46` 先生成形状 `(2500,1,64,64,1)` 的 `u0_rand`；下一行生成 `(2500,1,2)` 的时滞。随后作者将两个物理时滞之和重新编码并铺满空间。配点仅在构造时生成一次。
3. 原始模型摘要 `3c15e9d2181515a55cb0583ea93a5f076789d249444c403856d4ffa1705f0651` 同时保存在预检和正式运行回执中，可核对重建的初始化。
4. 预检 `epoch1_profile_training_state.pt` 与正式 `epoch500_final_training_state.pt` 都保存 CPU 和 NumPy RNG 状态。脚本读取两个已回收清单，校验这两份文件及其网络文件摘要，再检查数据处理后的 NumPy 状态和网络内归一化边界。
5. 原训练的 CPU RNG 后续只用于每轮 DataLoader 的 base seed 与 RandomSampler seed；前向、反向在 CUDA 上，FNO 无 dropout 等随机层。脚本使用同版本 Torch、同长度、同批量和同默认采样设置，以整数数据集重放 500 轮索引迭代，要求第 1 / 500 轮 CPU RNG 均与历史保存状态逐字节相同。它不访问 500 轮状态批次，不计算损失。

## 改动与证据边界

- 在新进程中导入原作者实现；只把构造设备改为 CPU、保存路径改到本诊断目录。CPU→CUDA 的参数复制不抽取 CPU 随机数。源码及历史配置不改。
- Torch 要求主版本精确为 2.5.1、NumPy 为 1.26.4，记录本地所有依赖版本及 DataLoader/RandomSampler 实际实现源码。历史 CUDA 构建为 2.5.1+cu121；本地 CPU 构建差异明确记录，能否通过 RNG 与初始化核验待执行确认。
- **原训练配点数组未直接保存，也没有当时记录的配点摘要。** 通过后可称“原代码、已知 seed 和两个历史 RNG 锚点验证的确定性重建”，不能称“已与当时实存数组直接逐字节核对”。重建的配点摘要和批序摘要是本次新生成的证据。
- CPU RNG 终态本身不证明数值输出相同；其作用是检查重建随机路径与原构造、原数据形状、原随机操作顺序的吻合。初始参数、NumPy 状态及边界作为额外独立锚点。所有锚点必须一起通过。
- 任一摘要、参数、边界、NumPy 或 CPU RNG 状态不符，脚本写 `RECONSTRUCTION_NOT_VERIFIED`，禁止后续把数组标为原训练配点，也禁止报告训练配点与新点的泛化差距。届时只比较独立新随机盒状态与真实状态。

## 成功输出

- `RECONSTRUCTION_REPORT.json`：逐项核验、历史输入摘要、边界、运行环境、零优化更新、零网络调用、证据限制。
- `artifacts/fixed_u0_normalized.npy`：2500 个 64×64×1 的作者归一化状态，float32。
- `artifacts/fixed_dt_normalized.npy`：每个配点的两个子时滞及总时滞编码，float32；保持作者的计算顺序。
- `artifacts/fixed_substeps_physical.npy`：两个物理子时滞，float32。
- 数据加载日志、实际 Torch 索引采样实现及输出摘要。

下一步 A 覆盖诊断必须先验证本报告成功，再加载上述固定数组；SG0 并未用这些配点训练，应将该组称作“原 GDSG 固定配点”。

## 实际核验结果

`RECONSTRUCTION_REPORT.json` 的状态为 `VERIFIED_DETERMINISTIC_RECONSTRUCTION`，包含完整输入/输出摘要。两个历史 CPU RNG 锚点、两次初始化摘要、两次 NumPy 状态和两次归一化边界均精确吻合；75 个冻结源码及所有历史输入前后摘要不变。

得到 2500 个固定配点，归一化状态范围为 ±0.9999997616；作者 float32 时滞边界为 `[0.5006624460, 1.4997749329]`。其中 2185 个（87.4%）物理总时滞超过监督最大单步时滞。该比例是恢复出的配点事实，不是模型好坏或约束失效的证据。

本次重建脚本摘要：`adf26a5deb838eb6eb84ff7be62626a82ead8bc7b57ca9da687fead9bbe71cb0`。固定状态 NPY 摘要：`5e03c331dbcc8b66389413cae8b14af4d7a425d8c2d01e18862127c4fd07b113`。本结果仍受上文“未直接保存原数组”的证据边界限制。

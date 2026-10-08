# 把自主生成元迁移到相同 FNO 骨干：可行性与边界

2026-10-04；条件性方案，尚未改模型或启动训练。路线在 SG0/GDSG 两个完整官方基线回收、复算后决定。

## 1. 判断

**技术上可行，适合作为机制对照；“FNO 生成元加 ODE 积分”本身已有直接先例。** 当前 SG0 诊断（历史文件路径；原始产物已按保留策略舍弃） 显示：在全部真实状态的 9,900 个转移上，一步平均相对 L2 为 0.05883%，自由滚动末端则为 284.10%。这说明沿真实状态的一步拟合不能代表滚动可靠性，但尚不能把原因唯一归结为缺少半群结构、参数容量或状态分布偏移。自主模型也会使用自己预测的状态，仍可能偏离数据覆盖范围。

## 2. 最接近的原文

| 工作 | 与候选路线的关系 |
|---|---|
| Chen、Rubanova、Bettencourt、Duvenaud，**Neural Ordinary Differential Equations**，NeurIPS 2018 | 用网络表示状态导数，由 ODE 求解器产生输出；§1 式 (2)、§2 给出前向与梯度框架。生成元加积分属于既有基本方法。[原文](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html) |
| Chuanqi Chen、Jin-Long Wu，**Neural Dynamical Operator**，JCP 520:113480，2025；预印本 2023 | **最直接先例。** §2.1–2.5 用 FNO 表示动力学右端，通过短轨迹监督学习；§3 使用固定步 RK4，包含 Burgers、NS 和 KS。§2.6 还使用长期统计与 EKI，但这些不属于当前最小迁移方案。[全文 v4](https://arxiv.org/html/2311.11798v4)、[出版信息](https://arxiv.org/abs/2311.11798)、[作者代码](https://github.com/AIMS-Madison/Neural-Dynamical-Operator) |
| Cho 等，**Operator-Learning-Inspired Modeling of Neural Ordinary Differential Equations**，AAAI 2024 | §4.3 式 (8) 将 Fourier operator 放入 ODE 右端；采用 BFNO，主要实验为分类/生成，不能把其结果直接当作本 NS 任务的证据。[原文](https://arxiv.org/html/2312.10274v1)、[作者出版页](https://hyundong98.github.io/publications/) |
| Hou、Huang、Perdikaris，**CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators**，ICLR 2026 | §4 用样条速度监督训练神经算子，推理再积分；附录 A.4.3 明确有端点监督 Neural ODE 对照。CFO 的 $N_\theta(t,u)$ 显含时间，不能直接援引为自主半群。该文说明积分训练成本及直接速度监督也是已有研究方向。[原文 §4、A.4.3、A.4.8](https://arxiv.org/html/2512.05297)、[会议原文](https://openreview.net/pdf?id=IQhaeSzyup) |

这些工作支持方法可行性与已有重叠，不证明该迁移会优于当前 GDSG。

## 3. 最小迁移定义

沿用官方训练数据确定的归一化 $z=A(u)=(u-\mu)/\sigma$，其中 $\mu=(u_{max}+u_{min})/2$、$\sigma=(u_{max}-u_{min})/2$，不是均值与标准差。当前直接模型为

$$G_\theta(z,\Delta)=z+\Delta N_\theta(z,\tau(\Delta)).$$

候选自主模型则定义

$$\dot z=F_\theta(z),\quad z(0)=A(u_0),\qquad \widehat S_\Delta(u_0)=A^{-1}(\operatorname{ODESolve}(F_\theta,A(u_0),0,\Delta)).$$

$F_\theta$ 保留 DUE 的 4 个 FNO 块、width 20、modes 12、GELU、空间 lifting/projection，只输出状态变化率。它不接收绝对时间、查询时长、子步长或剩余时间。归一化 RHS 的单位是“归一化状态/物理时间”；物理 RHS 为 $\sigma F_\theta((u-\mu)/\sigma)$。

首个候选应从相同监督端点重新学习，用与 SG0 相同的归一化相对 L2 比较积分终点与标签。沿积分步骤反传；先明确离散计算图，再考虑内存优化。不要将现有 OSG 权重去掉外层 $\Delta$ 后就宣称得到瞬时生成元：OSG 学的是一段时间的平均增量，且归一化时间输入 0 并不代表物理时间间隔 0。

这一候选暂不加入新的物理项、频谱滤波、记忆、长期标签、额外 loss 或模型容量。旧 Burgers 的固定黏性项、保守通量，以及其他方程的能量下降约束，都不能未经重新论证移入受迫 NS。方案目的是比较“直接时段映射”与“共享生成元再积分”的整体建模选择。

## 4. 理论可以承诺什么

以下是对候选架构的数学分析，尚非当前模型已取得的误差证书。

- **固定网格上的连续流：** 若使用当前有限权重的线性/Fourier 运算和 GELU，去掉时间依赖后得到有限维、全局 Lipschitz 的 RHS，因而有全局唯一连续流，并满足 $\Phi_0=I$、$\Phi_{s+t}=\Phi_t\circ\Phi_s$。该结论指精确实数 ODE；Lipschitz 常数可能很大，不能据此承诺长期有界、误差小或浮点计算不溢出。
- **实际积分器：** RK4 通常只近似满足任意时间分拆的半群关系。相同内部步长序列重新分组得到零缺陷，只验证运算路径一致；不能作为学到正确动力学的证据。必须用步长细化区分积分误差与模型误差。
- **准确性仍需真值一致性：** 若目标状态路径为 $z(t)$，且在相关域内有生成元残差 $\|\dot z(t)-F_\theta(z(t))\|\leq\varepsilon$ 和合适的 Lipschitz 界 $L$，才能得到形如 $E(t)\leq e^{Lt}E(0)+\varepsilon\int_0^t e^{L(t-s)}ds$ 的连续误差界，再加时间离散误差。小半群缺陷不能代替 $\varepsilon$。
- **不自动得到的性质：** 真实 PDE 生成元辨识、网格细化收敛、能量/涡量守恒、耗散，以及对未见状态的稳定性。观测间隔约 0.5–1.5 时，拟合有限端点数据也不等于辨识瞬时生成元。粗状态是否具有足够闭合性仍是条件。

还有一个具体表达限制：当前官方 FNO 不输入空间坐标，周期网格上的共享权重网络对整数格点平移等变；对空间常量状态只能输出空间常量。归一化也对整个空间使用相同标量，因此不改变这个性质。取物理零状态 $u=0$，真实受迫涡度方程的 RHS 为非均匀 $f(x)$，该骨干的 RHS 则只能是常量。所以它不能在**包含零状态的完整状态域**上精确表示真实 RHS；同理，直接 OSG 对零状态只能输出常量，不能表示任意小正时间的真实流。这里是对 [网络源码](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/networks/fno.py#L52) 与 [全空间统一归一化](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/datasets/pde.py#L151) 的数学推导，**没有证明本次轨迹分布内的误差来自这个限制**。实际轨迹状态可能携带外力相位信息，允许网络在该子集上拟合很好。

此外，FNO 的 spectral 分支保留 12 个模式，不代表整个输出严格带限：逐点支路和非线性仍传递/生成高频。因此不能把 NDO 的经验稳定性解释直接变成当前骨干的滤波或耗散定理。[官方骨干](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/networks/fno.py#L52)

### 坐标与外力输入已经有先例

- Tran、Mathews、Xie、Ong，**Factorized Fourier Neural Operators**，ICLR 2023：§4 表 1 把涡度、黏度、外力 $f_t$ 作为 TorusVisForce 的输入；§5 “Flexible input representations” 和图 5(a) 对外力/黏度输入做消融，图 5(b) 检查额外坐标与速度，表 A.7 列出各数据集的输入组合。其坐标输入为 $(x,y)$ 两通道。该论文已有结构输入的直接先例，但数据、骨干和训练设置均不同，不能将其收益数值移用于 DUE。[原文 §4–5、表 A.7](https://arxiv.org/html/2111.13802v3)、[作者代码](https://github.com/alasdairtran/fourierflow)
- NeuralOperator 官方 **Grid Embeddings** 文档直接演示在物理变量后拼接 $x,y$ 通道，并给出 `GridEmbedding2D`。位置条件不是新概念。[官方文档](https://neuraloperator.github.io/dev/auto_examples/layers/plot_embeddings.html)

若以后采用坐标或已知固定外力，应先称为**结构表达控制**：它改变了输入信息和对称性假设，不能自动算作方法创新。固定 $f(x)$ 的条件 RHS $F_\theta(z;f)$ 仍可对每个固定 $f$ 定义自主流；若外力显含时间，则需要非自主演化或把外力状态纳入状态空间。周期域上坐标编码也需说明周期接缝，而直接拼接已知 $f(x)$ 注入了额外物理信息。要归因于时间建模结构，相应输入应在直接映射与生成元两侧对齐，不能同时加坐标与改成 ODE 后将收益全归于半群。这里只记录条件与先例，当前不增加训练臂。

## 5. 公平比较必须保留的记录

| 项目 | 条件性方案 |
|---|---|
| 数据与监督 | 相同训练轨迹、相同 2,500 对端点、相同训练归一化、相同损失口径；不以全轨迹/导数标签额外增强自主臂。冻结独立的初始化、batch 排序随机源。 |
| 空间骨干与参数 | 当前 OSG 为 467,661 个 tensor 元素，折合 928,461 个实自由度。删除时间输入使 lifting 少 20 个实权重，即 928,441；若保留恒零输入则有 20 个无效权重。报告名义和有效容量，不靠无用参数声称严格匹配。 |
| 数值步长 | 用开发数据上的步长细化确定误差是否足够小，再冻结最大物理步长与末尾短步规则。随机时滞时可用 $m_i=\lceil\Delta_i/h_{max}\rceil$、$h_i=\Delta_i/m_i$；报告实际执行量及批处理 padding，不把每个查询固定子步数误称固定物理步长。当前不指定未经验证的 $h_{max}$。 |
| 训练成本 | SG0 每 batch 1 次、GDSG 6 次 forward；自主 RK4 每条区间为 $4m_i$ 次 RHS，反传/重算另计。相同 epoch 或更新次数只匹配数据遍历，不能匹配计算成本。记录 wall time、峰值显存、前向/反向 NFE、优化更新数及有限性。 |
| 推理成本 | 单个时滞的直接模型 1 次 forward，自主模型 $4m_i$ 次；两条直接模型在推理时都不计算正则。以精度—实际时间/NFE 一并评价，不能只呈现参数量。 |
| 模型选择 | 步长、训练停止与 checkpoint 选择仅使用预留开发数据。当前官方基线保持原版 final epoch 500 规则，不回头改选更好的 test checkpoint。若未来另做相同算力预算比较，应预先冻结各臂规则并单独报告。 |
| 测试独立性 | 当前公开 100 条 test 已用于诊断，今后仍可作复现/探索比较，但不能作为新方法开发后的首次独立测试。确认性结论需要冻结后未查看的数据。 |

明确积分法还暴露刚性风险：更小步长会增加成本，较大步长可能使离散模型不稳定；训练也可能学到依赖所用积分器的补偿。必须验证步长细化后结论是否保持。逆向连续 adjoint 不应默认为与离散反传等价，尤其在耗散动力学中；若采用它，需要单独检查梯度数值一致性。

## 6. 如何定位，何时决定

第一阶段名称应是“同 FNO 骨干的自主 Neural ODE 对照”或“NDO 思路的受控迁移”。半群存在性、RK4、FNO RHS、任意时间查询和参数共享均已有先例，不能列为新贡献。即使胜过 SG0，也只能先说明本任务中该整体选择有效；不能越过 GDSG 对照，也不能把额外计算量造成的收益全部归因于半群。

值得继续的条件是：两条官方基线完成后，已有诊断支持明确的失败机制；候选在积分收敛、容量和成本清楚的条件下提供可重复的增益；其机制能与已有 NDO/GDSG 的已知作用区分，并在未触碰数据上检验。若效果只来自更小步长、更多计算、长程监督或重新选 checkpoint，应按这些原因报告。若 GDSG 已解决主要问题，自主路线也可能只是一项代价更高的对照。

**当前只完成可行性与风险准备；没有认定自主 FNO 更优，也没有冻结或执行新的训练矩阵。**

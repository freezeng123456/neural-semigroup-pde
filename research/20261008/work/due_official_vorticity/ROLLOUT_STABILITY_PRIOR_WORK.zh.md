# 滚动误差传播：针对性文献核对与研究门槛

2026-10-04。只读文献与已有诊断；未改模型或启动训练。等待 SG0/GDSG 两条完整基线后选择路线。

## 1. 新证据支持什么

已核对 [误差传播诊断](../../outputs/due-official-vorticity-20261004/sg0-error-propagation/ERROR_PROPAGATION_DIAGNOSTIC.json)：SG0 在真实状态上的局部误差仍小，但已产生误差经过下一次模型调用后通常被放大。

| 输入时刻 | 局部误差均值 $\|b\|/\|u_{t+1}\|$ | 实际误差方向 gain 中位数 |
|---|---:|---:|
| 40 | 0.000512 | 1.0751 |
| 60 | 0.000479 | 1.1075 |

同一推进接口下，令 $e_t=\hat u_t-u_t$、$b_t=G(u_t)-u_{t+1}$，则严格有

$$e_{t+1}=\underbrace{G(u_t+e_t)-G(u_t)}_{a_t}+b_t,\qquad g_t=\frac{\|a_t\|}{\|e_t\|}.$$

这里的 $g_t$ 是**沿已发生有限误差的割线增益**，不是最大 Jacobian 奇异值、全局 Lipschitz 常数或 Lyapunov 指数。归档回放还单列了 CPU/GPU 接口差异。$\|e_{t+1}\|$ 也取决于 $a_t,b_t$ 的方向，不能把各时刻增益中位数直接相乘来重建曲线。当前证据尚未测出真实流在同一扰动下的增益，因此不能将全部 $g_t>1$ 都定义为不物理，更不能推出必须全局收缩。

## 2. 必须承认的已有做法

| 原始工作；已读位置 | 机制与对我们的边界 |
|---|---|
| **Message Passing Neural PDE Solvers**，Brandstetter、Worrall、Welling，ICLR 2022；§3.1、§4.2、附录 F Algorithm 1。[原文](https://arxiv.org/html/2202.03376v3)、[作者代码](https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers) | Pushforward 先让模型从真实历史推进，随后把自身预测作为输入，监督未来真实状态；前面的推进截断梯度，只反传最后一次预测。两步情形可写为 $\ell(G(\mathrm{stopgrad}(G(u_{t-1}))),u_{t+1})$。它直接处理模型诱发的输入分布偏移。Temporal bundling 是一次预测多个时刻，属于另一项设计。自生成误差增强、停止梯度、短滚动监督均已有先例；有限采样的训练损失不构成全域稳定定理。 |
| **PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers**，Lippe 等，NeurIPS 2023；§2–3、§4.1、附录 C/E.1/E.8。[原文](https://arxiv.org/html/2308.05732v2)、[作者代码](https://github.com/pdearena/pdearena) | 初次预测后，用逐渐减小噪声尺度的条件去噪反复修正输出；训练时向真实下一状态加噪并预测噪声。目标是恢复被普通损失忽略的低幅值频率，既包括高频，也可能包括低频。它增加每步推理调用量；不是普通输入加噪的同义词。作者在自己的 KS 设置中发现 pushforward 未明显获益；这不预示我们这里的结果。E.8 还出现极高频积累，额外截断才维持更久的频谱稳定；频谱稳定时间不等于轨迹准确时间。 |
| **Learning Chaotic Dynamics in Dissipative Systems**，Li 等，NeurIPS 2022；arXiv 标题为 *Learning Dissipative Dynamics in Chaotic Systems*；§2–3 式 (7)–(9)、Theorem 1、附录 C。[会议页](https://papers.neurips.cc/paper_files/paper/2022/hash/6ad68277e27b42c60ac228c9859fc1a2-Abstract-Conference.html)、[正文](https://arxiv.org/html/2106.06898v2)、[代码](https://github.com/neuraloperator/markov_neural_operator) | MNO 在吸引子外的壳层采样，惩罚 $\|G(u)-\lambda u\|^2$，$0<\lambda<1$；另可在远域把模型与耗散安全映射混合。软正则本身没有普遍耗散保证，硬后处理与假设需单独审查。Theorem 1 是固定有限步数、紧初值集和局部 Lipschitz 条件下的存在性逼近结论，不保证训练得到该模型，也不保证无限时间准确。保持有界、吸引子统计正确，与逐轨迹准确必须分开。 |
| **Towards Stability of Autoregressive Neural Operators**，McCabe 等，TMLR 2023；§3.1、§4.1–4.2。[原文](https://arxiv.org/html/2306.10619v2)、[作者代码](https://github.com/mikemccabe210/stabilizing_neural_operators) | 分析 FNO 高频积累与分布外敏感性；采用频域谱归一化、逐通道可分离卷积，并重排块结构，使经过非线性的路径随后经过可学习滤波；还研究输入依赖的滤波。单层谱范数受控不能直接等同于整个残差推进器收缩。我们不能把“约束增益、加滤波、修正 aliasing”本身当成新发现；本次 gain 也未定位其来源必然是高频或 aliasing。 |

另有两项近期**预印本**直接触及潜在方案，只作为查重边界：Nie、Suzuki 的 **JAWS**（arXiv:2603.05538v3，§3.3–3.5）已有空间自适应 Jacobian 正则与短滚动组合；Yang、Lu、Mou 的 **Derivative-Informed Training … via Sketched Tangent Consistency**（arXiv:2609.32797v1，§3.1–3.4）已有随机方向 JVP 与 PDE 切线方程一致性。后者要求已知可微物理，其 NS 实验是稳态问题。不能据这两篇推导当前任务的稳定保证，也不能将自适应 Jacobian 惩罚或物理敏感性匹配重新包装为首创。[JAWS 原文](https://arxiv.org/html/2603.05538v3)、[sTCL 原文](https://arxiv.org/html/2609.32797v1)

## 3. 更具体、可证伪的问题

**待检验假说：实际滚动误差集中在某些方向；模型在这些方向上相对真实流产生额外传播误差。针对这些方向校准，比同成本的随机扰动或 pushforward 更有效，同时保留真实动力学允许的增长。** 这比“gain 大，所以加正则”明确，但尚未成为已证新贡献。

先在开发数据上比较

$$d_G(u,\delta)=G(u+\delta)-G(u),\qquad d_S(u,\delta)=S_{\rm ref}(u+\delta)-S_{\rm ref}(u).$$

其中 $\delta$ 来自模型短滚动残差，停止梯度并限制到参考求解器有效的状态范围；另用等幅随机方向对照。既检查范数增长，也检查传播向量的方向。尺度细化用于区分有限扰动非线性与局部导数效应。**只有额外传播误差确实存在，才考虑**把

$$R_{\rm dir}=\mathbb E\frac{\|d_G-d_S\|^2}{\|\delta\|^2+\varepsilon^2}$$

作为候选约束；$\varepsilon$ 按物理单位及数值分辨率冻结。它约束的是与参考动力学的差异，允许真实 $\|d_S\|/\|\delta\|>1$。该式属于既有导数/割线匹配家族；可能的贡献只能来自**误差方向选择的必要性、相对真动力学的校准及其对滚动性能的可重复因果证据**，不能仅靠这个损失公式立项。

这个假说有明确失败条件：

1. 参考流沿同一方向也有相近增长，且 $d_G-d_S$ 很小：当前增益不支持“模型额外放大”解释。
2. 实际误差方向与等幅随机方向同样有效：定向选择没有独立价值。
3. 校准降低了局部传播误差，滚动却不改善，或破坏一步精度/真实不稳定模态：该机制不足以解释收益。
4. 收益在匹配额外标签、训练计算与推理调用后消失：应归因于新增监督或算力。

## 4. 路线选择与比较边界

- 首先给完整 GDSG 做相同的 $a,b,g$ 诊断。若它已改善传播并解决主要误差，就先解释这一已有方法的作用，避免强行新增方法。
- 若需要新训练，**同 FNO 的 pushforward** 是最直接的必要控制；有低幅值频谱证据时才优先 PDE-Refiner，有明显壳层逃逸时才优先 MNO。谱归一化/滤波属于明确敏感性或 aliasing 假说的控制。这里未冻结新增实验矩阵。
- Pushforward 需要连续时刻标签和实际时滞；方向校准需要新的参考推进结果。使用同一 MAT 不等于使用相同监督量。GDSG 的组合约束不需要这些新标签，必须单列差异。
- $S_{\rm ref}$ 需与原数据的方程、外力、网格投影和精度兼容，并通过数值收敛检查。若粗尺度状态不闭合，就不能把任意粗网格重启器当作真流；没有可信参考时，只能做经验稳定化，不能声称识别或消除了“非物理放大”。
- 本次公开 100 条 test 已用于发现机制；之后调方向、扰动尺度、训练停止和正则强度必须使用独立开发数据。确认性结论还需未查看数据、多 seed 与成本记录。

**结论：已有文献已经覆盖滚动状态增强、去噪修正、耗散回拉和敏感性控制。我们的潜在研究空间需要由“哪种传播误差、在哪些方向、为什么此干预有效”的证据支撑；当前只有诊断线索，尚未选定新方法。**

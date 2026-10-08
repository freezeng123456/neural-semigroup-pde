# 半群、连续时间算子学习：文献核查

检索截止：2026-10-03。以下均回到作者全文或正式出版页核对；没有运行作者代码。不同论文的数据、空间分辨率、监督信息和计算预算不同，不能据表中误差排一个统一的领先榜。

## 对本项目最重要的判断

1. “把半群一致性加入神经算子，并改善小数据或噪声下预测”在 2023 年已有直接先例。仅将 soft consistency 改成 autonomous ODE，不能自动形成充分的新贡献。
2. 现有路线分成：直接流映射加组合损失；学习生成元再积分；在网络参数空间学习连续或随机流；利用特定 PDE 的几何结构；在受限半群类别内证明可逼近性和误差传播。它们解决的问题并不相同。
3. 对我们而言，最急需补上的强基线是 Deep-OSG / DUE 的直接流映射，以及 CFO 的直接速度监督训练。前者检验硬约束相对软约束的价值，后者检验我们是否主要受训练穿过积分器的成本和优化困难限制。
4. 三种“噪声”必须分开：观测误差、训练中的人为扰动、动力学本身的随机性。不能将 SINGER 的 dropout 稳定化或 CFO 的插值噪声直接当作测量噪声鲁棒性证据。
5. 2026 年理论前沿已经细化到凸单调半群、Chernoff 一步逼近、Hölder 空间稳定性和覆盖数速率。“存在性、半群律、Gronwall 误差界”本身不足以支撑普适性能优势。

## 核心论文

### 1. Deep-OSG — 直接先例，必须复现

Junfeng Chen, Kailiang Wu. **Deep-OSG: Deep Learning of Operators in Semigroup**. *Journal of Computational Physics* 493 (2023), 112498；arXiv:2302.03358。

- 机制：`u + Δ Nθ(u, Δ)` 严格满足零时刻恒等；半群组合由 GDSG 损失软约束，额外随机状态和时滞无需新标签。LISG 用已观测短轨迹的多步一致性。
- 理论：Theorem 1 假设有效增量的全域逼近误差、Lipschitz 性和可达域；Theorem 2 在不同时间划分的预测接近这一前提下界定误差方差。两者不证明训练会收敛，也不证明组合残差小必然预测准确。
- 证据：ODE、刚性化学反应、平流、Burgers、Navier–Stokes；线性 ODE 的 2%/5% 乘性均匀观测噪声。该噪声展示主要是方法预测图，不能当作容量严格匹配的硬/软半群因果比较。部分 ODE 训练 100,000 epochs。
- 我们的缺口：当前 autonomous/query 比较未包含这一直接、可变时滞、半群正则化的强对照。

正文核查位置：[§3.1–3.3、Theorems 1–2、§5.1.1、§5.2](https://arxiv.org/html/2302.03358v2)。[正式出版页](https://www.sciencedirect.com/science/article/abs/pii/S0021999123005934)。原论文未找到直接代码链接；同作者后续 [DUE](https://github.com/AI4Equations/due) 提供 OSG-Net / GDSG 实现，不将其误标为原始实验复现。

### 2. Rotman et al. — 单终点监督加组合约束

Michael Rotman, Amit Dekel, Ran Ilan Ber, Lior Wolf, Yaron Oz. **Semi-supervised learning of partial differential operators and dynamical flows**. *UAI 2023*, PMLR 216:1785–1794。

- 机制：时间超网络产生 FNO 权重；训练含终点、初始恒等、二段组合及多段到终点损失，均为软约束。
- 证据：1D 广义 Burgers / Chafee–Infante、2D Burgers、2D/3D Navier–Stokes；单一终点标签得到中间时刻预测，另有组合损失消融。1000 训练 / 100 测试样本。未在全文找到 noise 实验；不应外推为抗噪结果。
- 界限：超网络、激活、额外复合调用一起变化，报告精度不能直接归因于半群律。Lemma 4.1 的生成元辨识论证需谨慎，见下方独立核查。

[出版页与作者](https://proceedings.mlr.press/v216/rotman23a.html)；[全文 §3–5，尤其 PDF p.3 /印刷1787](https://proceedings.mlr.press/v216/rotman23a/rotman23a.pdf)；[官方代码](https://github.com/rotmanmi/Semi-Supervised-Learning-of-Dynamical-Flows)。

### 3. SINGER — 高维函数表示的参数空间随机演化

Mingquan Feng, Yixin Huang, Weixin Liao, Yuhong Liu, Yizhou Liu, Junchi Yan. **SINGER: Stochastic Network Graph Evolving Operator for High Dimensional PDEs**. *ICLR 2025*。

- 机制：子网络表示 `u(x)`，GNN 演化其权重，结合随机扰动；物理残差监督。论文在连续随机模型下主张强/弱半群性质及条件性稳定性，实际使用 dropout 和数值积分。
- 理论：Theorems 1–5 有可微、Lipschitz、噪声强度等假设；不等于离散实现自动满足全部条件，更不是物理解误差上界。
- 证据：8 种 PDE，5/10/15/20 维；含去噪声与非图结构对照。对照 PINO 被改造成网络权重演化器，并非标准网格 PINO；NODE 和 GNN 的宽度也不同。每个新初值需要拟合初始子网络。
- 我们的缺口：这是高维、已知方程残差、网络参数流问题；不能把其精度直接与我们 1D 网格、快照监督模型对比。噪声主要是训练扰动，而非观测误差。

[ICLR 出版页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5b288823575bb29654b0953a251e933b-Abstract-Conference.html)；[全文 §3.2–3.3、§4、Appendix B](https://proceedings.iclr.cc/paper_files/paper/2025/file/5b288823575bb29654b0953a251e933b-Paper-Conference.pdf)；[官方代码](https://github.com/FengMingquan-sjtu/SINGER)。

### 4. DUE — 适合建立可复现基线的成熟工具

Junfeng Chen, Kailiang Wu, Dongbin Xiu. **DUE: A Deep Learning Framework and Library for Modeling Unknown Equations**. *SIAM Review* 67(4):873–902, 2025；DOI 10.1137/24M1671827；arXiv:2504.10373。

- 延续流映射学习，提供 ResNet、gResNet、OSG-Net、多步损失、GDSG；dual-OSG-Net 用两个网络和时滞门控处理跨尺度刚性问题。
- 已把部分观测的历史窗口、多步训练和噪声数据处理作为实际工作流。摆例使用 5%/10% 相对噪声；另有真实耦合振子数据；这些并非“半群约束单因素”噪声实验。
- Burgers 示例使用模态降维；NS 使用 OSG-FNO。说明“加入记忆、多步损失、模态模型”也已有基础，研究须精确说明超越点。
- 这是教育与软件框架论文，适合复现入口，不是所有方法同预算排名，也没有证明任意有限训练数据下的预测优势。

[正式论文](https://epubs.siam.org/doi/10.1137/24M1671827)；[可读全文 §2.5、§3、§5.1–5.6](https://arxiv.org/html/2504.10373v1)；[官方代码](https://github.com/AI4Equations/due)。

### 5. CFO — 2026 年值得加入的连续时间强基线

Xianglong Hou, Xinquan Huang, Paris Perdikaris. **CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators**. *ICLR 2026*；arXiv:2512.05297。

- 机制：轨迹样条和随机插值产生速度目标，直接回归 `Nθ(t,u)`，避免训练反传穿过 ODE 求解器；推理再积分。显含物理时间，因此一般是二参数演化族，不应称为自治一参数半群。
- 理论：Appendix A.2 在正则性和理想总体最优速度场下，得到网格时刻边缘分布运输；不保证真实生成元、个体轨迹或长期外推。
- 证据：Lorenz、Burgers、2D diffusion–reaction、shallow water；不规则保留 25%/50%/100% 时刻。报告相同骨干 NFE/实测耗时，以及 Neural ODE、PDE-Refiner 对照。
- 界限：主 AR 对照使用 teacher forcing；需补多步/pushforward 等训练对照。插值噪声消融不等于观测噪声鲁棒性；高阶差分速度目标对真实噪声的表现需要另测。

[ICLR2026 出版页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8bfbf4ec87e1e331f0b1adc483b53b6b-Abstract-Conference.html)；[全文 §4–5、Appendix A.2/A.4](https://arxiv.org/html/2512.05297)；[官方代码](https://github.com/shannon-hou/CFO_official)。

### 6. Diffeomorphic Neural Operator Learning — 更强物理结构的示例

Seth Taylor, Alex Bihlo, Jean-Christophe Nave. **Diffeomorphic Neural Operator Learning**. arXiv:2508.06690, 2025。检索未核实正式发表，按预印本标注。

- 机制：预测空间微分同胚并作用于初始场，复合发生在坐标映射空间；对适用的平流几何，结构来自群作用。并非一般耗散 PDE 的通用硬半群网络。
- 理论：给出几何表示与跨分辨率性质；数值实现还依赖映射插值及微分同胚条件，不能把抽象几何结论直接当作任意网络的精确保证。
- 证据：不连续场输运及 Euler 湍流能量级联；FNO-maps 与 FNO-fields 比较，两者输入五帧，但输出通道和损失有差异。作者将实验称为概念验证；映射历史存储随步数增长。
- 启发：针对方程选择确切的守恒/几何约束，比单一组合恒等更有辨别力。该结构不能直接替代我们黏性 Burgers 的耗散项。

[全文 §3–4](https://arxiv.org/html/2508.06690v1)。文中给出的 [代码地址](https://github.com/seth-tl/Diff_flow_learning) 本次未成功打开，不能声称已核实可用。

### 7. Chernoff / envelope neural operators — 2026 年理论边界

Jonas Blessing, Philipp Schmocker, Alessandro Sgarabottolo. **Neural operators approximate strongly continuous convex monotone semigroups**. arXiv:2609.02727v1, 2026-09-02。预印本。

- 学习 Chernoff 一步算子，再用稳定迭代逼近强连续凸单调半群；特化 envelope 网络把控制参数上的上确界替换成有限可学习候选的最大值。
- Theorems 3.5–3.6 给出加权 Hölder 空间中的存在性和有限时间一致逼近；§4 将覆盖数、一步误差、步长关联，形成特定结构的速率。
- 假设涵盖一致性、稳定性、正则性与受限算子类别。并未证明 SGD 找到所需网络，也不是任意非线性 PDE 的无条件定理。
- §5 是一维半线性 PDE splitting、随机控制、模型不确定性示例，主要说明理论结构可实现；不是大规模 PDE 公共榜单。
- 我们的启发：将目标改成明确生成元类别的稳定性、辨识性或误差分解，比再次证明泛泛的神经流半群性质有意义。

[全文 Theorems 3.5–3.6、§4.1–4.2、§5](https://arxiv.org/html/2609.02727v1)；[官方代码](https://github.com/sgarale/chernoff_neural)。

## 独立数学核查：单一时滞不足以辨识生成元

以下具体反例是我们的核查，正式写入论文前应再次独立审阅；不作为对 Rotman 实验结果的否定。这类 system aliasing 并非新的研究问题；已有 [Zeng et al., A Sampling Theorem for Exact Identification of Continuous-time Nonlinear Dynamical Systems](https://arxiv.org/abs/2204.14021) 研究采样频率与连续时间系统辨识条件，本分工仅核对该文书目与摘要，不对其定理作额外总结。

Rotman 正文 §3 / Eq.(1) 给出 2π 周期、自治微分算子、光滑有界初值唯一正则解等条件。Lemma 4.1 位于 PDF 第 3 页（印刷页 1787），要求完整初值集上 `Φ_T` 正确且满足组合律，然后断言所有时间均正确。其式(10)后的推导从两个算子的相同 p 次复合推出根相同；正文未补唯一根、谱带宽或已知小时间生成元条件。

在周期圆上取 `A_c = ν ∂xx + c ∂x`，`ν > 0`。Fourier 模态 k 上的半群乘子为 `exp(-ν k² t) exp(i k c t)`。将 c 换成 `c+2π/T` 后，在 t=T 的所有模态上完全一致，在一般中间时刻不同。二者均连续、自治、光滑化、L² 能量耗散；它们在整个初值空间拥有同一终点映射。

若要求非线性也可构造：令 `B_t` 为周期黏性 Burgers 流，`R_a` 为平移。Burgers 平移等变，所以 `R_(ct) B_t` 与 `R_((c+2π/T)t) B_t` 都是半群，分别生成 `ν u_xx - u u_x + c u_x` 和再加 `(2π/T)u_x`；t=T 相同，中间不同，均保持能量耗散。

**对本项目的结论：**半群残差为零并不足以辨识真实生成元。多时滞、足够短时间信息、物理参数约束和时空分辨率，都应作为可辨识性条件明确写出。多个时滞也不能一概保证唯一；若共享基本时格，仍可能出现相应混叠。

## 建议综述后的研究决策

- 先列齐同任务对照：直接时间条件网络；Deep-OSG 软组合正则；自治积分生成元；CFO 式速度监督；相同物理先验下的简单数值/参数模型。
- 研究问题改写为：“在可辨识、足够闭合的状态与固定工作量下，硬自治约束相对软组合约束何时提高噪声/稀疏数据下的统计效率？”这只是候选问题；本次检索没有证明其无人研究。
- 同时记录有效参数、RHS/NFE、训练耗时、推理耗时、训练数据量、空间投影误差、测试时滞范围。区分观测噪声与训练扰动。
- 暂不据任何一篇的绝对误差宣布领先，也不以更低组合缺陷替代长时间预测精度。

## 额外发现，尚未作为主证据

NOAH: Neural Operator Algebra in Hilbert Space，*Electronics* 15(18):4215，2026-09-16，作者出版页声称在 FNO 上添加投影约束的平流/扩散/反应生成元。[出版页](https://www.mdpi.com/2079-9292/15/18/4215)。本分工未精读，因此仅列后续核查线索。

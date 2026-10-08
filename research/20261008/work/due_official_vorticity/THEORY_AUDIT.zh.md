# 当前官方基线与“组合正则”的理论核对

核对日期：2026-10-04。官方 DUE checkout：`1e629104a12a9c3e3064996cbd4e301cec641ad8`，本次只读检查，工作区干净。本文说明方法，不报告尚未回收验证的训练效果。

## 1. 当前跑的模型是什么

官方 `VorticityOSG.py:9` 实例化 `osg_fno2d`。令 $z=A(u)$ 为训练数据确定的状态仿射归一化，$\tau(\Delta)$ 为归一化时间，则实际网络为

$$G_\theta(z,\Delta)=z+\Delta\,F_\theta(z,\tau(\Delta)),\qquad \widehat S_\Delta=A^{-1}\circ G_\theta(\cdot,\Delta)\circ A.$$

这里 $F_\theta$ 是 FNO：傅里叶空间的可学习卷积与逐点网络组成的算子网络。当前配置为 4 个块、宽度 20、两个方向各 12 个模式。它直接输出一段时间后的状态增量，允许增量网络依赖这段时间；没有通过 ODE 求解器推进该网络。数学上 $\Delta=0$ 时恒等是结构性质，任意两个正时间的组合关系则不是。当前 `multiscale=False`。

旧仓库 Burgers 的自主分支学习空间离散生成元 $\dot u=F_\theta(u)$，再用 RK4 积分。其**连续精确流**在存在唯一解的范围内满足半群律；实际 RK4 映射仍有时间离散误差。两者共享半群动机，但不是同一模型，也不能把旧模型的结构结论移给当前 FNO。

代码依据：[官方入口](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/examples/VorticityOSG/VorticityOSG.py#L9)、[FNO forward](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/networks/fno.py#L81)、[配置](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/examples/VorticityOSG/config.yaml)。旧实现：run_burgers_semigroup_screen.py（历史文件路径；原始产物已按保留策略舍弃）。

## 2. 半群定律什么时候成立

$$S_0=I,\qquad S_{s+t}(u)=S_t(S_s(u))=S_s(S_t(u)),\qquad s,t\geq0.$$

含义是：从同一状态直接前进 $s+t$，应等于先前进 $s$、再前进 $t$。适用前提包括自治动力学、状态足以确定后续演化、解存在且唯一，以及演化仍处于所讨论的状态域。若只保证局部存在，关系仅适用于存在时间内；显含绝对时间的系统通常需要二参数演化算子。DUE 的涡度案例外力 $f(x)$ 不显含时间。[DUE §2.2–2.3，式 (6)、(14)–(16)；§5.6，式 (37)](https://arxiv.org/html/2504.10373v1)

我们的解释：只观测部分变量或粗网格，并不会自动继承完整状态的半群。两个相同的观测状态可能对应不同的未观测变量和未来。Deep-OSG §4.1 也明确采用“适当空间近似可形成自主离散系统”的假设。[Deep-OSG §4.1](https://arxiv.org/html/2302.03358v2)

## 3. “组合正则”具体惩罚什么

从一个随机状态 $z$ 出发，构造三种预测：

$$a=G_\theta(z,s+t),\qquad b=G_\theta(G_\theta(z,s),t),\qquad c=G_\theta(G_\theta(z,t),s).$$

组合正则惩罚 $a$ 与 $b,c$ 的差异。它不需要这三个预测对应的额外真值标签。论文 GDSG（global direct semigroup-informed）以平方 L2 展示正则和总目标：[DUE §3.2，式 (32)–(33)](https://arxiv.org/html/2504.10373v1)。

**当前官方配置实际用相对 L2，并非平方 L2。** 对一个批次，定义

$$d(a,b)=\frac1B\sum_{i=1}^{B}\frac{\|a_i-b_i\|_2}{\|a_i\|_2},\qquad L_{\rm fit}=d(y,G_\theta(x,\Delta)),$$

$$R_{\rm SG}=\tfrac12\{d(b,a)+d(c,a)\},\qquad L_{\rm GDSG}=\frac{L_{\rm fit}+\lambda R_{\rm SG}}{1+\lambda}.$$

范数展平空间与通道，公式要求分母非零。代码中正则分母分别是**复合预测** $b,c$ 的范数，梯度穿过全部分支，没有 `detach`；这些量均在归一化状态空间计算。当前 $\lambda=1,Q=1$。`sg_pairing=0` 分支只优化 $L_{\rm fit}$，因此两臂记录的训练 loss 不能直接横比；应比较同一物理空间测试指标。

官方默认在初始化时为每条训练样本生成 $Q$ 个随机全网格状态（各元素均匀分布于 $[-1,1]$），并独立采样 $s,t\in[t_{\min},t_{\max}]$，随后复用这些样本。和时间 $s+t$ 可超出单步监督区间。这些是当前约束的采样范围，不是对全部状态、全部时间的验证。

代码依据：[PDE_osg 的采样与两个训练分支，38–61、79–111 行](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/models/pde_osg.py#L38)、[相对 L2 定义，356–360 行](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/utils.py#L356)、[数据归一化，152–163 行](https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/due/datasets/pde.py#L152)。

## 4. 原论文理论保证到哪里

- **Deep-OSG Theorem 1 / §3.3：** 对有限维系统，假设真生成元在域 $D$ 上 Lipschitz，真值和预测的相关轨迹留在可达域内，且网络对真实时间平均增量 $\phi$ 有一致逼近误差 $\varepsilon=\|N_\theta-\phi\|_{L^\infty(D\times[\Delta_{\min},\Delta_{\max}])}$。其证明得到

  $$E_{k+1}\leq e^{L_f\Delta_k}E_k+\Delta_k\varepsilon.$$

  这是依赖**真值逼近误差**的条件性滚动误差界，不是用组合缺陷替代 $\varepsilon$，也没有证明有限样本训练能达到这个一致误差。

- **Theorem 2 / §3.4：** 先假设 $K$ 种时间划分所得末态两两相距至多 $\epsilon$（式 (25)），再证明其预测误差的标准差不超过 $(K-1)\epsilon/K$。它约束跨划分的离散程度，不保证平均预测误差小；式 (25) 也不是由有限采样正则自动推出的。

来源：[Deep-OSG §3.3，Theorem 1 与式 (24)；§3.4，式 (25) 与 Theorem 2](https://arxiv.org/html/2302.03358v2)。这些结论不能直接升级为当前神经网络的普遍稳定性、守恒、耗散、PDE 收敛或更高精度保证。

## 5. 为什么“组合缺陷小”仍可能预测错

下面是我们用于解释逻辑边界的简单反例，不是论文中的实验。真实系统为 $\dot u=-u$，故 $S_t(u)=e^{-t}u$；模型却取 $\widehat S_t(u)=u$。模型满足恒等和精确半群律，组合缺陷为零，但当 $u_0=1$ 时预测绝对误差为 $1-e^{-t}$。

因此，组合正则只提供“不同时间拆分应自洽”的先验。数据拟合负责把自洽的模型约束到目标动力学；有限采样覆盖、优化以及滚动中经过的状态会影响实际效果。**当前完整基线对照要检验的是：这项训练约束在同一任务、同一模型骨干下是否改善测试预测，而不是预设它必然有效。**

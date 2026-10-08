# 结构保持的神经 PDE 演化：定向文献笔记

检索截止：2026-10-03。只使用作者论文、会议/期刊页面和作者代码。以下是定向阅读，不是系统综述；没有运行这些作者代码，也不声称独立复现其结果。

## 对本项目最重要的结论

1. **连续动力学结构、数值更新结构、真实物理量、预测精度是四件事。** 自治 ODE 产生半群，并不意味着数值离散在任意时间划分下严格满足半群；所学能量严格守恒，也不等于真实能量轨迹正确。
2. 2019—2020 年已出现从能量函数导出向量场、在离散更新中保证守恒/耗散的网络。单独增加一个结构恒等式，现在通常不足以构成强贡献。
3. 较新的工作已经比较稀疏观测/超分辨率、统计吸引子、离散积分精度、长时间稳定性、可辨识性和计算成本。对于我们，值得提出的问题是：**哪一种结构，在何种观测/噪声/分辨率条件下改善真实动力学预测，并付出多少成本？**
4. 最新 GENERIC-FNO 也出现结构满足而精度不佳的情况；其 v6 明确讨论了恒等预测和物理耗散不足。我们目前的负结果并不反常，但必须据此改变问题定位。

## 六篇核心论文

### 1. Hamiltonian Neural Networks — Greydanus, Dzamba, Yosinski；NeurIPS 2019

- [论文](https://proceedings.neurips.cc/paper_files/paper/2019/file/26cd8ecadce0d4efd6cc8a8725cbd1f8-Paper.pdf)；[作者代码](https://github.com/greydanus/hamiltonian-nn)。读取：正式摘要、作者仓库方法说明、hnn.py。
- 以神经网络表示标量 Hamiltonian，经辛梯度得到向量场。连续精确流守恒的是学习到的能量；通用 RK4 数值推进仍会引入误差。主要是低维机械系统，不是 Burgers 神经算子。
- 借鉴：把“向量场具有结构”和“离散预测具有结构”分开验证。本文不足以支持我们在含噪 PDE 上的独特抗噪论断。

### 2. Deep Energy-Based Modeling of Discrete-Time Physics / DGNet — Matsubara, Ishikawa, Yaguchi；NeurIPS 2020

- [正式论文](https://proceedings.neurips.cc/paper_files/paper/2020/file/98b418276d571e623651fc1d471c7811-Paper.pdf)；[作者代码](https://github.com/tksmatsubara/discrete-autograd)。读取：§3.3–3.4、§4 的比较设定及正式摘要。
- 自动离散微分使学习能量满足离散链式法则；配合指定的反对称/半负定算子，构造守恒或耗散的隐式更新。实验为简化起见假定结构算子 G 已知。
- 关键边界：论文也为节约计算采用显式预测；显式替换后不能沿用隐式离散更新的严格守恒保证。比较 NODE/HNN++ 与不同积分器，不能把差异全部归因为网络结构。
- 借鉴：若我们以后要宣称离散物理保证，DGNet 是必要基线；当前积分加密误差很小，暂不优先重做积分器。

### 3. Learning Chaotic Dynamics in Dissipative Systems / MNO — Li et al.；NeurIPS 2022

- [正式页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6ad68277e27b42c60ac228c9859fc1a2-Abstract-Conference.html)；[全文 arXiv v2](https://arxiv.org/html/2106.06898v2)；[作者代码](https://github.com/neuraloperator/markov_neural_operator)。arXiv 标题为 Learning Dissipative Dynamics in Chaotic Systems；读取 §2–4、定理1及作者仓库。
- 学固定时间步的状态映射；用 Sobolev 损失增强空间导数信息，在吸引子外的壳层加入收缩数据，并可在更远处用硬后处理保证耗散。它不是保证所有实数时间的半群恒等式。
- Lorenz、KS、二维 Kolmogorov 流；长期评估能谱、自相关、分布和耗散率。有限步紧集上的存在性逼近定理，不是已训练模型的无限时精度保证。
- 借鉴：长时间目标应有物理/统计指标，而非只看组合缺陷。其扰动稳定性不能等同于有噪监督训练优势；本次未核实噪声强度扫参与同算量比较。

### 4. Energy-consistent Neural Operators for Hamiltonian and Dissipative Partial Differential Equations / ENO — Tanaka et al.；AISTATS 2025

- [正式页面](https://proceedings.mlr.press/v258/tanaka25a.html)；[最终21页PDF](https://raw.githubusercontent.com/mlresearch/v258/main/assets/tanaka25a/tanaka25a.pdf)；[作者代码](https://github.com/yusuk-e/eno)。读取最终版 §3–6、附录 B.2/B.4/B.6。本地 PDF 和提取文本已保存。
- 同时学解算子和能量密度，惩罚时间导数与能量梯度流的偏差；这是**软约束**。最终版从候选 G 中按验证误差选择；不要沿用2024预印本“G必须已知”的旧限制。
- KdV、Cahn–Hilliard及附录 Allen–Cahn/二维 Cahn–Hilliard；核心任务是低分辨率训练到高分辨率预测，五次重采样。比较 Vanilla NO、DeepONet、FNO，以及使用真实能量的 EnerReg。
- 代价：作者报告训练最多约为 Vanilla NO 的20倍；并非同训练算量优势证明。没有在所读最终版发现独立含噪监督消融。适合做低数据/低分辨率软约束基线。

### 5. UEPI: Universal Energy-Behavior-Preserving Integrators for Energy Conservative/Dissipative Differential Equations — Celledoni et al.；NeurIPS 2025

- [正式页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e467e15cbcbd7b5ecc814cfc27c4ea9-Abstract-Conference.html)；[PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/3e467e15cbcbd7b5ecc814cfc27c4ea9-Paper-Conference.pdf)；[代码ZIP](https://proceedings.neurips.cc/paper_files/paper/2025/file/3e467e15cbcbd7b5ecc814cfc27c4ea9-Supplemental-Conference.zip)。读取 §3–5及附录 D.6；本地35页PDF已保存，未解压代码。
- 对已给能量和结构算子，学习离散梯度的反对称修正，保持离散守恒/耗散。紧相空间上给出受约束积分器的通用逼近定理；它学习数值方法，问题设定不同于未知生成元识别。
- 测试非线性保守/耗散 ODE 与 KdV，比较二阶、半步长及四阶离散梯度方法，并报告含制数/训练的总成本。隐式求解代价仍须计入。没有在全文找到实质性的含噪实验。
- 借鉴：现在的前沿已要求把结构与精度/成本放在一起研究。不要仅以通用逼近定理和一个能量图作为研究终点。

### 6. GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators — Sulskis, Ravi；2026预印本

- [arXiv记录](https://arxiv.org/abs/2606.08343)；[v6全文，2026-09-16](https://arxiv.org/html/2606.08343v6)；[作者匿名代码](https://anonymous.4open.science/r/GENERIC-FNO-F414/)。记录注明 TMLR under review，不能写成已发表；读取 §4–6、表1/7/19、附录 B.3/E。
- 通过投影和 Fourier 乘子保证 GENERIC 退化恒等式；未强制 Jacobi 恒等式。保证的是连续动力学中的学习能量/熵；显式推进存在漂移。
- 关键反例：200步热方程/Burgers预测有界，但误差接近恒等预测，真实物理耗散不足；该长程表每格仅一次运行。去除归一化在1D改善明显，在2D却妨碍优化。短期基准用五种子，不能把两套证据混写。
- 成本：128²设置推理约4.4倍、训练步约10倍；有更大FNO的近似算量对照。可作为近期趋势和问题来源，不能当作成熟普适方案；本次未找到观测噪声抗性实验。

## 附加线索：可辨识性，不直接作为 Burgers 精度基线

[Identifiable learning of dissipative dynamics](https://www.nature.com/articles/s41467-026-76255-3)，Zhu、Soh、Pavliotis、Li，Nature Communications 17:9460，2026-08-06；[代码](https://github.com/MLDS-NUS/UIDD-jax)。读取摘要、Results的架构和定理1。对具有唯一不变分布的非退化SDE构造势能/可逆不可逆分解，强调表示的可辨识性。实验含聚合物粗粒化和SGLD。其过程噪声不是观测噪声；不能转述为我们的噪声0.01监督实验已有文献保证。

## 建议提炼成研究路线的三个问题

- **结构到底作用在哪里？** 自治、边界、真实能量、质量、数值更新应分别列出保证条件和违反指标。对我们当前结果，先不要将自治等同于耗散或稳定性。
- **在什么困难条件下有收益？** 噪声/稀疏采样/粗粒化分开；采用同有效参数量、同训练算量的无约束和软约束基线，加入恒等/零预测以及已知物理生成元。
- **收益来自约束还是训练方案？** 初态分布与滚动状态分布、一步与多步训练、归一化、积分器、数据参考精度都应单独消融。本文献包支持这些比较的必要性，并未替我们证明任何新机制。

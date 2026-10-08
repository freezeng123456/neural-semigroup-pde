# 官方 DUE VorticityOSG：完整流程审计

日期：2026-10-04。只读审计；未训练、未修改 DUE checkout。

源码：`AI4Equations/due`，固定提交 `1e629104a12a9c3e3064996cbd4e301cec641ad8`。本地目录 `/Users/Apple_501/Documents/Codex/2026-10-03/cha/work/literature_baselines/due`，审计时已跟踪文件无改动。

## 1. 这次复现的对象

目标是**直接运行作者 VorticityOSG 的完整500轮数据处理、网络、训练与预测流程**。不沿用前一阶段的局部小MLP、Burgers数据、MSE、in-range配点或验证选模。

官方入口依次调用 `pde_dataset_osg.load → osg_fno2d → PDE_osg.train → save_hist → mynet.predict → pde2dregular_evaluate`。`network.py`虽然被import，其中的自定义`res_fno`并未被实例化；实际模型为`due.networks.fno.osg_fno2d`。来源：VorticityOSG.py:1（历史文件路径；原始产物已按保留策略舍弃）。

对应论文为 [DUE论文§5.6，式(37)、图14–15](https://arxiv.org/html/2504.10373v1#S5.SS6)：二维周期不可压Navier–Stokes涡度，黏性系数1e-3、周期外力；100条训练轨迹、64×64空间网格、随机时滞[0.5,1.5]，OSG-FNO，GDSG的λ=Q=1，500轮、batch20。测试100条未见轨迹，固定时滞1，考察到t=100；图14展示t=60/100涡度，图15展示训练损失及测试平均相对L2误差曲线。论文没有在该节给出可逐位核对的单个表格误差值。

**对应范围：**这是DUE论文的官方示例复现，不是2023年Deep-OSG论文所有实验的完整复现。固定checkout晚于论文arXiv v1；论文没有给出原始实验的源码commit、完整运行环境或逐位指标，因此不能宣称重建原作者同一条优化轨迹。当前默认入口只训练GDSG模型；论文图15还比较无组合约束模型。只完成默认一臂，可以复核GDSG曲线和预测，不能声称已复现该图的全部比较结论。

**论文时间标签与公开数据需要区分：**实际公开测试MAT包含100个状态和99个单位时滞。相对首个提供状态，预测跨度为99；文件没有绝对起始时间。保持作者原数据和99步预测，使用相对时间0–99标记结果；不能自行添加第100步，或直接把最后状态称为绝对t=100。详见第8节数据回执。

## 2. 固定官方配置与计算路径

| 项目 | 官方配置/行为 |
| --- | --- |
| seed、dtype | 0、single/float32 |
| 任务 | 2d_regular，problem_dim=1，multiscale=False |
| 训练样本 | 每条轨迹随机取25个不重复起点，对应相邻快照对 |
| 网络 | osg_fno2d，modes1=modes2=12，4块，width20，GELU |
| 训练 | 500 epochs，batch20，Adam，lr=.001，默认无weight decay |
| 调度 | CosineAnnealingLR，每个训练batch后step |
| 组合约束 | sg_pairing=1，sg_weight=1，默认随机盒内状态和独立子时滞 |
| 损失 | `rel_l2`，包括监督与两个组合残差；不是MSE |
| 保存 | 训练目标下降时覆盖`model/model`；完整训练结束后写history |
| 官方测试 | 直接使用内存中的最终模型；没有重新载入最低训练目标检查点 |

配置来源：config.yaml:1（历史文件路径；原始产物已按保留策略舍弃）。Adam与scheduler实现：utils.py:396（历史文件路径；原始产物已按保留策略舍弃）。

实际MAT已确认100条训练轨迹、每条50个可选相邻转移。按官方25个burst，训练集应为2500对；每轮125个batch，完整500轮应为**62,500次optimizer更新**，cosine的T_max也为62,500。运行时仍需核对官方loader生成的数据与实际更新数。Q=1的每个batch有6次网络forward，其中5次用于GDSG；对应375,000次批次forward或7,500,000次按状态样本累计的网络评估，不包含预测。

网络确实包含空间傅里叶卷积：每块保留两个频谱区域并以复权重混合通道，再经1×1卷积MLP与线性支路相加。最后计算`x0 + dt * increment`，没有RK积分器或固定物理黏性项。来源：fno.py:10（历史文件路径；原始产物已按保留策略舍弃）、fno.py:52（历史文件路径；原始产物已按保留策略舍弃）。

静态计数：460,800个复数权重元素和6,861个实数元素。作者`count_params()`按`numel()`计为**467,661**；把每个复数元素计作两个实自由度时为**928,461**。两种口径必须标明，运行时应按实际参数dtype和shape再次核对；不能与Stage A的纯实参数直接混称同口径。

## 3. 官方数据处理与随机性

### 数据加载与归一化

训练MAT需要`trajectories`、`dt`、`coordinates`。二维数据预期维度为`(N,H,W,D,T)`，坐标为`(H,W,2)`。先使用**完整训练轨迹**估计每个物理变量的全局min/max，归一化到[-1,1]并clip，再选burst。时间范围同样由训练dt的实际min/max得到；不是强行写死理论端点0.5/1.5。测试状态保持物理单位读入，预测器再用训练min/max归一化初态、预测后反归一化。来源：datasets/pde.py:39（历史文件路径；原始产物已按保留策略舍弃）、datasets/pde.py:93（历史文件路径；原始产物已按保留策略舍弃）、datasets/pde.py:130（历史文件路径；原始产物已按保留策略舍弃）、datasets/pde.py:152（历史文件路径；原始产物已按保留策略舍弃）。

不能换成只使用训练初态归一化、每轨迹归一化、从测试集估计范围，或在读取后改变网格/时滞。坐标会归一化但没有作为该FNO的输入通道；输入是状态与时滞。

### burst与GDSG随机数

- 数据模块import时执行`np.random.seed(0)`；dataset构造器没有使用传入的seed。因此默认seed0入口可复现，但改YAML seed并不能改变所有随机部分。
- 每条轨迹的25个起点用NumPy抽取，重复则重抽整组；每次完整加载只生成一组burst。
- FNO模块import时执行`torch.manual_seed(0)`；`osg_fno2d.__init__`没有调用`set_seed(config['seed'])`。
- PDE trainer构造时再次重置Torch seed0，随后一次性生成固定的GDSG状态与时间，再建立`shuffle=True`的DataLoader。配点不是每轮重采样。
- PDE的`set_seed`同时设`cudnn.benchmark=True`与`cudnn.deterministic=True`；没有启用全局deterministic algorithms。因此固定seed不是跨硬件/库版本逐位一致的保证。

来源：datasets/pde.py:1（历史文件路径；原始产物已按保留策略舍弃）、datasets/pde.py:106（历史文件路径；原始产物已按保留策略舍弃）、fno.py:3（历史文件路径；原始产物已按保留策略舍弃）、models/pde.py:134（历史文件路径；原始产物已按保留策略舍弃）。正式运行需用全新Python进程，并保持与官方入口相同的导入、加载、构造顺序；预检不能在同一进程消耗随机数后直接转入正式训练。

## 4. 必须保留的损失细节

作者`rel_l2_norm(true,pred)`定义为每个样本的`||true-pred||₂ / ||true||₂`，再对batch取平均，没有平方、也没有epsilon。官方GDSG调用顺序是`loss_func(pred012,pred2)`与`loss_func(pred021,pred2)`，所以分母是**组合预测的范数**。改为直接预测范数，或者对称分母，会改变方法。来源：utils.py:356（历史文件路径；原始产物已按保留策略舍弃）、models/pde_osg.py:104（历史文件路径；原始产物已按保留策略舍弃）。

默认GDSG状态是每个网格点独立的[-1,1]均匀值；两个子时滞分别独立抽样，先还原物理时长相加，再重新归一化总时长。因此总时长可超过训练时滞范围。必须保留这一作者默认行为，不能套用Stage A的“平滑训练状态+总时长in-range”采样。来源：models/pde_osg.py:38（历史文件路径；原始产物已按保留策略舍弃）。

## 5. 预测、指标与保存模型的风险

### 官方主输出

`predict`以100个测试初态为一批，递归预测全部测试dt；返回初态和后续预测，形状必须与truth相同。它不做测试期间clip。必须使用实际test_dt决定时间轴，不能仅凭文件名推断100步或101个快照。来源：fno.py:99（历史文件路径；原始产物已按保留策略舍弃）。

`pde2dregular_evaluate`生成：

- `pred.mat`：全部物理单位预测轨迹。
- `rel_err.csv`及`rel_err.png`：每个时刻先算每条轨迹的空间相对L2，再对测试轨迹平均；这对应论文图15右图的指标。
- stdout的`Relative error is`：先把每条轨迹的空间和**全部时间**合并后算相对L2，再平均。这个单值**不是**`rel_err.csv`的时间均值，不能互换。

来源：utils.py:125（历史文件路径；原始产物已按保留策略舍弃）。有truth时官方函数不输出图14那样的涡度快照图；从`pred.mat`补绘状态图应标为同一预测的后处理，不重新选模型。公开数据目前只能可靠标注相对首态的时间，例如elapsed=60和99；论文绝对t=60/100与数组索引的对应关系尚无来源证明。

### 最低训练目标模型与最终模型不同

`PDE_osg.train`在epoch在线批次目标的均值下降时保存整个模型对象到`model/model`。没有独立验证集、没有保存epoch/optimizer/scheduler/RNG状态，也没有自动恢复训练。训练结束的示例入口直接对原`mynet`预测，因此主输出来自**最终epoch500模型**，不保证等于磁盘上的最低训练目标模型。来源：models/pde_osg.py:120（历史文件路径；原始产物已按保留策略舍弃）、VorticityOSG.py:12（历史文件路径；原始产物已按保留策略舍弃）。

外部运行包装应额外归档最终模型、最终optimizer/scheduler/RNG和实际轮次，保留作者原`model/model`，并标明二者身份。不得悄悄改为best模型测试或按测试曲线选模型。仅有`model/model`存在不证明训练完成500轮；`training_history.csv`也是训练全部结束后才保存。

## 6. 启动阻断与最小兼容处理

**静态审计尚未发现默认seed0、single、multiscale=False配置必然触发的模型/训练逻辑异常。** 这不等于CUDA端到端已验收。以下运行前条件需要实证检查：

| 项目 | 当前判断与可接受处理 |
| --- | --- |
| 相对路径 | 入口固定读取当前目录的config和MAT，并import同目录network.py；外部工作目录须提供官方原文件，设置pinned checkout的PYTHONPATH。允许改变输出目录，不改内容。 |
| 依赖 | 完整import需要Torch、NumPy、SciPy、PyYAML、Matplotlib；setup.py固定NumPy1.26.4。允许独立环境安装缺失依赖、设置MPLBACKEND=Agg；不得用抽取AST替代完整包。 |
| MATLAB格式 | 训练loadmat对v7.3有mat73回退，测试loadmat没有同样回退。两份官方MATLAB5.0文件已经实际载入，数组与官方loader契约一致；本数据无需v7.3兼容修复。 |
| GPU内存/FFT | 完整batch20含6条前向计算图，预测批次为100；短预检须覆盖二者。显存不足时先报告；不能静默缩小训练batch或空间网格。 |
| 归一化和relative L2分母 | `vmax>vmin`、`tmax>tmin`、非零目标范数需要核验。没有epsilon与非有限值停止保护；出现NaN不得在结果中补零或跳过。 |
| 路径写入失败 | trainer的mkdir异常被吞掉。运行前确认新输出目录可写、没有旧model混入；不要等训练保存时才发现。 |
| checkpoint加载兼容 | 官方存whole-model pickle，加载需要相同类定义；恢复审计时使用已校验可信本地文件、明确map_location和适合当前Torch的加载方式。默认示例训练后的predict并不调用torch.load，所以此项不是其训练启动必然错误。 |
| 中断恢复 | 原checkpoint不含完整续训状态，不得从它加载后声称接着原500轮无缝续训。任何外部完整快照方案须另有状态一致性记录。 |

源码依据：setup.py（历史文件路径；原始产物已按保留策略舍弃）、models/pde.py:50（历史文件路径；原始产物已按保留策略舍弃）、networks/nn.py:15（历史文件路径；原始产物已按保留策略舍弃）。本次没有实施任何兼容修改。

有界预检只训练第1轮时，长程递归预测可能尚未稳定。应分别记录训练前向/反向与参数更新是否有限、文件重载是否一致、完整预测路径是否执行，以及早期预测是否发散；不能把epoch1的预测误差或晚程非有限值直接当作500轮方法的科学结论。任何非有限预测都保留原值并明确标注，不能伪造为通过精度验收。

## 7. 完整500轮的验收条件

1. **来源一致：**固定官方commit、逐文件源码哈希、原config哈希、公开MAT下载来源和文件哈希、实际环境与GPU。未改变网络、损失、抽样、归一化或训练预算。
2. **数据一致：**真实MAT的shape、dtype、min/max、dt、finite检查完成；25burst规则及实际2500对训练数据可追溯；保存归一化元数据及分组后数组摘要。
3. **进度真实：**500个完整epoch、每轮125个batch、62,500次optimizer与scheduler更新；500行history全部有限。保留初始/最终参数摘要，证明参数发生变化。
4. **模型身份明确：**保留作者最低训练目标model和额外的最终epoch500模型；官方主评测使用最终模型，保存足够信息能重新加载该模型。
5. **预测齐全：**100条测试轨迹、全部官方test_dt预测完成；无非有限值；`pred.mat`形状与truth逐维一致，首帧与初态一致到浮点容差；时间轴来自test_dt。
6. **指标可复算：**从归档pred与truth独立重算逐轨迹、逐时刻相对L2，再取轨迹均值，核对`rel_err.csv`；另外复算stdout的时空联合指标并分开标注。报告相对首态elapsed=60/99、完整0–99曲线及已定义汇总，不能只挑表现好的时间段。若以后取得绝对时间原点的来源证据，才补充与论文t60/t100标签的对应关系。
7. **结果完整：**scheduler正常退出、日志完整、history/model/pred/metrics/config/environment全部归档并哈希回收；本地重新加载最终模型并核对一组固定输入预测。
8. **复现结论分层：**完整软件流程运行成功、官方指标可复算、与论文图示趋势一致，是三件事。论文没有公布精确标量阈值，不自创“误差必须低于某值”的过关线；若预测变差或发散，如实报告，不调整默认参数后冒称原版复现。

## 8. 已完成的数据验证与尚待补齐的运行记录

公开作者数据已经下载，来源与完整回执见 [download_manifest.json](data/download_manifest.json)：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| VorticityOSG_train.mat | 93,123,263 | `0ebdd969d52292b4304d3583b3cd432c288397b367f75372d06944d629c5d111` |
| VorticityOSG_test.mat | 181,969,568 | `a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a` |

两文件均为MATLAB5.0，匿名HTTP下载成功且字节数与公开目录相符。这些SHA是本地计算的完整性回执，未发现作者另发的密码学校验值。

[数据验证manifest.json](data/manifest.json)已通过32项检查，包括完整文件哈希、MAT压缩数据读取、字段、形状、数值有限、时滞和状态数量一致、读取后原文件未变，以及训练/测试初态没有精确重复。该检查不能证明两个集合统计独立。

| 字段 | 实际形状 | SciPy默认读取dtype | 取值范围 |
| --- | --- | --- | --- |
| train trajectories | (100,64,64,1,51) | float64 | [-2.1540215015411377, 2.148057460784912] |
| train dt | (100,50) | float64 | [0.5006624504780224, 1.4997749156113407] |
| train coordinates | (64,64,2) | float64 | [0, 0.984375] |
| test trajectories | (100,64,64,1,100) | float64 | [-2.1548023223876953, 2.1645889282226562] |
| test dt | (100,99) | uint8 | 全部为1 |

测试MAT没有coordinates，官方loader复用训练坐标。测试dt的MATLAB元数据class为double，而SciPy默认`mat_dtype=False`读取结果为uint8；官方loader统一转换为配置的float32，因此不需要修改原数据或加载代码。实际loader及CUDA运行环境仍应在预检中记录，不能把本地SciPy数据解析环境等同于训练环境。

测试文件没有绝对起始时间字段。其100状态明确对应99个单位转移；官方`predict`依照`test_dt.shape[1]`执行99步，含初态返回100帧，官方绘图横轴为数组索引0–99。论文t=100的标签目前存在时间原点证据缺口，需要如实保留。

GPU有界预检和完整500轮尚未计入本文件；数据与静态源码审计通过不代表CUDA完整流程已经成功。

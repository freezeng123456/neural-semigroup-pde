# Burgers 与 KS 参考数据

参考数据采用官方 APEBench 场景及配套 Exponax 求解器，模型学习256点周期网格上的闭合数值映射，三种训练目标共享全部数据字节

## 来源与配置

- [APEBench v0.1.1](https://github.com/Ceyron/apebench/tree/dca24a00eeacbe5524a99542f6d5c2bc2deb8da8)，提交 `dca24a00eeacbe5524a99542f6d5c2bc2deb8da8`
- [Exponax v0.1.0](https://github.com/Ceyron/exponax/tree/b39a5edec4b5fde4bd7e7feefa2fa2104242e011)，提交 `b39a5edec4b5fde4bd7e7feefa2fa2104242e011`，这是 APEBench 0.1.1 指定的求解器版本
- 场景实现为 [BurgersSingleChannel](https://github.com/Ceyron/apebench/blob/dca24a00eeacbe5524a99542f6d5c2bc2deb8da8/apebench/scenarios/physical/_nonlinear.py) 与 [KuramotoSivashinskyConservative](https://github.com/Ceyron/apebench/blob/dca24a00eeacbe5524a99542f6d5c2bc2deb8da8/apebench/scenarios/physical/_convection.py)
- 初态和数据生成逻辑遵循 [BaseScenario](https://github.com/Ceyron/apebench/blob/dca24a00eeacbe5524a99542f6d5c2bc2deb8da8/apebench/_base_scenario.py)，分辨率映射使用 [官方 Fourier 映射](https://github.com/Ceyron/exponax/blob/b39a5edec4b5fde4bd7e7feefa2fa2104242e011/exponax/_interpolation.py)

`SOURCE_LOCK.json` 记录11个相关官方源码文件的大小与 SHA-256，以及官方 PyPI 发布包的 URL 和 SHA-256；这11个文件在 PyPI wheel 中的字节已与发布提交逐项核对，清单还从已验证的官方 wheel 锁定全部105个包内 Python 文件，实际运行会再次核对安装后的源码

| 场景 | 周期长度 | 宏步长 | 官方系数 | 热身宏步 |
|---|---:|---:|---|---:|
| BurgersSingleChannel | 1 | 0.1 | convection_sc_coef=-0.125，diffusion_coef=0.0003 | 0 |
| KS Conservative | 60 | 0.1 | convection_coef=-3.6，diffusion_coef=-1.44，hyp_diffusion_coef=-0.4 | 500 |

两场景均保留官方 ETDRK2、2/3 去混叠、16个圆周积分点、圆半径1以及 `fourier;5;true;true` 初态分布，正常标签使用单个宏步内1次求解器步进

求解器在独立 CPU JAX 环境执行，启用 float64 积分，最后写为 float32 物理状态；模型训练环境保持独立

## 数据划分与窗口

每个 PDE 各生成128条训练轨迹、32条开发轨迹、64条封存测试轨迹

训练轨迹包含所给初态及50次转移，开发和测试轨迹各包含所给初态及200次转移，轨迹按照独立固定种子生成，完整初态哈希集合要求跨划分无交集

Burgers 的训练、开发、测试、数值审核种子依次为14001、14002、14003、14004，KS 依次为24001、24002、24003、24004

从128×49个可用连续两转移训练窗口中无放回抽取2000个固定窗口，Burgers 与 KS 的窗口抽样分别使用 PCG64 种子54001、54002

`dataset.npz` 仅保存原始物理状态、步长、窗口索引和坐标时间信息，模型归一化参数只从训练划分拟合，封存测试标签只用于预定最终模型的一次评估

NPZ 使用确定成员顺序与固定 ZIP 时间戳，便于同一参考运行配置的字节复现；清单同时记录每个数组的内容哈希、形状和类型

## 数值审核

生成完整数据之前，先对两个 PDE 各使用4条独立审核初态执行数值审核，两者全部通过才生成训练、开发和测试数据

时间与空间细化均从同一个已热身256点状态开始，高分辨率初态由该状态按官方 Fourier 映射得到，避免分别热身导致 KS 混沌相位不同

时间细化对照每宏步1、2、4、8个子步，空间细化对照256、512、1024点，空间比较均使用每宏步4个子步并投影到共同256点网格

数值门槛在运行前写入代码和意图记录，比较指标为各轨迹未来前8个快照合并后的相对 L2，以审核集最差一条轨迹判定

| 门槛 | 上限 |
|---|---:|
| 时间细化，4子步与8子步 | 0.005 |
| 空间细化，512点与1024点 | 0.005 |
| 标签步进1子步与8子步 | 0.02 |
| 标签网格256点与1024点，均4子步 | 0.05 |
| 下一档细化误差相对上一档 | 1.25倍加1e-6绝对余量 |
| 全200宏步空间均值绝对漂移 | 1e-5 |

全部参考张量须有限，50步和200步细化误差作为长时诊断记录，前8步用于审核门槛

在第0、25、50、100、150宏步额外记录投影后重启的1、2、8步误差，并比较具有同一256点观测的两个细网格状态，分别量化网格步进差异和被投影高频引起的闭合误差

完整数据直接由256点官方映射生成，高分辨率投影只用于审核，投影非闭合效应不会被混入训练标签生成流程

任一审核失败就保存真实审核结果并停止，不自动修改参数、重生成数据或提交训练

## 调用与产物

`python prepare_data.py --output-root /ABSOLUTE/RUN/data --device cpu`

每个 PDE 输出 `dataset.npz`、`DATA_MANIFEST.json` 与 `REFERENCE_AUDIT.json`，根目录输出生成意图和最终回执；已有生成意图时拒绝重新生成，以便先核对已有完整或部分产物

`--audit-only` 只运行独立小规模参考审核，应使用单独输出目录，不写完整训练、开发或测试数据

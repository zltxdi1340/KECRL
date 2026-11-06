# LIBERO source 内部 SPT/SPI 可行性诊断结果（2026-11-03）

后续进度：[同18个500端点的360 episodes配对闭环](libero-source-internal-spt-spi-closed-loop-diagnostic-2026-11-03.md)已按冻结协议执行并独立复核：task0 A/B/C为8/12/12，task1为0/11/1（各60条），B收益集中于个别seed，未形成三seed稳定优势。保持500端点和5步稳定化，validation、held-out、qualification继续封存。以下离线结果和解释边界不变。

**已完成训练、曲线评价和独立审计。** 本轮首次在 LIBERO 上执行多个 source SPI 反馈到隔离候选生成器的上下文条件化 FOMAML 对照。B 相对原始 SPT 的学习曲线面积在两探针各三 seed 均降低，但没有稳定优于普通预训练 C。不能据此接受候选 SPT、声称可靠闭环迁移或进入 Module qualification。

本轮是 source 内部、仅离线的可行性诊断。任务0/1早已暴露，只在当前 fold 的生成器更新中排除，不是正式 held-out；没有环境交互或封存任务数据访问。维持每个探针 Policy 500 updates、现有闭环默认稳定化5步。

## 设置与实际执行

科学设计见[原协议](libero-source-internal-spt-spi-protocol-2026-11-03.md)，执行差异见[确定性修订](libero-source-internal-spt-spi-execution-revision-2026-11-03.md)。[v2配置](../configs/libero-source-internal-spt-spi-diagnostic_v2.json)、[执行绑定](../configs/libero-source-internal-spt-spi-execution-binding_v2.json)均在实际更新前冻结。

| 项目 | 冻结设置 |
|---|---|
| 探针0 / source更新任务 | 0 / 1,3,4,7 |
| 探针1 / source更新任务 | 1 / 0,3,4,7 |
| seed | 20261022、20261023、20261024 |
| A | 原始未训练生成器实例化；无source反馈更新 |
| B | source support一步SGD适应后，以query一阶梯度更新隔离候选生成器 |
| C | 相同上下文生成容量与source批次的普通未适应Policy pooled BC |
| source更新 | 每fold/seed/臂50次生成器Adam；每次四SPI；B每SPI一步SGD，lr0.001 |
| 探针适应 | 每模型500次Adam，lr0.0003、clip1、batch8、sequence10、stride10 |
| Adam细节 | betas0.9/0.999、eps1e-8、weight_decay0 |
| Policy / loss | 既有CNN-only双相机；均匀七维masked MSE；无jitter或局部loss |
| 数据 | source demos0..29 support、30..49 query；仅causal `obs[t-1] -> action[t]`、t>=1 |
| 评价与checkpoint | 0/50/150/300/500固定日志点；每fold全部9个support fit后才读取探针query；不选择checkpoint |

source更新任务的query是外循环**训练反馈**，C读取同批次，不能把它当独立评价。独立离线读出是该fold排除的探针query。两个fold参数独立，不跨fold调参。探针适应的500-step Adam与source的一步SGD分别记在SPI学习规格，不把500当作元训练总预算。

B/C各匹配source数据批次、数据梯度次数、生成器容量、VJP和生成器optimizer更新次数。B另有一次性内循环Policy更新，C没有；**不宣称所有参数写入次数或FLOP、耗时完全相同**。正式比较仍需独立预算协议。

## 500端点：query七维动作MSE

每格为三seed算术均值，越低越好。

| source内部探针 | A 原始SPT | B FOMAML | C 普通预训练 |
|---|---:|---:|---:|
| task0 | 0.085372 | 0.081612 | 0.079437 |
| task1 | 0.052681 | 0.052996 | 0.053334 |

| 探针 / 比较 | 三seed端点Delta：20261022 / 23 / 24 | 均值Delta | 改善seed |
|---|---|---:|---:|
| 0：B-A | -0.000970 / -0.007220 / -0.003091 | -0.003760 | 3/3 |
| 0：B-C | +0.005145 / -0.000342 / +0.001723 | +0.002175 | 1/3 |
| 1：B-A | -0.004203 / +0.012453 / -0.007304 | +0.000315 | 2/3 |
| 1：B-C | +0.000039 / +0.003400 / -0.004453 | -0.000338 | 1/3 |

Delta为左减右。task1的B-C均值略有利，但只在1/3 seed改善；不得以此选择seed或宣称稳定收益。task1的B-A也是两个seed改善、一个seed恶化后均值略差，全部保留。

## 完整学习曲线

query MSE三seed均值如下，全部固定点都保留。

| 探针 / 臂 | 0 | 50 | 150 | 300 | 500 |
|---|---:|---:|---:|---:|---:|
| 0 / A | 0.254973 | 0.215832 | 0.122928 | 0.101847 | 0.085372 |
| 0 / B | 0.208319 | 0.153536 | 0.112777 | 0.099927 | 0.081612 |
| 0 / C | 0.194463 | 0.147427 | 0.111640 | 0.101923 | 0.079437 |
| 1 / A | 0.235962 | 0.160443 | 0.084733 | 0.079943 | 0.052681 |
| 1 / B | 0.188036 | 0.111792 | 0.079560 | 0.069832 | 0.052996 |
| 1 / C | 0.172901 | 0.104730 | 0.080776 | 0.072421 | 0.053334 |

归一化曲线面积为固定0..500区间、上述五点梯形积分再除500，仅描述有限预算内动作误差，不是闭环成功效率：

| 探针 | A | B | C |
|---|---:|---:|---:|
| 0 | 0.128576 | 0.112937 | 0.111308 |
| 1 | 0.095564 | 0.081101 | 0.080563 |

B相对A两探针均为3/3 seed面积降低。C均值同样低于B；task0 B-C三个seed面积都较高，task1方向混合。因此支持本预算下source反馈改善早期离线误差的线索，**不支持将该收益特指为FOMAML优于普通预训练**。500端点与曲线面积分开报告，不按好看的指标切换主张。

support与query完整曲线图：`/tmp/libero-source-internal-spt-spi-curves-20261103-v2.png`、同名SVG。阴影为三seed最小—最大范围，不是置信区间。[绘图入口](../experiments/plot_libero_source_internal_spt_spi_diagnostic.py)只读已有结果，无新模型调用。

## 分量与夹爪读出

500端点query分量MSE三seed均值：

| 探针 / 臂 | 平移 | 旋转 | 夹爪 | 夹爪符号准确率 |
|---|---:|---:|---:|---:|
| 0 / A | 0.085020 | 0.003357 | 0.332475 | 0.883812 |
| 0 / B | 0.082554 | 0.003208 | 0.313996 | 0.889622 |
| 0 / C | 0.079628 | 0.003143 | 0.307745 | 0.894489 |
| 1 / A | 0.053620 | 0.002466 | 0.200507 | 0.926739 |
| 1 / B | 0.051565 | 0.002306 | 0.209360 | 0.925327 |
| 1 / C | 0.052056 | 0.002273 | 0.210350 | 0.923659 |

这些是合法专家输入上的目标误差，不是真实控制成功率。夹爪连续幅值与实际命令的区别已由[动作映射核对](libero-gripper-action-mapping-audit-2026-11-03.md)说明；本轮未据分量取舍另改loss。

## 实际预算、设备和成本

- 完成18个探针Policy、9000次适应Adam更新、90个固定曲线checkpoint；另有18份生成器权重，共108份新权重文件，全部在`/tmp`，未复制到仓库。
- source完成600次生成器Adam更新、1200次一次性Policy SGD更新、4800次数据梯度和2400次生成器VJP。训练Policy前向为source4800、探针9000；离线评价6885。对应有效预测行为371410、689751、525375；模型重复行不是独立环境样本。
- 完整v2训练/评价阶段153.683s，含前置核验的入口总耗时155.957s；source更新循环41.227s、探针适应及日志69.965s、离线评价37.205s、缓存4.674s分别记录，不把总耗时都称训练。
- Python3.10.22、PyTorch2.4.1+cu124、CUDA12.4、NumPy1.22.4、RTX4090。物理GPU1，经`CUDA_VISIBLE_DEVICES=1`映射为进程`cuda:0`；GPU0其他作业未动。强制确定性开启、TF32关闭；host tensor cache峰值3174367630 bytes，CUDA allocated峰值486916096 bytes。
- 新环境step/reset、官方evaluation初态（含qualification/reserved）、validation/held-out数据payload及private simulator state读取均0。source示教索引30..49属于允许的meta_query，与同数字的官方初态索引没有配对含义。没有活动SPT接受/切换、Knowledge更新或Module注册。

## 失败、资源门槛与复核

v1首次CUDA反向因AdaptiveAvgPool的强制确定性限制中止，全部optimizer更新0；额外一次已完成Policy前向和一次未完成backward单列，旧代码/绑定/partial保留。v2使用固定网格下数学等价的普通平均池化，所有臂重新初始化。启动时合成池化验证的CPU/CUDA输出与梯度差均0、两次CUDA读出位级一致，验证池化的前向/反向与研究Policy调用分开记账。

v2第一次启动被其他GPU作业占用的门槛拒绝，没有创建训练产物或更新。GPU释放后使用同一配置/绑定重试，未抢占。等待时另做9次合成CPU前向/反向速度测量、optimizer0、source读取0；**没有CPU训练运行或第三个科学协议版本**。失败、资源和CPU测量JSON均保留。

独立[审计入口](../experiments/audit_libero_source_internal_spt_spi_diagnostic.py)不导入runner或Policy，检查108个文件及tensor hash、180组指标的七维加权恒等式、9000条更新事件、source窗口角色和预算、探针query开启顺序、生成参数到各臂0-step Policy的一致性。初始化重构最大误差0（预设容差abs1e-6）；全部通过。审计新增Policy前向/环境/optimizer为0，仅复查授权source action长度元数据。

新增相关专项49 passed；CPU回归 **739 passed in 14.84s**。排除缺少可选Crafter依赖的`tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`、`tests/test_smoke.py`，不声称全量pytest。编译、隔离环境检查、来源pin与diff检查另有最终verification记录。

## 核心产物与SHA-256

| 产物 | SHA-256 |
|---|---|
| v2科学配置 | `9bbf6a3a39b7a8cb4772fcfcd17022a95f040f02f10941fc2754dbe7a11379df` |
| v2执行绑定 | `a94cb9e81160587a846d1fbe739f0ed1f297d5714a5ae3225a802d4104cccc86` |
| `/tmp/libero-source-internal-spt-spi-20261103-v2.json` | `66a4fb7fac6288a710072271e440f6cea42167f4fb782b9a774c7554af6d0169` |
| `/tmp/libero-source-internal-spt-spi-independent-audit-20261103-v2.json` | `c9b8b0d5174374cdd0f9c7493bc9211e0f30ee14a7f0be4c5224527a3f7ca68b` |
| `/tmp/libero-gripper-action-mapping-audit-20261103-v1.json` | `e08d181b9c28a5d2b0dcc2fa8ac1754940deb83f4d87d9ec6642e991dbbc66d9` |

108份权重及逐次事件位于`/tmp/libero-source-internal-spt-spi-artifacts-20261103-v2/`。首次失败`/tmp/libero-source-internal-spt-spi-20261103-v1-failed-partial.json`、补充`v1-failure-audit`、`resource-gate`和`cpu-throughput`结果均以同一`libero-source-internal-spt-spi`前缀另存。

## 结论与下一步

SPT/SPI的source内部更新与受控实例适应已经实际跑通，超出了普通单任务BC接口检查；当前证据仅支持上述离线范围。没有得到FOMAML相对普通预训练的稳定优势，也没有测试闭环迁移，不能把可行性跑通等同主假设成立。

下一步先为**同一批18个500-step端点checkpoint**冻结source candidate闭环对照：两探针、三seed、三臂、每模型20个已使用的candidate状态0..19，共360 episodes；5步稳定化、horizon300、环境seed20261022。所有臂保留，不根据本轮曲线选模型或增预算。该闭环尚未执行，本轮交互预算保持0；新协议须写清最多108000个Policy动作、1800个稳定化动作、同顺序reset和三臂公开reset对齐门槛后再运行。正式validation/held-out/qualification仍封存，主实验方法与统计冻结另行推进。

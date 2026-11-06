# LIBERO source稳定化缓存的RGB／本体响应分解（2026-11-02）

执行前冻结，承接[120对闭环轨迹只读分析](libero-task4-task7-stabilization-trace-alignment-diagnostic-2026-11-02.md)。检验[source稳定化0/5](libero-task4-task7-source-stabilization-control-diagnostic-2026-11-02.md)首观察引起的固定Policy动作差，分别来自RGB、本体和输入交互的哪些响应。本项没有训练候选，也不直接解释六组闭环成败。

## 配对、来源与预算

- task4/7，训练seed20261022/23/24，六个原source_baseline500、CNN-only。现行执行默认仍为稳定化5步和500 updates，不选择checkpoint。
- 仅四份已采集的公开NPZ：每任务source-support demos0..29，稳定化0/5各30行，共60对、120个观察。示教20..29属于meta_support，和封存official qualification初态20..39是不同命名空间。原采集含固定零动作和物理演化，本项不创建环境、不重新读取初态。
- 两RGB相机及joint7+gripper2沿用原接口，joint/gripper一起替换。mask位1=agentview、2=eye-in-hand、4=本体；0表示全零步观察，7表示全五步观察，八组合均执行。混合输入可能不在自然轨迹上，不当可执行控制方案。
- [配置](../configs/libero-task4-task7-source-stabilization-input-decomposition-diagnostic_v1.json)SHA-256 `efcdcb8a090923ce5b6726eb746bf5cfa7e1012dd0c16de16d6c68d9d8874613`；[入口](../experiments/analyze_libero_task4_task7_source_stabilization_input_decomposition.py)SHA-256 `be31dbd71adeb4bebb35952285a7fa92050fa0ef73840449bf71ca6de563f575`。固定父结果、审计、角色协议、Policy配置/代码及离线helper，四缓存、六checkpoint、六旧读出的路径/hash亦逐一冻结。
- 每模型自然mask0/7必须精确复现旧CPU读出，各重复一次且精确一致，再执行mask1..6。固定batch30、CPU单线程、分析seed20261022，隐藏CUDA，OMP/BLAS/MKL=1。权重/输入hash不变，grad全None；来源或复现门槛失败即停止，保留事件、部分预测和failed结果，不加预算重跑。

| 本项研究预算 | 固定数量 |
|---|---:|
| checkpoint加载 | 6 |
| 主前向批次／预测行 | 48／1440 |
| 重复前向批次／预测行 | 12／360 |
| 自然端点精确复现／重复门槛 | 12／12 |
| 新optimizer／backward／Policy控制动作 | 0／0／0 |
| 新环境reset或step／HDF5或初态读取 | 0／0 |
| 新candidate观察或特征／专家标签行 | 0／0 |

## 冻结的响应与交互定义

对两个参照a=0/7分别计算，符号差为`f(a XOR mask)-f(a)`。报告七维动作L2、平移3维L2、旋转3维L2和夹爪绝对差的30行均值/中位数/最小最大值，保留逐demo七维符号向量与符号变化数，不选择有利相机或seed。

- RGB替换mask3，本体替换mask4，完整替换mask7；相机分别mask1/2。
- 相机交互：`f(a XOR3)-f(a XOR1)-f(a XOR2)+f(a)`。
- RGB×本体交互：`f(a XOR7)-f(a XOR3)-f(a XOR4)+f(a)`。
- 预测float32先转float64再相减；完整差应等于RGB差+本体差+交互，RGB差应等于两相机差+相机交互。向量恒等式最大残差须<=1e-12。范数不能相加作贡献百分比。
- 主描述指标：RGB与本体七维L2均值之差；只有同一任务三个seed在**两个参照下均严格同向**才记一致响应大小关系。eye-in-hand与agentview为预设次级比较；同时报告每个seed/参照和逐demo胜出数，不做显著性检验。

稳定化后的source观察和混合观察没有新专家目标，不使用旧action[0]计算MSE。不访问source query、qualification/reserved/validation/held-out/final evaluation，不更新SPT/Knowledge、不注册Module。响应较大不表示错误、必要性、物理根因或闭环收益；本项不因描述指标触发训练或rollout。

前置专项及相关测试52项通过，包含两个参照、交互符号、事件计数、门槛失败后禁读混合输入、输入/权重不变、角色/坏hash拒收及无环境/HDF5依赖导入。结果与独立复核在执行后追加。

## 结果：RGB响应更大，眼在手相机响应最强

**已完成，独立复核通过。** 两个任务在三个seed、两个参照下均满足预设描述条件：RGB替换动作L2均值大于本体替换，eye-in-hand大于agentview。每个task×seed×参照的30/30个demo都满足这两个大小关系；同一demo在seed和参照之间重复使用，不解释为360个独立样本或显著性检验。

下表每格为**零步参照／五步参照**，是30个demo的动作七维L2差均值，没有专家动作目标。完整观察替换在两参照下仅符号相反，范数相同。

| task | seed | RGB替换 | 本体替换 | 完整观察替换 |
|---|---:|---:|---:|---:|
| 4 | 20261022 | 0.100005 / 0.101582 | 0.006517 / 0.007441 | 0.103948 |
| 4 | 20261023 | 0.120649 / 0.121825 | 0.005645 / 0.006414 | 0.125456 |
| 4 | 20261024 | 0.112133 / 0.113498 | 0.007511 / 0.008639 | 0.119030 |
| 7 | 20261022 | 0.325929 / 0.332650 | 0.006666 / 0.013626 | 0.338547 |
| 7 | 20261023 | 0.320107 / 0.328546 | 0.007791 / 0.015537 | 0.335135 |
| 7 | 20261024 | 0.546124 / 0.552909 | 0.009250 / 0.016757 | 0.561528 |

| task | seed | agentview替换 | eye-in-hand替换 |
|---|---:|---:|---:|
| 4 | 20261022 | 0.001053 / 0.001351 | 0.100089 / 0.101567 |
| 4 | 20261023 | 0.000972 / 0.001067 | 0.120558 / 0.121744 |
| 4 | 20261024 | 0.003450 / 0.003871 | 0.112588 / 0.113823 |
| 7 | 20261022 | 0.002483 / 0.004435 | 0.325647 / 0.333055 |
| 7 | 20261023 | 0.001488 / 0.004083 | 0.318630 / 0.328332 |
| 7 | 20261024 | 0.003523 / 0.006370 | 0.546701 / 0.553854 |

对于这批source首观察的0/5差异，固定输入替换下眼在手响应远大于外部相机，并与两相机一起替换的响应处于相近量级。这比此前“眼在手像素MAE较大”多了一层固定Policy输出证据，但不是对所有观察扰动归一化的普适敏感度，也不是贡献百分比或闭环根因。

RGB替换引起的平移L2均值范围为task4 `0.098912–0.120935`、task7 `0.260253–0.497390`；夹爪绝对差范围为task4 `0.001361–0.007443`、task7 `0.096395–0.228213`。**所有seed、参照和预定替换均未改变夹爪命令符号。** 因而当前响应主要表现为连续动作幅值变化，不能拿它代替闭环task7/seed23/state3首步close/open反转的证据。这是source示教初态，不是六组candidate结果相反的同一批场景。

## 交互、反证与未解决问题

| task | 相机交互动作L2均值范围 | RGB×本体交互动作L2均值范围 |
|---|---:|---:|
| 4 | 0.000152–0.000690 | 0.001395–0.001956 |
| 7 | 0.002848–0.004085 | 0.008403–0.010924 |

这些范围跨三个seed及两个参照，逐demo有符号向量、中位数及分量统计保存在JSON。主分析中相机、RGB×本体两条向量恒等式最大残差均为0。RGB×本体交互并非零，尤其task7的本体替换响应随RGB参照变化，因此不能说本体无作用，也不能将各范数相加或转为百分比。

本项支持“稳定化引起的这批source输入差异，在当前固定Policy中表现出较强眼在手响应”。不支持以下升级：

- 响应更大等于动作更错；本轮没有合法新目标，不计算新观察的action MSE。
- 眼在手是低成功率的唯一原因；已有fresh闭环0/5仍各7/120，没有总体收益。
- 压低敏感性即可改善控制。[既有4像素平移增强](libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic-2026-10-28.md)曾降低敏感性但拟合和candidate表现变差，原停止决定保持。它使用不同任务/seed/数据范围，只作反证，不与本轮合并指标。
- 这批source首帧代表整条轨迹、source query、candidate六组或其他任务；本轮均未测量这些推广。

## 实际运行、provenance与验证

北京时间 **13:54:11–13:54:15**，主程序elapsed **4.34546s**。这不是500-update重训耗时：仅做60个batch30的前向，没有反向、optimizer或仿真。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、CPU单线程/MKLDNN开启；CUDA明确隐藏。原六模型训练seed保持不变，分析seed20261022。

实际六个checkpoint加载、48主前向/1440行、12重复/360行，12个自然端点与12个重复均精确通过。四份公开缓存、六份原读出及权重来源全部匹配；权重hash、grad和输入不变，122条事件逐批记录。没有新增训练、环境、HDF5、初态、专家标签或candidate数值读取，没有更换默认或选checkpoint。旧读出NPZ只解码source_demo_indices及两个source prediction键，其他内容仅随文件hash核验。原source采集的pre-forward Path失败artifact仍保留，父结果与本项分开。

| 证据 | SHA-256 |
|---|---|
| `/tmp/libero-task4-task7-source-stabilization-input-decomposition-20261102-v1.json` | `2cc2ab1f14cf6bfdc2ede73d01a2bcd3df1f3d62ed72adc6c722f739df287ecc` |
| `/tmp/libero-source-stabilization-input-decomposition-independent-audit-20261102-v1.json` | `88688b21a7edd74ccf49634129e2addd9c59463eb9a32ece1489a810802d02c5` |
| `/tmp/audit_libero_source_stabilization_input_decomposition_20261102_v1.py` | `da04b7e066ada708cbeb4a465bd57ae2b41d9017665e67324e8e92c0746a910d` |
| `/tmp/libero-source-stabilization-input-decomposition-20261102-v1.log` | `17b886729e2e00ce32320ec69881f6b4281c3ac7276dd35b29d0dedf7792fa1d` |

六份新预测NPZ及事件日志在`/tmp/libero-task4-task7-source-stabilization-input-decomposition-artifacts-20261102-v1/`，路径/hash逐一写入结果。父结果仍为`a2d3df183cb2f830fdcfdd7a1621ac45792920b7ce18a56f8e1ed797b5229115`。独立审计不调用新入口/helper、不重新加载模型，用保存预测重算全部符号向量、范数、分层统计和三seed判断，核验自然/重复精确一致、原文件hash、122条事件及实际预算；耗时0.11158s，共23,458项递归核对，浮点最大绝对差2.22e-16，整数/角色/hash/复现仍精确比较。

新增18项测试、相关52项通过；新入口py_compile、审计内存compile、`check_environment.py`通过。CPU回归 **649 passed in 14.42s**，排除当前环境可选Crafter依赖的`tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`和`tests/test_smoke.py`，不声称全量pytest。前置测试发现的事件字段命名冲突在协议冻结/研究运行前修复，本项研究执行没有失败或追加前向。

最终来源pin、代码/配置/报告hash、文档链接、原进程退出0与验证命令记录于`/tmp/libero-source-stabilization-input-decomposition-final-verification-20261102-v1.json`。新源码/配置/报告无行尾空格，`git diff --check`通过。没有修改共享训练/环境实现、旧checkpoint或旧结果，没有暂存、提交或清理用户其他工作区内容。

## 下一步

保持baseline500、稳定化5步，不直接重启已停止的4像素增强或局部夹爪加权。下一项应先冻结**合法专家输入上的相机干预—目标动作误差对照**，检验较大的眼在手响应是否伴随目标误差及必要几何信息的损失；先核对已有专家状态敏感性和增强负结果，避免重复或搜索有利强度。它须使用原causal专家观察/目标及明确的观察干预语义，不能把本轮五步或混合观察与旧action[0]配成新标签。本轮未运行该后续项，也不由本轮输入响应触发新训练、闭环或qualification。

后续协议更新：[目标误差对照的研究设计](libero-task4-task7-expert-camera-target-error-protocol-2026-11-02.md)现已冻结并校验，尚未执行。新协议只取原source-support合法专家配对，采用八条件、共同像素MAE和有符号误差分解；空间probe有纹理/分布外限制，不宣称几何必要性。保留500 updates与闭环5步，下一步实现并绑定runner；本文的原实验结果和artifact不改写。

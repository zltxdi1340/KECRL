# 六组source内部配对轨迹只读诊断（2026-11-03）

**分析和独立复核已完成。训练设置保持不变。** 现有证据把行为差异定位到首步Policy映射和依赖seed的夹爪命令组织；不支持统一“更早close”“更多close”“更早open”或“压低joint位移”的训练修改。没有专家动作、RGB或目标几何，尚不能唯一归因到loss、视觉encoder、action head或数据覆盖。

本轮只读[已完成的360条闭环轨迹](libero-source-internal-spt-spi-closed-loop-diagnostic-2026-11-03.md)，task0/1×三seed全部六组，18个500-update端点、稳定化5步不变。分析执行新增Policy/环境/训练/HDF5/RGB/初态/checkpoint调用均0；validation、held-out、qualification继续封存，不接受活动SPT、不更新Knowledge、不注册Module。

## 定义与可追溯范围

先冻结[分析定义](libero-source-internal-six-group-paired-trace-protocol-2026-11-03.md)和[配置](../configs/libero-source-internal-six-group-paired-trace-readonly-analysis_v1.json)，再绑定[执行代码](../configs/libero-source-internal-six-group-paired-trace-analysis-execution-binding_v1.json)。[入口](../experiments/analyze_libero_source_internal_six_group_paired_traces.py)只使用NumPy/标准库及既有纯数值命令指标，不导入Torch、HDF5或仿真器。旧指标中的物理阶段heuristic字段没有输出到新结果。

- 360条唯一episode，99817行公开trace、99457个动作。
- 主比较240对：六组×B-A/B-C×20初态；全部四格（共同成功、仅左成功、仅右成功、共同失败）保留。
- 次级比较360对：每task、每臂的全部三个seed两两组合，共18个20对组。它们复用原episode，不是新增独立样本。
- action[t]使用obs[t-1]，trace row[t]为动作后状态；row0为五步稳定化后reset。所有配对必须有相同reset摘要和公开初态。
- close指正夹爪命令，open指负命令，release仅指close后的首个负命令。没有close与close后没见open分别计数；首段持续到episode结束记右删失，不能当实际完整持握时长。
- 同步点固定1/5/10/20/50/100/200/300；缺失均为null并保留有效n，不向后填充成功终点。固定20步主窗口与较短共同前缀分别命名。

## 六组B的命令模式

首步close、未close和close后未open均以每组20条计。首次close和首段长度的中位数仅在观测到close的n条上计算；长度可能右删失。每组A/B/C成功数仍是原始闭环结果，没有重算或筛选模型。

| task / seed末两位 | A/B/C成功数 | B首步close | B全程未close | B首次close中位数（有效n） | B首段close长度中位数 | B close后未open | B符号切换中位数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 / 22 | 0/11/4 | 0 | 9 | 45（11） | 43 | 8 | 1 |
| 0 / 23 | 6/0/5 | 18 | 0 | 1（20） | 11 | 1 | 1.5 |
| 0 / 24 | 2/1/3 | 9 | 0 | 3（20） | 2 | 0 | 4 |
| 1 / 22 | 0/0/0 | 0 | 14 | 123.5（6） | 103 | 4 | 0 |
| 1 / 23 | 0/0/1 | 0 | 2 | 59.5（18） | 241.5 | 18 | 1 |
| 1 / 24 | 0/11/0 | 2 | 4 | 67.5（16） | 53 | 12 | 1 |

seed完整值为20261022/23/24。B首段close右删失数依次为8、1、0、4、18、12。首段长度与success终止有关，不得由更短长度直接推出控制更好。

### task0 seed22：缺少或极短close与局部收益相伴

B的11条成功全部观测到close，其余9条失败全部未见close。B首次close中位45步、首段中位43步，初始命令全部为open。

- 对A的11个B-only成功：A有10条全程不close，另1条到249步才close且首段仅1步；B全部close，中位45步。
- 对C的7个B-only成功：C有5条不close，另外2条虽然比B更早close，但首段仅1步。B首段中位43步。
- 四个B/C共同成功案例，两臂均观测到close。另九个共同失败案例两臂均未close。

这支持该组存在命令进入与维持的行为差异，不能单独证明close是成功原因，也不能把B-C的收益全部写成“更早close”：其中两对B反而更晚开始。

### task0 seed23：B首步组织与A不同，但“首步close”本身不能解释失败

B有18/20首步close，A仅3/20；B首次close中位1步，而A中位41.5步。但C也有19/20首步close且成功5条。

在C-only成功的5对中，两臂首步夹爪符号全部相同，4对首次close时刻相同；B首段close中位9步、C为2步。B仍失败，说明不能从“首步close”直接判定动作错误，后续命令与运动差异仍待分离。B-A的6个反向结果中既有更早、相同也有更晚close，不是一条统一时序规则。

### task0 seed24：首段较短、切换较多，但同样存在反例

B首段close中位2步、符号切换中位4次，A分别为248步/1次，C为87步/3次；B只成功1条。该模式提示命令组织不稳定的候选现象，不能直接命名为物理接触抖动。

唯一B-only相对C成功对中，两臂都在第1步close、第3步open，首段均2步；三个C-only成功对的首次close先后又分别为B更晚、相同、更早各一对。因此修改单一close时机或首段长度不足以覆盖该组差异，n=1的B成功也不能推广为规则。

### task1 seed22：不close只是B的一类失败，不能解释整个组

B有14/20未close，其余6条close中位123.5步；但A有19条、C全部20条出现close，三臂仍全部失败。A/C是反证：仅恢复close不能由本组推出能解决控制问题。该组没有成功样本，成功分层保持空值，不从其他seed借样本补齐。

### task1 seed23：长close仍可能失败，未open也不等于未完成任务

B有18条出现close，全部没有后续负命令，首段中位241.5步且18段均被episode结束右截断，全部失败。C唯一成功案例也没有close后的负命令。

该C-only成功对中B/C分别在62/58步close，首段观测长度239/96步，后者在成功时终止。不能把长度差解释成更短持握导致成功，也不能从未见open直接推断物体未释放或LIBERO目标未完成；Module契约中的释放/资源约束尚未检查。

### task1 seed24：B收益不是统一“更早close”

B的11条成功全部从open开始，在68步附近close（中位数）；对应A首次close中位60步、C为第1步。**这11对中，B的close逐对都晚于A和C。** 相对C的首步符号在11对中全部不同；相对A则全部相同。

B成功中10/11未见close后负命令，且首段在成功结束时右截断。B的失败又包含4条不close和若干早close/早open，不能只依据成功段给所有状态施加延迟close或持续close规则。C全部首步close且0成功，与task0 seed23的C早close成功案例共同说明行为依赖任务和状态。

## 首步映射差与公开状态差

全部240个跨臂配对和360个跨seed配对，首动作七维向量均已不同，而其输入reset观测摘要完全相同。首步前公开joint/gripper距离为0；首动作差不能由此前Policy动作积累出的观察漂移解释。但没有专家目标，不能判定哪一个首动作正确，亦不能拆成视觉encoder或action head的唯一贡献。

下表为各20对的中位数。动作L2包含连续夹爪幅值；须同时看符号差，同号幅值差在相同runtime控制历史下不改变夹爪增量。

| task / seed末两位 | 首动作L2：B-A | 符号不同对数 | 第20步joint距离：B-A | 首动作L2：B-C | 符号不同对数 | 第20步joint距离：B-C |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 22 | 0.101967 | 0 | 0.143324 | 0.117916 | 0 | 0.147856 |
| 0 / 23 | 0.818837 | 15 | 0.333930 | 0.191789 | 1 | 0.069162 |
| 0 / 24 | 0.469738 | 7 | 0.234822 | 0.286387 | 5 | 0.122079 |
| 1 / 22 | 0.131929 | 0 | 0.156070 | 0.173602 | 0 | 0.104186 |
| 1 / 23 | 0.312141 | 0 | 0.071699 | 0.124186 | 0 | 0.121215 |
| 1 / 24 | 0.588324 | 2 | 0.110555 | 0.571321 | 18 | 0.660860 |

task0 B跨seed22/23的首动作L2中位1.244651、18/20对夹爪符号不同；task1 B跨seed23/24为0.575855，仅2/20对符号不同，仍有11对成败反转。seed差异包含连续机械臂映射及后续命令组织，不能统一压缩为首步夹爪符号。

公开状态距离不是专家误差。task1失败的B seed23和成功11条的B seed24，第20步相对reset的joint位移中位数分别0.483242和0.476765，大小接近；同初态跨seed joint位置距离中位0.149175，方向和路径仍不同。现有记录不支持简单以更小位移作为训练目标。

所有360条都至少72步，因此前20步、50步配对都完整。主配对在100步仅214/240可比，200步和300步仅177/240；300步全部是共同失败的177对。晚期差异带有终止选择，不能与成功结束后的“补值”混合，也不能用来证明失败组发生了更大的专家状态漂移。

## 训练修改决定与未解决问题

**保持500 updates和现有loss/观测设置，本轮不启动训练。** 具体限制如下：

- “没有close”与“close后仍失败”共存；不能据task0 seed22单组收益提高全局close偏好。
- B胜出时既有更早、也有更晚close；不能统一提前或延迟夹爪命令。
- 多条成功轨迹结束前未见后续负命令；不能把未open当统一失败原因或直接提高release loss。
- 同close/release时刻仍有成败反转；需要区分机械臂动作与命令持续/切换，不能仅改夹爪权重。
- 早期公开位移不是目标误差；不因某些失败移动较多就增加状态抑制或jitter。

已解释的是行为差异的发生时间与分组模式；尚未确定差异是在source生成器初始化、500步适应还是自由运行输入分布中产生，也没有目标几何/接触证据来唯一定位失败阶段。

下一项建议先冻结**固定合法source-support专家观察上的0点/500点动作目标误差对照**，覆盖全部task/seed/臂，分离生成器初始化与适应后的变化。0点只作诊断，不选择执行checkpoint，执行端点继续500；不读qualification/validation/held-out。该新前向诊断需另行明确预算与hash，**本轮尚未执行**，不直接恢复已停止的局部夹爪loss或增强分支。

## provenance、验证与产物

科学配置SHA-256 `5f4d08db8535612c3ff2b1c31bc3865be6725c2300eda1bb6bb68c6b505b393d`；执行绑定 `9c6a8d4d222821c90c871f5e81eed9c4c5b4f2fd6e3e7ce10c31c35e39014753`。

分析结果：`/tmp/libero-source-internal-six-group-paired-trace-analysis-20261103-v1.json`，14805740 bytes，SHA-256 `8684fbf5e0f3dde28ddad0c0c41aa7afd4f4d1c5b18aefb5a39e83cbc57ff439`。含360条episode特征、240条主配对、360条次级配对和全部分层的均值/中位数/有效n/缺失n。运行elapsed2.38479s，CPU、Python3.10.22、NumPy1.22.4，不进行新模型前向或环境操作。

独立审计：`/tmp/libero-source-internal-six-group-paired-trace-independent-audit-20261103-v1.json`，SHA-256 `80c8598bc3807393f6b932842a29a3f1bb576beab4f118f395074acc0ac31c79`。独立脚本`/tmp/audit_libero_source_internal_six_group_traces_20261103_v1.py`不导入分析入口，从原始public trace核对360条事件/数值特征、600对首步与同步状态距离、60个主比较分层和54个episode分层；133166项递归数值检查通过，预设abs/rel1e-12，最大绝对差8.8818e-16。审计新增Policy/环境/HDF5/RGB/初态/checkpoint调用0。

新增21项分析测试，与原审计相关32项通过；CPU回归 **813 passed in16.52s**。排除可选Crafter依赖的`tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`、`tests/test_smoke.py`，不声称全量pytest。空组、事件删失、共同前缀、状态距离/位移区分、全部seed组合、禁止导入训练/仿真包与旧产物保留均有测试。唯一首次单测失败是合成序列符号切换期望误写5（实际四次），在生成分析结果前更正测试预期，分析实现和研究定义未变。

源闭环结果、原配置与绑定、checkpoint、用户指导和无关工作区修改均保持。最终hash、链接和工作区验证见`/tmp/libero-source-internal-six-group-trace-final-verification-20261103-v1.json`。所有结论为source内部描述性机制诊断，不是独立验证、因果根因、SPT接受、Knowledge更新或Module qualification。

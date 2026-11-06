# LIBERO task0/1/4/7现有结果横向整理（2026-11-05）

**四个task均已有500端点的闭环诊断，但分属两套实验：task0/1为source内部A/B/C初始化对照，task4/7为单任务BC及jitter诊断。** 可以并列查看现状，尚不能据此给任务排固有难度，或将两套方法的成功率差当成算法效应。本项只读整理旧产物，没有训练、模型前向或新rollout。

## 任务与闭环总表

共同目标为把黑碗放到盘子上；task4的上层抽屉初始已经打开，不包含学习开抽屉。

下表每个已测格均为3个训练seed×20个已用candidate初态，60条episode。横线表示本表没有同协议结果，不代表成功数0。

| task | 黑碗起始位置 | A原始初始化 | B上下文FOMAML | C普通预训练 | 单任务Full Policy BC | BC+public-state jitter |
|---|---|---:|---:|---:|---:|---:|
| 0 | 盘子与小烤碗之间的桌面 | 8/60（13.33%） | 12/60（20.00%） | 12/60（20.00%） | — | — |
| 1 | 小烤碗旁边的桌面 | 0/60（0%） | 11/60（18.33%） | 1/60（1.67%） | — | — |
| 4 | 已打开的木柜上层抽屉内 | — | — | — | 3/60（5.00%） | 3/60（5.00%） |
| 7 | 炉灶上 | — | — | — | 4/60（6.67%） | 3/60（5.00%） |

A/B/C来自[2026-11-03 source内部闭环](libero-source-internal-spt-spi-closed-loop-diagnostic-2026-11-03.md)。BC baseline来自[2026-10-30 Full Policy闭环](libero-task4-task7-full-policy-closed-loop-diagnostic-2026-10-30.md)，jitter来自[2026-10-31固定500闭环](libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic-2026-10-31.md)。jitter比较复用历史baseline，并非两臂同时fresh重跑；不选其他budget或较高单seed替代这些结果。

## 逐seed结果与各自能够支持的结论

三列seed按20261022/23/24排列，每格分母20，不删除失败seed。

| task | 方法 | seed22 | seed23 | seed24 | 总成功/60 |
|---|---|---:|---:|---:|---:|
| 0 | A | 0 | 6 | 2 | 8 |
| 0 | B | 11 | 0 | 1 | 12 |
| 0 | C | 4 | 5 | 3 | 12 |
| 1 | A | 0 | 0 | 0 | 0 |
| 1 | B | 0 | 0 | 11 | 11 |
| 1 | C | 0 | 1 | 0 | 1 |
| 4 | BC | 0 | 2 | 1 | 3 |
| 4 | BC+jitter | 0 | 3 | 0 | 3 |
| 7 | BC | 0 | 1 | 3 | 4 |
| 7 | BC+jitter | 0 | 0 | 3 | 3 |

- task0：B相对A合计多4次成功，但相对C持平；B-C逐seed为+7/−5/−2，未表现为一致元学习优势。
- task1：B合计高于A/C，但11次成功全部来自seed24；B-C为0/−1/+11，不能把单seed55%当稳定迁移效果。
- task4：jitter总体持平，seed23多1次、seed24少1次。配对四格为共同成功2、仅baseline1、仅jitter1、共同失败56，持平不等于每个场景结果相同。
- task7：jitter少1次成功；共同成功1、仅baseline3、仅jitter2、共同失败54。没有三seed一致改善。

四任务各自的控制可靠性仍有限；当前没有稳定跨seed的FOMAML优势，也没有支持把jitter作为task4/7统一修复。此结论不认定唯一失败机制。

## 共同设置与不可混同的设置

| 项目 | task0/1 A/B/C | task4/7 BC/jitter |
|---|---|---|
| 任务角色 | 已暴露source内探针，非正式held-out | 已暴露source单任务诊断 |
| 500的含义 | 每个探针Policy适应500 updates | 每个单任务Policy训练500 updates |
| 额外来源经验 | B/C有来源更新；探针0用1/3/4/7，探针1用0/3/4/7 | 单任务support BC；jitter改变训练输入 |
| source预算 | 每fold/seed/臂50次生成器Adam；B另有一步内循环适应 | 不存在对应FOMAML来源更新流程 |
| Policy/loss | CNN-only双相机，均匀七维MSE，无jitter | CNN-only双相机，均匀七维MSE；候选为public-state jitter |
| 闭环共同条件 | 环境seed20261022、5步稳定化、最多300动作、candidate0..19 | 同左 |
| 实际执行 | 18模型/360条fresh；物理GPU1确定性推断/渲染 | baseline与jitter跨次运行/设备；reset配对核对，baseline是历史reference |
| 隔离评价 | qualification/validation/held-out/final evaluation未访问 | 同左 |

“都为500”只描述端点预算，不能当成整个学习过程等数据、等更新、等算力。B/C的来源更新含额外数据梯度；B还含一次性Policy内循环写入，不声称全部参数写入/FLOP相等。A生成初始化与普通BC初始化也不是同一控制变量。

跨task的初态bank不同，同一个state编号不构成跨任务场景配对。这些0..19状态已经多轮使用，不称未见测试。当前只能比较每套协议内部的A/B/C或BC/jitter；不能把task0的20%与task4的5%之差归于算法，或据此证明task4更难。

## 离线query误差作为背景

以下均为对应协议500端点的三seed均值，越低越接近其专家动作；只在各协议内比较方法，不能换算为闭环成功率。

| task | A | B | C | 单任务Full Policy BC |
|---|---:|---:|---:|---:|
| 0 | 0.085372 | 0.081612 | 0.079437 | — |
| 1 | 0.052681 | 0.052996 | 0.053334 | — |
| 4 | — | — | — | 0.038517 |
| 7 | — | — | — | 0.042630 |

task0最低MSE是C，闭环B/C合计持平；task1的A/B/C query MSE接近，但闭环分别0/11/1次成功。task4/7有较低的单任务动作误差，闭环仍只有3/60、4/60。整体上，离线拟合改善没有保证可靠自由运行。

数据来源：[A/B/C离线报告](libero-source-internal-spt-spi-diagnostic-2026-11-03.md)与[task4/7示教泛化报告](libero-source-meta-query-generalization-diagnostic-2026-10-30.md)。前者明确t>=1、探针query在该fold来源更新中排除；后者为同任务不同示教、causal pre-action含初始RGB重建。帧覆盖、初始化和学习过程不同，这张MSE表也不能用来做任务固有难度排序。

task1后来的固定编码器/小读出头替换另属机制子实验：[同候选权重真实历史0/60、重复当前4/60](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)。控制器已改变，不能替代本表完整B Policy的11/60。其后专家目标误差、表示探针同样单列，不混入四任务A/B/C成绩。

## task4/7稳定化复核单列

后续[同栈fresh 0/5稳定化对照](libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic-2026-11-02.md)不重训，另行执行两臂240条；不是jitter实验，不追加到上表分母。

| task | fresh稳定化5步 | fresh稳定化0步 |
|---|---:|---:|
| 4 | 3/60 | 2/60 |
| 7 | 4/60 | 5/60 |

五步计数及逐seed成功数与历史baseline相同，但完整轨迹只117/120与历史精确一致；旧完整复现失败保持failed，不能因计数相同改判。零步没有任务内三seed一致收益，继续保留5步，不为横向表挑task7较高的零步结果。

## 整理验证、证据和下一步

本轮核对6个原始结果JSON的SHA-256，从episode重算10个方法×任务单元、30条seed记录及jitter四格；另提取240条fresh稳定化复核和8个离线均值。未新建Policy或环境，不读取HDF5、RGB、checkpoint参数或初态文件，无新训练、rollout、SPT/Knowledge更新或Module注册。不运行pytest：本轮仅报告/数据整理，以来源hash、逐episode断言、聚合一致性和文档检查验证。

可复现整理脚本：`/tmp/summarize-libero-task0-task1-task4-task7-20261105-v1.py`。输出包括：

- `/tmp/libero-task0-task1-task4-task7-comparison-20261105-v1.json`，SHA-256 `97bea7a875431771e420cb7089f2017c2fca1e6f5291f07885e06203fc2dbfa2`；包含六个来源路径/hash、设置与比较边界，不计算四任务混合总成功率。
- 同前缀`-closed-loop.csv`：10个方法×任务单元。
- 同前缀`-seed-results.csv`：30条逐seed记录。
- 同前缀`-overview.csv`：中文四任务横向表，可直接用Excel打开；缺项明确标为未按同协议测量。

四任务都属于diagnostic，不构成新任务上的正式SPT接受、Knowledge机制或Module qualification。validation、held-out、qualification、reserved和final evaluation继续封存。

若后续要比较四task上的同一方法效应，需要另冻补齐task4/7的A/B/C协议，明确每fold来源任务、几何覆盖、来源更新/适应预算、端点与闭环配对。尤其task4是唯一开放容器几何，留一设计需要单独交代来源几何缺失，不能直接照搬0/1的解释。本轮只整理，不自动追加训练或改现有500/5步/loss；原观测层机制诊断建议保持待冻结状态。

给老师可概括为：**“四个来源任务都有闭环数据，但目前是两条实验线。task0/1的元学习候选有局部收益且依赖seed；task4/7的单任务BC可靠性低，jitter未形成统一改善。我们已经整理逐seed和协议差异，下一步需要同协议补齐后才能回答四任务上的稳定方法优势。”**

最终文件保全与聚合核验记录：`/tmp/libero-task0-task1-task4-task7-comparison-final-verification-20261105-v1.json`。既有1224份受保护文件与HEAD保持，用户AGENTS.md/.vscode及无关改动不动。

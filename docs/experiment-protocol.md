# KECRL 最小实验协议

> 状态：阶段性实验方案已冻结（2026-10-03）；小规模预算与阈值为待验证候选，不代表最终正式实验设置。CPU/GPU smoke 只验证代码和数据流，不是正式实验结果。

## 研究问题

1. Knowledge Evolution 是否能按 Beta-Binomial 规则更新、确认和拒绝原子命题？
2. 合格 Module 是否能在契约、范围和资源兼容时复用，并在不兼容时返回 `unavailable`？
3. 上下文条件化 FOMAML 是否改善后续 SPI 的学习效率？
4. Knowledge/Skill 双反馈分流和持续任务闭环是否可追溯运行？

## 环境与任务

- 第一阶段正式验证环境：受控离散资源环境，提供可检测的结构化状态谓词、可控机制干预和确定的资源转移；实现时冻结环境规则、观测接口与版本。
- Crafter 作为后续外部环境验证，不属于本阶段冻结范围。已有 Crafter 木镐案例仅为理论流程示例。
- 任务族包含机制识别和技能元学习两部分。机制识别任务检验支持、反证、UNKNOWN、去重、预算耗尽及机制状态变化；技能任务由共享 Skill Family 下的多个 SPI 构成，包含不同资源/对象绑定及独立 query 实例。
- 训练、support、query、qualification 和 SPT candidate validation 使用互斥 episode/seed 列表；不得跨集合复用样本。每个任务固定起始 Knowledge/SPT 版本视图。
- 当前原型位于 `src/environments/discrete_resources.py`，包含 gather/craft/use 资源任务、成功与缺少前置资源路径、支持/反证/UNKNOWN 机制样例，以及按 seed 生成互斥 train/support/query/qualification/SPT-validation episode ID 的接口。
- 该原型提供确定性转移和机制真值，尚无策略训练、随机地图、完整数据集或防泄漏统计审计；具体 SPI 清单、episode 生成规则与数据集版本仍需在正式训练前冻结。
- 当前阶段 manifest 由 `python -m experiments.prepare_dataset --seeds 0 1 2 3 4` 生成到 `datasets/discrete_resource_manifest.json`；它记录任务、seed、角色划分、初始资源和目标规格，`formal_result=false`，不包含训练样本或实验结果。当前默认每个角色每个 seed 只有 2 个 episode，是可执行性验证规模，不是协议中的正式预算。

## 基线与消融

- 主方法：启用 Knowledge Evolution、context-conditioned FOMAML Skill Evolution 和兼容 Module 复用。
- Baseline：保留兼容 Module 复用，关闭 Knowledge Evolution 和 Skill Evolution；其余规划、环境交互、Policy 训练预算和评估数据保持一致。
- Ablation K：关闭 Knowledge Evolution，保留 Skill Evolution 和 Module 复用。
- Ablation S：关闭 Skill Evolution，保留 Knowledge Evolution 和 Module 复用。
- 每种设置使用 seeds `0,1,2,3,4`。消融只改变对应演化组件，不改变任务划分、数据预算或资格/评价规则。

## 指标

- 主指标：独立 query 学习效率，定义为达到 held-out query 成功率门槛所需的 support 环境交互步数；预算内未达到门槛按右删失记录，不得记为零步或丢弃。每个 SPI 单独报告并在任务族内宏平均。
- 次指标：固定 support 预算下的 query 成功率/损失曲线；Module qualification 通过率和复用率；任务 completed 比例及 unavailable/unknown 比例；Knowledge 原子命题与机制识别 precision/recall、有效证据消耗；运行时间和峰值显存。
- Knowledge 指标只使用独立标注的机制真值，不用 Module 成功率替代结构证据。
- 报告每个 seed 的结果、跨 seed 均值与标准差；置信区间和显著性检验方法待小规模数据形态确认后冻结。不得仅报告汇总均值。

阶段统计分析方案已在
[`configs/statistical_analysis_v1.yaml`](../configs/statistical_analysis_v1.yaml)
中冻结为下一轮受控比较的分析规则：按 seed 配对 baseline，报告 support
budget query 曲线、每个 seed 差值、均值和样本标准差；bootstrap 仅作当前五个
seed 的诊断区间。正式论文推断、最终 seed 预算和多重比较处理仍待最终评审。

## 小规模预算与候选阈值

以下为第一阶段候选上限，作用是检查流程与估计成本；完成实现和先导检查后需复核，不能据此声称充分统计功效：

| 项目 | 候选值 | 状态/边界 |
|---|---:|---|
| 随机种子 | `0,1,2,3,4` | 已冻结 |
| 每个 seed 的任务 episode | 200 | 候选；训练/适应总预算，按任务族分层 |
| 每个 SPI support 交互步上限 | 500 | 候选；达到上限仍未成功则右删失 |
| 每個 SPI query episodes | 20 | 候选；独立于 support |
| 每個 Module qualification episodes | 20 | 候选；独立 held-out 集 |
| 每個 SPT 比较 SPI 数 | 10 | 候选；候选与 active 使用完全相同 SPI 列表及预算 |
| Knowledge 验证预算 | 每 seed 200 个验证单元 | 候选；每项任务成本按公开计数规则记录 |
| Qualification 成功率门槛 | `>=0.8` | 候选；20 个有效 episode 至少 16 次成功，未知/无效单独报告并补测但不得超过预算 |
| Contract/safety 门槛 | 所有有效 episode 零硬违规 | 候选硬门槛；任何硬违规阻止形成 Module |
| Knowledge `n_min` | 10 个有效、同范围原子证据 | 候选；UNKNOWN 不计数 |
| Knowledge posterior 阈值 | `tau_confirm=0.8`, `tau_reject=0.2`, 置信水平 `0.95` | 候选；后验区间算法须在实现前明确并测试 |

### 阈值预算诊断

在当前 Beta-Binomial 实现和上述候选阈值下，受控扫描显示：纯支持证据首次达到
`confirmed`、纯反证首次达到 `rejected` 均需要 13 个有效同类证据；UNKNOWN 证据不增加
有效样本，100 次尝试后仍保持 `candidate`。该扫描保存在
`results/knowledge_budget_scan.json`，并标记 `formal_result=false`。13 个证据是当前实现的
局部诊断结果，不是正式统计功效承诺；正式实验仍需在冻结任务、范围和验证预算后复核。
| FOMAML 内循环 | 每 SPI 最多 20 次更新 | 候选；query 不参与内循环 |
| FOMAML 外循环 | 每 seed 200 次更新 | 候选；批大小与更新数据索引另存配置 |
| SPT 最小 query 改善 | 相对 query 学习步数至少改善 `10%` | 候选；active/candidate 配对比较，需达到预先选定的不确定性判据 |
| 既有 SPI 退化界 | query 成功率下降不超过 `0.05` | 候选非劣界；重要 SPI 列表及独立评估集预先固定 |
| 退化检查窗口 | 连续 2 个独立 validation batch | 候选；硬契约/安全违规不等待窗口 |

SPT candidate 只有在独立配对 validation 上达到最小改善、既有重要 SPI 均满足非劣界且没有硬契约/安全违规时接受；改善不足、退化超界或硬违规时拒绝；有效样本不足或结果不确定时标记 inconclusive 并保留 active SPT。candidate、active、previous stable 分别版本化，candidate 评估不能污染训练、support、qualification 或其他 query 集。

每次运行保存完整配置、随机种子、Git commit、Python/框架/CUDA/GPU 信息、数据划分索引、结构化日志、JSON/CSV 结果、checkpoint 和失败日志。训练步数、Knowledge/FOMAML 预算、worker 数和混合精度均从独立配置读取。单卡起步；双卡是否有必要待吞吐测量后决定。

## 未决项目与进入训练门槛

阶段性已选环境、任务族、baseline、ablation、主指标和 seed 如上。候选数值仍须经任务实现和先导运行检查后确认；FOMAML 学习率、query 成功判定、后验区间算法、SPI 具体清单、分层数据切分、置信区间/显著性方法及每 seed 总 wall-clock 上限仍待定。进入正式对比训练前必须冻结这些项目、环境代码版本、依赖锁文件及完整配置，并确认自动化测试通过。GPU smoke 不满足该准入条件，也不作为正式结果。

# Crafter 可行性诊断状态（2026-10-09）

当前仍处于正式实验前的可行性诊断阶段，尚未注册 Module，也没有更新 Knowledge 或 SPT。

本版归档诊断源码、配置、测试、中文记录，以及最新 actor head 对照的 JSON 汇总与 PNG 图表。原始 checkpoint、监督 head 权重、特征、RGB 数据和逐步轨迹保留在实验机器上，未上传；仓库包含协议与运行入口，但从 GitHub 新检出后不能直接恢复本轮的全部本机原始产物。个人编辑器配置 .vscode 不纳入提交。

## 已确认的结果

- 在稳定环境 wrapper 和确定性 CUDA 设置下，100k 交互步的 CNN-only / CNN+GRU wood>=3 资格诊断均值分别为 0.533 / 0.500，均为0/3达到候选0.8；GRU没有显示稳定收益。固定死亡惩罚 -1 的资格均值为0.317，do_wood_gain辅助损失系数0.1为0.333，均不采用。这些是此前自然任务诊断，与下面的受控局部采集指标分开报告。
- 三个已有 100k CNN-only checkpoint 的 encoder、critic、PPO optimizer 和环境接口均被冻结；原始 Policy 没有被监督副本更新替换。
- 在冻结 CNN 特征上比较了原始线性 actor、原始特征线性拟合、训练集标准化后折叠回同一 Linear(128,17) 的线性拟合，以及原始特征 MLP64。
- 在与此前背景不重复的16个新背景上，八步 greedy 指定树采集率均值为：原策略 8.0%、原始特征线性头 9.7%、标准化线性头 76.9%、原始特征 MLP64 28.6%。
- 标准化线性头三个 source seed 的采集率分别为 75.5% / 73.3% / 81.8%，说明改善没有只来自单个 seed。
- 标准化线性 head 的 533 次失败中，119 次首步未执行正确的 do，174 次首步转向错误，240 次首步转向正确但随后没有执行 do。因此方向判断改善后，动作切换和执行链仍是主要问题。
- 该结果支持 actor 监督拟合存在优化条件瓶颈的解释，但不能证明原始线性 head 结构必然不足，也不能直接推出 teacher-free PPO 加标准化一定有效。

## 验证状态

- 全量测试：224 passed。
- paired 拟合和 fresh 评估均完成独立进程重复；轨迹、数据 digest、拟合选择和保存 head 状态一致。
- 保存的9个 head 重载审计通过147项检查；分析与轨迹边界检查通过378项检查。
- 原始 checkpoint、encoder、critic、PPO optimizer、Crafter 安装包和正式训练边界保持不变。
- 本轮没有启动 PPO 正式训练，没有注册 Module，没有更新 Knowledge/SPT；GPU1 当前空闲。

## 仍不能宣称的结论

- 受控相邻树的八步采集率不是 wood >= 3 Module 资格结果。
- 本轮没有重新验证自然 reset 下的 wood、stone、coal、pickaxe、长期生存或持续任务链。
- fresh 背景是在 paired 结果之后按固定协议开展的确认性评估，不是预注册的独立正式测试。
- MLP 只使用一个初始化，尚没有标准化 × MLP的完整因子对照。

## 下一步诊断顺序

1. 冻结原策略和标准化线性副本，做背景替换和自然 reset 迁移测试，记录“正确转向后不 do”、无相邻树时的探索停滞、重复采集和生存退化。
2. 检查构造数据中的 noop 标签是否让策略在自然任务中缺少探索行为；不把 oracle 或规则动作作为 Policy 输入。
3. 只有当局部执行和自然状态迁移边界更稳定后，才开展固定环境 wrapper、奖励、动作集和交互预算的 PPO 优化条件对照。
4. 通过独立的完整资格检查后，才考虑 Module 注册、Knowledge Evolution、SPT 更新和正式持续任务链实验。

详细协议、数据边界、逐背景结果和复现命令见[actor 优化条件诊断记录](crafter-wood3-actor-head-conditioning-diagnostic-2026-10-09.md)。

本轮归档的主要指标见[聚合结果](../results/crafter_wood3_actor_head_ablation_cuda_20261009_v1/analysis/aggregate.json)，剩余失败计数见[失败归因](../results/crafter_wood3_actor_head_ablation_cuda_20261009_v1/analysis/fresh_failure_attribution.json)。历史诊断文档中的未提交状态和计算计数描述的是各轮运行完成时的状态；本状态说明用于本次 GitHub 版本的阅读入口。

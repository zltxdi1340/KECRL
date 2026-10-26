# KECRL 研究路线冻结（2026-10-20）

本路线将 SPT/SPI、持续任务流、Knowledge Evolution 和最终联合验证拆成逐级实验目标。各环境只承担它能够支持的主张，不把不同环境的成功率拼接为一个总分。

## 当前顺序

| 阶段 | 环境 | 主要问题 | 当前证据边界 |
|---|---|---|---|
| 1 | LIBERO | 同一 Skill Family 的 SPI 迁移、support/query 适应和 SPT 更新 | 主要是操作技能和知识迁移；标准设置偏 lifelong imitation learning |
| 2 | Continual World | 任务按顺序到达后的 forward transfer、forgetting 和历史 Module 保持 | 主要是持续强化学习；CW10/CW20 不等于同族元学习 |
| 3 | Crafter | 长期记忆、探索、Capability 关系和因果 Knowledge Evolution | 不把当前 wood gather 结果升级为自然 Module 或完整 KECRL |
| 4 | RoboCasa365 | 原子 Module、复合任务、任务流和 Knowledge–Skill 联合收益 | 先做冻结子集，再扩展到更大任务规模；不直接把 benchmark 分数称为 Knowledge Evolution |

## 阶段 1：LIBERO

先选择一个共享动作结构的任务族，例如 `Pick`、`Place` 或 `Open`，而不是把整个 130 任务集合直接作为一个 Skill Family。

每个实验应包含：

1. 训练 SPI 和独立 held-out SPI；
2. A：原始 SPT；
3. B：来源 SPI 经验更新后的 SPT；
4. C：相同来源数据和预算的普通策略预训练；
5. 相同观测、动作接口、适应预算、评价初始状态和随机种子；
6. 独立 query、Module qualification 和迁移效率评价。

B 优于 A 才支持来源经验对共享 SPT 有帮助；B 优于 C 才支持收益来自 SPT 更新，而不是普通预训练。LIBERO 原生实验偏 imitation learning，因此若使用 RL 或 FOMAML，必须单独冻结算法和数据协议。

## 阶段 2：Continual World

Continual World 用来补充阶段式任务流证据。使用 CW10 做快速校准，CW20 做正式任务流候选；每个阶段只训练当前任务，并在阶段边界重测全部历史任务。

至少报告：

- forward transfer；
- backward transfer；
- forgetting；
- 历史 Module 保持率；
- 新任务达到资格门槛所需的交互步数；
- 任务顺序改变后的性能变化。

Continual World 的默认任务并不都属于同一个 Skill Family。因此它用于持续性、保持和遗忘，不作为 LIBERO 同族元学习结果的替代品。

## 阶段 3：Crafter

Crafter 继续承担 Knowledge Evolution 侧实验。当前 wood gather 诊断和历史 RGB/PPO 结果已封存，后续重点应转向：

- 可检测 Capability 的定义和验证；
- 资源、工具和任务前置条件的机制干预；
- 机制关系的证据累积、修正和置信度变化；
- Knowledge 对后续任务规划和任务可达性的影响。

Policy 成功率、Module 性能和梯度信息不直接写入 Knowledge Bank。Crafter 结果只支持 Knowledge Evolution 范围内的主张，除非另有独立的 SPI 和 Module qualification 协议。

## 阶段 4：RoboCasa365

RoboCasa365 是联合验证环境，不是早期调参环境。先冻结一个小任务子集：

- 2–3 个原子 Skill Family；
- 每个族包含训练实例和 held-out 实例；
- 2–3 阶段复合任务；
- 少量 4–5 阶段复合任务；
- 固定机器人 embodiment、观测、动作和数据来源。

在这个子集上比较四个臂：

1. 无 Knowledge Evolution、无 Skill Evolution；
2. 仅 Knowledge Evolution；
3. 仅 Skill Evolution；
4. Knowledge–Skill 双演化。

评价同时包含 Knowledge 结构正确性、SPI 适应效率、Module qualification、复合任务成功率、任务成本和遗忘。只有子集协议稳定后，才扩展到更多厨房环境、移动操作和更长组合任务。

## 统一边界

- 不把 LIBERO、Continual World、Crafter 和 RoboCasa365 的成功率直接合并。
- 不使用最终评价集选择 checkpoint、调参或修改任务顺序。
- 不把标准 benchmark 的多任务分数直接称为 SPT 更新或 Knowledge Evolution。
- 不要求同一个 Policy 后端跨 Crafter 和机器人环境复用；跨环境共享的是理论接口和评价边界。
- 每个环境都记录配置、seed、设备、预算、实际交互步数和数据来源。

## 当前状态

截至本路线冻结日：

- Crafter 仍是诊断/Knowledge Evolution 候选环境；没有自然 wood Module 通过完整资格检查。
- LIBERO 已完成 Python 3.10 联合环境、完整 `libero_spatial` 数据审计、10 个任务的 KECRL adapter smoke、候选 Skill Family 和数据隔离审计、CUDA 训练路径、候选 BC Policy/SPT/SPI pipeline、单 source SPI 固定预算学习与闭环归因。HDF5 时序审计确认 `obs[t]` 在 `action[t]` 后记录。随后完成 same-index/causal pre-action 对照、hidden-state follow-up、逐步动作/状态偏差、单步反事实、受控状态分布偏移、RGB/本体感知和 agentview/eye-in-hand 相机分解；三 seed 相机分解均确认 eye-in-hand 局部敏感性大于 agentview。单变量 eye-in-hand 平移增强 pilot 将 support/query 闭环从 baseline `0/9, 0/9` 变为增强臂 `3/9, 3/9`，但 query MSE 均值略升，新增成功集中在一个 seed。固定三 seed 的 499 个 expert-state 合成平移敏感性诊断显示增强臂在三个半径的四方向聚合中均降低 action、translation、rotation 和 gripper 敏感性。最新动作误差—状态偏移对齐显示：增强臂前 20 步 expert-state action MAE 更低，但两臂 joint-state 首次越界都在第 2–3 步，后段 drift 和 gripper-state 越界继续放大；因此相机敏感性不是唯一瓶颈，动作 head/数据覆盖与闭环反馈需要分开诊断。随后完成 source-SPI 条件覆盖、relation×geometry×phase 分层误差、专家重放一致性、HDF5—当前重放状态对齐、回放语义冻结、当前 runtime 轨迹和输入对齐审计；四条示教都在首个动作后偏离 HDF5 状态，包括成功控制示教，而检查的 controller 配置和动作边界没有差异，当前 runtime 重复轨迹确定，两种 causal 输入视图从第一个动作后开始分叉；随后固定重建 task 3/7 Policy 做输入替换敏感性，四条示教共 456 步的 shared-hidden action MAE 为 0.02185，独立 recurrent 分支为 0.03544，且差异均从 step 1 开始。输入语义分叉会改变动作输出，但不能解释全部失败；随后对 5 个 source-SPI 任务的 150 条 meta-support 示教做了完整 replay，只有 17 条（11.33%）能覆盖 HDF5 action 序列末端，平均前缀覆盖率为 87.35%。因此已冻结 HDF5 causal 作为完整离线 BC 协议，当前 runtime 仅用于闭环/对齐诊断。阶段/夹爪损失 targeted diagnostic 的两任务加权 query action MAE 为 baseline 0.11593、phase-balanced 0.11324、6x gripper-weighted 0.13180；阶段均衡有小幅候选收益，但单独加大夹爪损失没有收益。随后容量匹配的 CNN-only/CNN+GRU 对照得到 0.10707/0.11324，只有 task 7 transport gripper 有局部 GRU 优势，因此不主张稳定 memory benefit。CNN-only + phase-balanced 扩展至 5 个 source task、2 个 seed 后，25,238 个 query step 的加权 action MAE 为 0.12295；其配对 uniform-loss 对照为 0.11708。phase-balanced 将 release action MAE 从 0.18487 降至 0.16041，但 transport 从 0.12186 升至 0.13935，两个 seed 的总体差值方向一致，因此不作为通用 loss 默认值，基础离线策略采用 CNN-only + uniform action MSE。当前 runtime 的闭环、action/proprioception trace 和 RGB 快照诊断均已完成，下一项是阶段对齐诊断。validation、held-out、qualification 未访问，SPT/Knowledge 未更新，Module 未注册。详细结果见 [`docs/libero-environment-feasibility-2026-10-21.md`](libero-environment-feasibility-2026-10-21.md)、[`docs/libero-spatial-skill-family-audit-2026-10-21.md`](libero-spatial-skill-family-audit-2026-10-21.md)、[`docs/libero-demonstration-phase-structure-audit-2026-10-21.md`](libero-spatial-demonstration-phase-structure-audit-2026-10-21.md)、[`docs/libero-spatial-transfer-split-2026-10-21.md`](libero-spatial-transfer-split-2026-10-21.md)、[`docs/libero-cuda-training-path-smoke-2026-10-21.md`](libero-cuda-training-path-smoke-2026-10-21.md)、[`docs/libero-bc-policy-pipeline-check-2026-10-21.md`](libero-bc-policy-pipeline-check-2026-10-21.md)、[`docs/libero-single-spi-bc-learnability-diagnostic-2026-10-21.md`](libero-single-spi-bc-learnability-diagnostic-2026-10-21.md)、[`docs/libero-bc-closed-loop-attribution-2026-10-21.md`](libero-bc-closed-loop-attribution-2026-10-21.md)、[`docs/libero-bc-causal-alignment-diagnostic-2026-10-22.md`](libero-bc-causal-alignment-diagnostic-2026-10-22.md)、[`docs/libero-bc-hidden-state-diagnostic-2026-10-22.md`](libero-bc-hidden-state-diagnostic-2026-10-22.md)、[`docs/libero-stratified-error-diagnostic-2026-10-25.md`](libero-stratified-error-diagnostic-2026-10-25.md)、[`docs/libero-expert-replay-consistency-audit-2026-10-25.md`](libero-expert-replay-consistency-audit-2026-10-25.md)、[`docs/libero-replay-state-alignment-audit-2026-10-25.md`](libero-replay-state-alignment-audit-2026-10-25.md)、[`docs/libero-replay-semantics-audit-2026-10-25.md`](libero-replay-semantics-audit-2026-10-25.md) 和 [`docs/libero-current-runtime-replay-audit-2026-10-25.md`](libero-current-runtime-replay-audit-2026-10-25.md) 和 [`docs/libero-replay-input-alignment-audit-2026-10-25.md`](libero-replay-input-alignment-audit-2026-10-25.md)。
- Continual World 尚未在本仓库中启动正式实验。
- RoboCasa365 尚未进入正式训练，需先完成环境安装、接口适配和小规模协议预检。
- 本路线是研究设计冻结候选，不是正式实验结果。

### LIBERO 当前诊断项

最新进展：100-update historical-vs-runtime-native 前缀训练 pilot 两臂均为 `0/15` candidate success。500-update 对照已有独立配置并引用同一冻结前缀来源；运行结果尚未产生，完成后再判断是否需要扩大 seed 或转查动作/状态分布问题。

补充：paired loss 对照中的 uniform arm 与 phase-balanced arm 都对七个动作维度等权；配置里的 `gripper_action_weight=6.0` 在这两种 loss mode 中未生效，仅用于独立的 gripper-weighted arm。完整配对结果见 [CNN-only phase-loss paired diagnostic](libero-cnn-only-phase-loss-paired-diagnostic-2026-10-25.md)。对应的 phase-uniform CNN-only Policy 已在 5 个 source task、2 个 seed、每组 3 个固定 `candidate_nonregression` 初始状态上完成闭环检查，仅 task 3 / seed `20261021` 的 2/3 状态成功，总计 `2/30`。结果和边界见 [CNN-only phase-uniform closed-loop diagnostic](libero-cnn-only-uniform-closed-loop-diagnostic-2026-10-25.md)。随后抽取 7 条成功/失败轨迹，确认机械臂关节持续响应、部分失败夹爪关闭命令不足，同时也有足够长的 close run 却失败的轨迹；完整表和限制见 [Policy trace diagnostic](libero-cnn-only-uniform-policy-trace-diagnostic-2026-10-25.md)。固定 RGB 快照随后在同一 7 条轨迹上完成，7/7 重放通过来源成功/步数和模型 SHA 校验，共保存 134 帧；画面支持末端接近与初态敏感性是候选瓶颈，但不能区分 CNN、action head、覆盖或夹爪时序根因。结果和边界见 [RGB snapshot diagnostic](libero-cnn-only-uniform-policy-rgb-snapshot-diagnostic-2026-10-26.md)。已有 [分层 expert-state / free-run 误差诊断](libero-stratified-error-diagnostic-2026-10-25.md)、[输入替换敏感性审计](libero-policy-input-replacement-sensitivity-audit-2026-10-25.md) 和 [source-support replay 可行性审计](libero-source-support-replay-feasibility-audit-2026-10-25.md) 已覆盖阶段漂移与 HDF5/runtime 观测分叉，因此不重复这些实验。新的 runtime-native 前缀采集已覆盖 150 条示教，前缀为 63–197 步；其中 17 条完整执行 HDF5 action，133 条在当前 runtime 提前 success，详见 [runtime-native prefix collection](libero-runtime-native-prefix-collection-2026-10-27.md)。随后完成 100-update historical-vs-runtime-native training pilot，两个 observation arm 都是 `0/15` candidate success，详见 [runtime-native prefix training pilot](libero-runtime-native-prefix-training-pilot-2026-10-27.md)。下一步在相同前缀协议上扩大到 500 updates；这仍是机制诊断，不是 qualification。


逐步动作漂移、局部单步反事实、状态分布偏移、RGB/本体感知模态分解、两路相机分解、三 seed 相机复核、单变量 eye-in-hand 平移增强 pilot、固定 expert-state 合成平移敏感性、动作误差—状态偏移对齐、action-head 分解、source-SPI 条件覆盖审计及 relation×geometry×phase 分层误差诊断均已完成。增强臂降低了固定 expert-state 的相机敏感性和前 20 步动作误差，但 joint-state 首次越界仍在第 2–3 步；只训练 action head 的 query MSE 明显高于 full Policy，说明表示学习是必要条件。条件覆盖审计排除了 support/query 缺少 relation、source geometry 或动作阶段类别的简单断裂；分层诊断显示 task 4 open-container 条件的状态/RGB 漂移最大，部分任务的 release 夹爪误差放大。专家 replay 一致性、HDF5—当前重放状态对齐、回放语义冻结、当前 runtime 轨迹和输入对齐审计显示，task 3 demo 1 对 stabilization 敏感、task 7 demo 0 跨 seed/设置稳定失败，四条示教都在首个动作后偏离 HDF5 状态，包括成功控制示教，但已检查的 controller 配置和动作边界没有差异，当前 runtime 三次完整 trace 确定，两种 causal 输入视图从第一个动作后开始分叉；固定 task 3/7 Policy 的输入替换敏感性审计已经完成，shared-hidden action MAE 为 0.02185，独立 recurrent 分支为 0.03544；source-support 全量 replay 进一步显示只有 17/150 条示教具有完整当前-runtime 前缀覆盖。因此 HDF5 causal 已冻结为完整离线 BC 协议，当前 runtime 只用于闭环/对齐诊断。阶段/夹爪损失 targeted diagnostic 显示 phase-balanced 有小幅离线收益，6x gripper weighting 没有收益；容量匹配的 CNN-only/CNN+GRU 对照没有发现稳定的 GRU 优势，CNN-only + phase-balanced 已扩展到 5 个 source task 和 2 个 seed，并完成同结构 uniform-loss 配对；阶段均衡没有总体收益，release 改善以 transport 恶化为代价。下一项在当前 runtime candidate-nonregression 初始状态上检查 CNN-only + uniform loss 的闭环完成能力。所有这些仍是机制诊断，不是视觉修复有效性或正式 SPI 结果，不访问 official candidate、validation、held-out 或 qualification，不选 checkpoint。

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
- LIBERO 已完成 Python 3.10 联合环境、完整 `libero_spatial` 数据审计、10 个任务的 KECRL adapter smoke、候选 Skill Family 和数据隔离审计、CUDA 训练路径、候选 BC Policy/SPT/SPI pipeline、单 source SPI 固定预算学习与闭环归因。HDF5 时序审计确认 `obs[t]` 在 `action[t]` 后记录。随后完成 same-index/causal pre-action 对照：同初始化、同 seed、每臂 500 updates、同 `37,534` 有效动作步；专家回放 6/6，same-index 连续 GRU 闭环 1/9、causal 0/9，官方 candidate 初态均 0/3。时序修正单独未解决闭环失败。之后在两组锁定权重上完成 hidden-state follow-up：重训权重哈希与 v1 完全一致；continuous/每 10 步清零/stateless 的总成功数为 same-index `1/9, 1/9, 0/9`，causal `0/9, 0/9, 0/9`。同一因果 support/query 轨迹上，continuous teacher-forced MSE 均低于每窗重置（same-index-trained query `0.08413` vs `0.09318`；causal-trained query `0.08513` vs `0.09683`），但未转化成更高 free-run 成功率。这说明重置方式改变行为，不能说明它是失败的唯一来源；stateless 同样无收益。当前已实现冻结模型下的逐步 teacher-forced action 与 free-run state drift 记录器，准备在 6 个已开放 demo 初态上定位首个夹爪分歧、动作误差增长和两路 RGB/本体状态偏移；该检查仍是单 seed 诊断。validation、held-out、qualification 未访问，SPT/Knowledge 未更新，Module 未注册。详细结果见 [`docs/libero-environment-feasibility-2026-10-21.md`](libero-environment-feasibility-2026-10-21.md)、[`docs/libero-spatial-skill-family-audit-2026-10-21.md`](libero-spatial-skill-family-audit-2026-10-21.md)、[`docs/libero-demonstration-phase-structure-audit-2026-10-21.md`](libero-demonstration-phase-structure-audit-2026-10-21.md)、[`docs/libero-spatial-transfer-split-2026-10-21.md`](libero-spatial-transfer-split-2026-10-21.md)、[`docs/libero-cuda-training-path-smoke-2026-10-21.md`](libero-cuda-training-path-smoke-2026-10-21.md)、[`docs/libero-bc-policy-pipeline-check-2026-10-21.md`](libero-bc-policy-pipeline-check-2026-10-21.md)、[`docs/libero-single-spi-bc-learnability-diagnostic-2026-10-21.md`](libero-single-spi-bc-learnability-diagnostic-2026-10-21.md)、[`docs/libero-bc-closed-loop-attribution-2026-10-21.md`](libero-bc-closed-loop-attribution-2026-10-21.md)、[`docs/libero-bc-causal-alignment-diagnostic-2026-10-22.md`](libero-bc-causal-alignment-diagnostic-2026-10-22.md) 和 [`docs/libero-bc-hidden-state-diagnostic-2026-10-22.md`](libero-bc-hidden-state-diagnostic-2026-10-22.md)。
- Continual World 尚未在本仓库中启动正式实验。
- RoboCasa365 尚未进入正式训练，需先完成环境安装、接口适配和小规模协议预检。
- 本路线是研究设计冻结候选，不是正式实验结果。

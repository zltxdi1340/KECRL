# Crafter gather 族 SPT→SPI 集成 pilot 记录

日期：2026-10-16。状态：已完成非正式集成 pilot、动作链审计、学习率修正和来源场景池校准，正式实验仍未启动。
当前校准配置为 [`crafter_gather_spt_transfer_pilot_v3.yaml`](../../../configs/crafter_gather_spt_transfer_pilot_v3.yaml)，代码入口为 [`run_crafter_gather_spt_transfer_pilot.py`](../../../experiments/run_crafter_gather_spt_transfer_pilot.py)。主结果目录为 `results/crafter_gather_spt_transfer_pilot_v3_run1/`，代码 hash 为 `e43d678`；独立进程复现为 `results/crafter_gather_spt_transfer_pilot_v3_run2/`，代码 hash 为 `155b204`。历史 v1/v2 结果和 v2 独立复现均保留，不合并为正式结果。

这次 pilot 的目的，是确认研究问题已经被实例化为真实的 `SPT -> SPI -> 生成 CNN Policy -> 目标适应 -> 独立评价` 链路，并检查成本、梯度、场景规则和失败记录。它不是方法优选实验，也不用于证明 SPT 更新优于普通预训练。

## 做了什么

- 新建 `quiet-material-gather-v1` 场景。每个场景同时放置一棵树、一块石头和一块煤，材料位置距玩家 Manhattan 距离为 1--3；场景构造器检查相邻可采集格可达。移动、朝向、`do`、木镐要求、库存变化和 RGB 渲染继续走安装包中的原生 Crafter 逻辑。
- 建立 `spt:crafter:gather@1.0.0`，绑定 `tree/wood`、`stone/stone`、`coal/coal` 三个 SPI。树不要求木镐；石头和煤要求初始库存有一把木镐。策略输入仍是 64×64×3 RGB，目标绑定只通过公开 SPI context 进入生成器。
- 建立共享 CNN-only SPT：encoder/critic 共享，`context_to_actor` 为对 `actor.weight` 和 `actor.bias` 的完整线性残差生成器，初始权重为零。A、B、C 三臂使用相同容量。
- B 臂用 source support、adapted-policy query 和一阶 FOMAML 更新 SPT 的模板与生成器；C 臂在 v3 中使用与 B 相同的 support/query 候选场景池，分别收集两个 256 步块后合并做普通 PPO 更新；A 臂不做 source 更新。
- 三臂在 coal SPI 上分别从 support=0、1024、4096 步评价，每点 40 个 episode。目标适应使用新建的 Adam，不复用 source optimizer；评价保存并恢复 Torch/CUDA RNG。

## 账本和接口核对

| arm | replica | source 步数 | target 步数 | 目标 optimizer 更新前/后状态数 | source template/generator 变化 | coal 冷列保持零 |
|---|---:|---:|---:|---:|---|---|
| A | 11 | 0/0 | 4096/4096 | 0/10 | 否/否 | 是 |
| B | 11 | 8192/8192 | 4096/4096 | 0/10 | 是/是 | 是 |
| C | 11 | 8192/8192 | 4096/4096 | 0/10 | 是/是 | 是 |
| A | 12 | 0/0 | 4096/4096 | 0/10 | 否/否 | 是 |
| B | 12 | 8192/8192 | 4096/4096 | 0/10 | 是/是 | 是 |
| C | 12 | 8192/8192 | 4096/4096 | 0/10 | 是/是 | 是 |

契约和诊断测试覆盖 SPT/SPI 版本绑定、工具边界、配对 RGB reset、生成器冷列、候选隔离、全新 optimizer、C 臂生成器梯度、RGB 可见性、原生短动作链、初始朝向、并列最短动作、实际学习率、source material 绑定和来源场景池。当前定向测试为 `20 passed`，全量回归为 `438 passed`。v3 两次运行的预算/优化器审计均返回 `valid=true`、`source_optimizer_rates_verified=true`；两次独立 provenance/场景池审计均返回 `valid=true`、`provenance_valid=true`、`strict_source_scene_pool_equal=true`。run1/run2 的数值训练记录、episode/scene/RGB manifest 和优化器设置一致，复现报告保存在 run2 的 `reproduction_verification.json`。E2 三 fold bounded smoke 的聚合审计和 candidate 成本/provenance 预检均返回 `valid=true`，但不产生正式统计。旧 v2 的 provenance 完整性通过，但严格场景池检查失败，原因已保留在审计结果中。

## 可学习性审计

- 32 个审计布局、3 种材料共 96 个目标均在初始 native RGB 中留下可检测变化，最小变化为 39 个像素。这验证了抽查布局的像素可见性，不证明 CNN 已学会读出目标，也不覆盖所有训练布局。
- source/target 的动作计数都与交互步数一致，首次目标增量步数与 success 计数一致；A/B/C 的 source/target 预算、评价 episode 数、fresh target optimizer 和 coal 冷列均通过独立复核。
- v1 的 seed 11 偏向 `do`，seed 12 的 `do` 比例低；两者均未形成稳定采集链。动作边际分布不能独自定位 RGB 表示、head、状态覆盖或奖励/信用分配的责任。
- 后续[动作链诊断](crafter-gather-execution-chain-2026-10-15.md)中，32 个布局的三种材料均可由原生脚本在最多 6 步完成，8/16/32 步正对照都是 96/96；冻结策略在相同 RGB 状态上的阶段选择仍较弱。这是组件执行审计，不是学习策略成绩或自然 Module 资格。

## 场景池校准版 v3 结果

v2 的独立进程复现表明，旧实现的 B/C 实际 source scene 集合不相同：每个 seed 中 C 覆盖了 B 的 support 场景并额外使用其他场景，B 的 meta-query 场景没有进入 C。v3 为每个 slot 声明 64 个 support 和 64 个 query 候选 scene seed，B 和 C 使用相同的两个候选池；C 仍保持普通 PPO，只将两段 256 步 rollout 合并后更新。v3 每个 arm 的 source/target 步数没有增加。

| arm | seed 11：0/1024/4096 步 | seed 12：0/1024/4096 步 | 终点合计 |
|---|---:|---:|---:|
| A：更新前 SPT | 0.150 / 0.050 / 0.350 | 0.100 / 0.125 / 0.025 | 15/80（0.1875） |
| B：FOMAML SPT 更新 | 0.150 / 0.225 / 0.150 | 0.100 / 0.150 / 0.025 | 7/80（0.0875） |
| C：普通 PPO 预训练 | 0.050 / 0.150 / 0.200 | 0.025 / 0.325 / 0.275 | 19/80（0.2375） |

v3 的声明候选池在两个 replica 中均为 B/C 各 2,048 个 scene seed，交集和并集都为 2,048；不同策略因 episode 提前结束而实际访问的 seed 子集不同，这是允许的轨迹差异，不再把它误称为同一条轨迹。v3 终点 240 集有 41 次成功、199 次 external truncation、0 死亡。v2 的 B `25/80` 优势没有在场景池对齐后复现，因此不能作为 SPT 优势证据。

## 修复版 v2 结果（历史校准）

v1 runner 在建立 FOMAML optimizer 后，无条件把两个 source optimizer 的学习率改成 `policy_learning_rate=0.0003`，覆盖了声明的 `meta_learning_rate=0.001`。这是配置实现错误。`908eb4c` 修复后，B 使用配置的外循环学习率，C 使用普通 PPO 学习率，并将实际值写入结果供独立审计。内循环和 target PPO 仍为 `0.0003`。

v2 保持原来的两个 seed、来源 8,192 步、目标 4,096 步、奖励、网络和评价预算；新增运行用于修复实现错误，不按方法胜负追加预算。A/C 的全部原有训练与评价记录与 v1 run8 一致，复核文件为 `correction_verification.json`。v2 run2 是带完整 episode manifest 的独立进程复现，数值记录与 run1 一致；但严格 source scene pool 审计失败，不能作为场景池已对齐的结果。

| arm | seed 11：0/1024/4096 步 | seed 12：0/1024/4096 步 | 终点合计 |
|---|---:|---:|---:|
| A：更新前 SPT | 0.150 / 0.050 / 0.350 | 0.100 / 0.125 / 0.025 | 15/80（0.1875） |
| B：FOMAML SPT 更新 | 0.150 / 0.125 / 0.275 | 0.100 / 0.200 / 0.350 | 25/80（0.3125） |
| C：普通 PPO 预训练 | 0.100 / 0.275 / 0.200 | 0.175 / 0.100 / 0.000 | 8/80（0.1000） |

终点 240 集中有 48 次成功、192 次 external truncation，无死亡。B 的 pilot 终点高于 A/C，但场景池边界随后被发现不一致；该结果仅作为修错校准，不支持正式方法优势。

## 历史 v1 结果（实际外循环 lr=0.0003）

下表是每个 seed 的 coal 评价成功率曲线，顺序为 `0/1024/4096` target support 步。每点 40 个 episode。

| arm | seed 11 | seed 12 | 两 seed 的终点均值 |
|---|---:|---:|---:|
| A：更新前 SPT | 0.150 / 0.050 / 0.350 | 0.100 / 0.125 / 0.025 | 0.188 |
| B：FOMAML SPT 更新 | 0.225 / 0.275 / 0.175 | 0.075 / 0.250 / 0.000 | 0.088 |
| C：普通 PPO 预训练 | 0.100 / 0.275 / 0.200 | 0.175 / 0.100 / 0.000 | 0.100 |

两 seed 的最终评价合计为 A `15/80`、B `7/80`、C `8/80`。终点 240 集有 30 次成功、210 次 external truncation，无死亡。受控场景没有水、熔岩或敌人且初始生存资源充足；这不验证自然 Crafter 的生存能力。

v1 证明了接口和训练信号可运行，但不符合声明的外循环学习率，不能作为该声明配置的校准结果。它保留为实现错误诊断；v1/v2 不合并为更多独立 seed，也不能据单次修复比较确定低成功率的全部原因。

## 下一步

1. 预检已通过；candidate 仍保持 `ready_for_formal_training=false`，下一步是确认正式 hash 封存、统计版本和启动边界。
2. 边界冻结后再由用户决定是否启动正式预算；候选仍为 10 个独立 seed、E2 三个 leave-one-binding-out fold、每点 100 个评价 episode，pilot seed 11/12 不合并。
3. 正式组件实验完成后，才讨论 Module qualification、自然 Crafter 任务链和 Knowledge/Pipeline 更新；当前没有活动 SPT 切换、Module 注册或 Knowledge 更新。

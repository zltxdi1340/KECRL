# Crafter gather 族 SPT→SPI 集成 pilot 记录

日期：2026-10-15。状态：已完成一次非正式集成 pilot，正式实验仍未启动。
运行配置为 [`crafter_gather_spt_transfer_pilot_v1.yaml`](../configs/crafter_gather_spt_transfer_pilot_v1.yaml)，代码入口为 [`run_crafter_gather_spt_transfer_pilot.py`](../experiments/run_crafter_gather_spt_transfer_pilot.py)。结果目录 `results/crafter_gather_spt_transfer_pilot_v1_run5/` 保留在实验主机，未把原始结果纳入正式结果集。

这次 pilot 的目的，是确认研究问题已经被实例化为真实的 `SPT -> SPI -> 生成 CNN Policy -> 目标适应 -> 独立评价` 链路，并检查成本、梯度、场景规则和失败记录。它不是方法优选实验，也不用于证明 SPT 更新优于普通预训练。

## 做了什么

- 新建 `quiet-material-gather-v1` 场景。每个场景同时放置一棵树、一块石头和一块煤，材料位置距玩家 Manhattan 距离为 1--3；场景构造器检查相邻可采集格可达。移动、朝向、`do`、木镐要求、库存变化和 RGB 渲染继续走安装包中的原生 Crafter 逻辑。
- 建立 `spt:crafter:gather@1.0.0`，绑定 `tree/wood`、`stone/stone`、`coal/coal` 三个 SPI。树不要求木镐；石头和煤要求初始库存有一把木镐。策略输入仍是 64×64×3 RGB，目标绑定只通过公开 SPI context 进入生成器。
- 建立共享 CNN-only SPT：encoder/critic 共享，`context_to_actor` 为对 `actor.weight` 和 `actor.bias` 的完整线性残差生成器，初始权重为零。A、B、C 三臂使用相同容量。
- B 臂用 source support、adapted-policy query 和一阶 FOMAML 更新 SPT 的模板与生成器；C 臂用相同 source 任务和步数，直接对生成策略的 PPO 损失更新同一个模板与生成器；A 臂不做 source 更新。
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

新增契约测试覆盖了 SPT/SPI 版本绑定、三种 context、树与石/煤的工具边界、配对 RGB reset、生成器冷列、候选与 active 隔离、全新 optimizer，以及 C 臂梯度穿过共享生成器。该测试组为 `7 passed`；之后全量回归需以最终代码版本重新执行并记录。

## Pilot 结果

下表是每个 seed 的 coal 评价成功率曲线，顺序为 `0/1024/4096` target support 步。每点 40 个 episode。

| arm | seed 11 | seed 12 | 两 seed 的终点均值 |
|---|---:|---:|---:|
| A：更新前 SPT | 0.150 / 0.050 / 0.350 | 0.100 / 0.125 / 0.025 | 0.188 |
| B：FOMAML SPT 更新 | 0.225 / 0.275 / 0.175 | 0.075 / 0.250 / 0.000 | 0.088 |
| C：普通 PPO 预训练 | 0.100 / 0.275 / 0.200 | 0.175 / 0.100 / 0.000 | 0.100 |

两 seed 的最终评价合计为 A `15/80`、B `7/80`、C `8/80`。全部失败主要是 external truncation；pilot 没有出现死亡，因为该受控场景没有水、熔岩或敌人且初始生存资源充足。这不能说明自然 Crafter 的生存问题已经解决，只说明本组件 pilot 没有把死亡混入当前信号。

结果显示，8,192 source 步足以让 B/C 的共享模板和生成器实际更新，且完整链路可以运行；但两 seed 方差很大，B 没有稳定超过 A 或 C，所有臂都远低于候选 0.8。当前可得结论是“接口和训练信号已接通，但该预算下 wood→coal 迁移尚未显示稳定收益”，不能把它写成 SPT 有效或无效的正式结论。

## 下一步

1. 先把本次 pilot 的配置、代码 hash、seed 角色和结果路径封存，补完最终代码版本的全量回归与独立进程回放审计。
2. 检查 pilot 的学习信号：按 source/target 分别统计成功、external truncation、首次目标增量步数和 action 分布，确认 32 步上限与 RGB 场景没有制造不可学习的遮挡或朝向偏差。
3. 只有在一次性成本检查通过后，冻结正式 E1/E2 的候选预算；正式运行仍需 10 个独立 seed、E2 三个 leave-one-binding-out fold、100 个评价 episode 和预注册分析，pilot seed 11/12 不合并。
4. 正式组件实验完成后，才讨论是否进入 Module qualification、自然 Crafter 任务链和 Knowledge/Pipeline 更新；当前没有活动 SPT 切换、Module 注册或 Knowledge 更新。

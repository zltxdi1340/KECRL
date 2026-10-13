# Crafter wood>=3 局部接近增量干预诊断（2026-10-13）

本轮完成了上一轮提出的最小 approach 对照：两臂共同使用 `turn_ready_do`，只在
实验臂加入局部安全接近动作选择。冻结的 baseline-only head 成功率从
`52/90 = 57.8%` 升到 `84/90 = 93.3%`，mixed coverage head 从
`63/90 = 70.0%` 升到 `83/90 = 92.2%`。所有 360 集通过独立 CUDA 环境与
RGB 策略回放，策略动作概率最大误差为零。

结果支持“局部接近动作选择及其连续衔接是当前执行瓶颈”的判断。高成功率来自
诊断 oracle，真实 RGB executor 尚未通过独立资格检查。本轮没有训练策略、更新
encoder、更新 Knowledge/SPT 或注册 Module，也没有挖矿任务结论。

## 本轮要回答的问题

此前的[只读审计](crafter-wood3-approach-death-audit-2026-10-13.md)发现，原策略
经常不选择已经存在的接近动作，而选中后的单步位置改变和距离减少都稳定。只读
结果无法确认这个选择缺口能否解释完整任务的失败。本轮固定其余边界，在新的
环境/action seed 段比较：

- 对照：`turn_ready_do`，可采集时执行 do，清醒且相邻树可用时转向树。
- 实验：`turn_ready_do_approach`，共同规则以外，存在局部安全接近动作且没有
  sleep override 时，从有效接近动作中选择一个执行。

两臂每步都先执行一次原始 RGB policy categorical 采样。采样动作有效时保留它；
否则在有效方向中选择 policy 概率最高的动作，同概率取最小 action id。门控不
消耗额外 policy RNG。两臂共享的 ready/turn 替换不构成增量 approach 差异。

这里的“安全”仅是已有诊断定义下的局部可通行、最近无遮挡可见树距离减少。它不
提供全局路径、目标身份保持、敌人规避或存活保证。没有加入生存、探索或奖励干预。

## 固定协议与验证边界

- 固定三个 `100000` 交互步 CNN-only checkpoint，以及既有
  `matched_baseline_only` / `matched_mixed_coverage` standardized linear head；
  没有新拟合或重新选择 head。策略输入始终为 `RGB64x64x3`，动作替换规则使用
  特权诊断 snapshot，因此整个实验 executor 不是 RGB-only。
- 使用 `balance-object-insertion-order-v1` 稳定 wrapper；没有修改安装的 Crafter 包。
- policy seed 为 `0/1/2`，每个 head、每个条件、每个 seed 各 30 集，共 360 集。
  初始 wood 为 0，目标 `wood>=3`，每集最多 256 步，环境 length 为 10000，
  action allowlist 为 `0..6`。death、success 与 external truncation 分开记录。
- 主诊断环境/action seed 基数为 `171000000/175000000`，每个 policy seed
  增加 `1000000`，每集增加 1。配对条件的初始 RGB、snapshot、环境/action seed
  相同。首次实际 approach 动作差异之前的轨迹一致；共同观察到的各步采样 RNG
  一致。动作分歧后的状态、概率和采样动作允许不同。
- preflight 使用独立的 `166000000/168000000` seed 段、seed 0 每臂 5 集，
  共 20 集，全部通过独立回放，不并入主结果。
- 主结果产生 21833 个交互事件；旧 turn runner 的 6 集参考审计另外使用 361 步。
  seed 0 的 120 集在独立进程重复，四条 trace 的 canonical digest 全部一致；
  重复集不计入 360 集。
- 独立分析器重新运行全部 360 集，验证 RGB 概率、原始采样、执行动作、环境
  转移、门控规则、初始配对及共同前缀。12 组回放全部通过，最大概率误差为 0。
  encoder、policy、optimizer 和输入文件哈希保持不变，梯度为空，使用真实 CUDA。
- 针对性测试 `24 passed`，全量回归 `348 passed`。测试覆盖原生方向动作、规则
  优先级、sleep override、配对前缀、RNG、trace/计数篡改、死亡分母和短期删失。

本轮是冻结执行器的配对评估，不是训练预算比较。各臂相同的 episode 数和最大步数
构成评估协议；实际交互步数因成功、死亡和截断时点不同而不同，按结果如实报告。

## 完整 wood>=3 结果

| head | 条件 | 成功 | 死亡 | external truncation | 交互步数 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-only | `turn_ready_do` | 52/90 = 57.8% | 34 | 4 | 8732 |
| baseline-only | `turn_ready_do_approach` | 84/90 = 93.3% | 4 | 2 | 2606 |
| mixed coverage | `turn_ready_do` | 63/90 = 70.0% | 26 | 1 | 8148 |
| mixed coverage | `turn_ready_do_approach` | 83/90 = 92.2% | 6 | 1 | 2347 |

同一批配对评估的成功率增加分别为 35.6 和 22.2 个百分点，死亡减少 30 和 20 集。
平均每集使用的交互步数分别从 `97.0` 降到 `29.0`、从 `90.5` 降到 `26.1`，
这些均值包含失败集。更快完成采集与更少死亡同时出现；本轮没有独立识别时间
缩短、状态分布变化和伤害暴露变化分别贡献了多少，因此不能宣称生存能力已学会。

| policy seed | baseline 对照 | baseline +approach | mixed 对照 | mixed +approach |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 14/30 | 28/30 | 16/30 | 29/30 |
| 1 | 29/30 | 30/30 | 25/30 | 29/30 |
| 2 | 9/30 | 26/30 | 22/30 | 25/30 |

| head | 两臂都成功 | 仅对照成功 | 仅 +approach 成功 | 两臂都失败 |
| --- | ---: | ---: | ---: | ---: |
| baseline-only | 51 | 1 | 33 | 5 |
| mixed coverage | 61 | 2 | 22 | 5 |

三 seed、两种 head 的改善方向一致，仍有 3 对出现反向变化。这里只有三个冻结
policy seed，不能把 90 集或相互重叠的时步当作大量独立训练重复。这里的对照
成功率与上一批 turn 诊断不同，因为使用了新的 seed 段；增量效果只按本批配对
条件估计，不拿跨批次均值作因果比较。

## 接近动作与采集衔接

下表的有效选择分母为全部清醒接近机会，位置改变和距离减少分母为实际有效动作。
8 步后续窗口只观察当前接近动作之后的最多 8 个事件，任务成功或原生死亡使窗口
完整，外部截断未观察完的窗口按删失排除；窗口之间相关，不能当作独立资格集。

| head / 条件 | 有效接近选择 | 位置改变 / 距离减少 | 8 步内有 wood gain | 外部删失窗口 |
| --- | ---: | ---: | ---: | ---: |
| baseline 对照 | 1201/5370 = 22.4% | 1201/1201 | 446/1201 = 37.1% | 0 |
| baseline +approach | 703/703 = 100% | 703/703 | 703/703 = 100% | 0 |
| mixed 对照 | 1234/5844 = 21.1% | 1234/1234 | 555/1234 = 45.0% | 0 |
| mixed +approach | 795/795 = 100% | 795/795 | 696/793 = 87.8% | 2 |

实验臂分别触发 703/795 次 approach 门控，实际替换 475/509 个原采样动作。
实验轨迹上，原采样本来有效的比例仍只有 `228/703 = 32.4%` 和
`286/795 = 36.0%`，显示高成功率仍依赖门控。这些比例与对照访问了不同的状态
分布，不能解释成 head 学习进步。

两个实验臂的有效接近动作都没有当步原生终止，8 步完整后续窗口也没有死亡；
当步损伤分别发生 3/1 次，因此安全集合没有保证不受伤。baseline 的有效接近
后下一事件采到 wood 为 `167/703`，mixed 为 `171/795`；多步接近不必在
下一事件就采到 wood，完整任务成功率仍是核心结果。实验臂 92/90 次相邻转向
均进入 ready，下一步均 do 并采到一份 wood。

阶段漏斗按“进入 / 完成”统计：

| head / 条件 | wood=0 | wood=1 | wood=2 |
| --- | ---: | ---: | ---: |
| baseline 对照 | 90/74 | 74/60 | 60/52 |
| baseline +approach | 90/89 | 89/86 | 86/84 |
| mixed 对照 | 90/81 | 81/69 | 69/63 |
| mixed +approach | 90/90 | 90/88 | 88/83 |

## 剩余失败与死亡

失败归因在每集最后一个未完成的 wood 阶段上互斥统计。

| 最后阶段类别 | baseline 对照 | baseline +approach | mixed 对照 | mixed +approach |
| --- | ---: | ---: | ---: | ---: |
| 没有观察到无遮挡可见树 | 4 | 6 | 1 | 6 |
| 有接近机会但从未选择 | 8 | 0 | 2 | 0 |
| 有接近进展但未到达局部采集机会 | 26 | 0 | 23 | 1 |
| 到达局部机会后仍失败 | 0 | 0 | 1 | 0 |
| 合计 | 38 | 6 | 27 | 7 |

实验臂剩余 13 集失败，包括 10 次死亡和 3 次外部截断。12 集在最后阶段没有
观察到无遮挡可见树；它不意味着整集没有树，更不意味着地图没有树，多数已经
完成 wood=1 或 wood=2 后才进入这个阶段。另 1 集是 mixed seed 0 episode 3，
最后停在可见树但无安全局部接近动作的状态，256 步外部截断。

实验臂 10 次死亡都发生在没有无遮挡可见树的状态，wood=1/2 分别为 3/7 次；
adapter 伤害提示为 hostile/projectile 5 次、mixed/unknown 4 次、depletion 1 次。
这些是剩余探索、受阻路径和生存问题的线索，伤害提示不能替代环境因果标签。

上一份只读报告将全部接近机会上的同步死亡数误配到有效动作分母，本次已修正：
`28/7152`、`24/6120`、`20/7036`、`20/6476`。原始 CSV 计数没有变化；有效
动作当步死亡的真正计数另列在该报告。此修正不改变“多数接近机会没有被选择”的
结论，也不与本轮新的 seed 数据混用。

## 下一项诊断与 SPT/SPI 的关系

本轮已经证明，在当前 seed 段上，把局部 approach、turn、do 连起来可以高概率
完成 wood>=3。下一项应检查如何让 RGB executor 自己选择并衔接这些动作，
而不是将门控副本直接注册成技能：

1. 先复用本轮独立验证的轨迹，检查冻结 RGB encoder 和既有 head 对
   approach/turn/ready 状态的可读性、有效动作概率及整段动作链覆盖。按完整
   episode 隔离分析，检查与此前拟合/选择数据的 seed 重叠；本轮评估集只用于
   诊断，不回流成训练或选择数据。
2. 若继续做执行器学习 pilot，另开独立的训练、validation、heldout 和 fresh
   seed 段，固定 encoder、head 容量、各类样本数、优化预算和终止边界，以
   “接近—转向—采集”的动作链监督做匹配对照。teacher 只在训练数据生成时
   使用；fresh 执行器只接收 RGB，取消所有 ready/turn/approach 替换。此前
   mixed coverage 没有稳定收益，不能把单纯追加相似数据预设为有效。
3. 同时保留失败阶段、无目标/受阻状态、死亡和截断统计。先检查动作链能否从
   oracle 转移到 RGB policy，再根据剩余失败决定单独的探索或生存干预，避免
   多项变化叠加后无法归因。

上述学习 pilot 尚未启动。这仍是实例化 SPI 的 executor 可行性工作：确认需要
学会哪些动作边界，并验证技能在真实接口下能否执行。候选 `0.8` 仍待验证，
20 个独立资格 episode 至少 16 次成功也只是成功率门槛，不能替代完整 Module
契约。只有取消 oracle 后的 RGB executor 通过独立资格与完整契约检查，才进入
Module 注册、SPT/SPI 复用和持续任务链。随后仍需逐项验证 stone、coal、pickaxe
及跨技能组合；wood 结果不能直接推广到这些任务。

## 产物与复核

- [固定配置](../configs/crafter_wood3_turn_ready_do_approach_intervention_cuda_v1.yaml)
- [冻结干预 runner](../experiments/run_crafter_wood3_turn_ready_do_approach_intervention.py)
- [独立分析器](../experiments/analyze_crafter_wood3_turn_ready_do_approach_intervention.py)
- [针对性测试](../tests/test_crafter_turn_ready_do_approach_intervention.py)
- [主验证清单](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/verification.json)
- [汇总结果](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/pooled_outcomes.csv)
- [分 seed 结果](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/outcomes.csv)
- [配对集](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/paired_episodes.csv)
- [接近动作结果](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/approach_outcomes.csv)
- [阶段漏斗](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/stage_funnel.csv)
- [失败归因](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/failure_attribution.csv)
- [死亡时点](../results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_verified/death_timing.csv)
- [preflight 验证](../results/crafter_wood3_turn_ready_do_approach_preflight_cuda_20261013_v1/analysis_verified/verification.json)

Git 保存配置、代码、报告和紧凑 CSV/JSON；完整 JSONL trace、checkpoint、head
与安装包快照保留本机。紧凑摘要不能代替原始输入和独立回放。下列复核命令依赖
本机封存输入，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.analyze_crafter_wood3_turn_ready_do_approach_intervention \
  --input results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1 \
  --output results/crafter_wood3_turn_ready_do_approach_intervention_cuda_20261013_v1/analysis_recheck
```

# Crafter wood>=3 接近与死亡时点只读审计（2026-10-13）

本轮复用已经独立验证的 `turn_ready_do` 主 trace，对 3 个 policy seed、2 个 head、
`ready_do`/`turn_ready_do` 两个条件各 30 集进行只读分析。输入共 360 集、44,662 个
交互事件。分析没有启动 Crafter 环境，没有 CUDA 交互，没有策略更新、拟合、选择、
encoder 更新、Knowledge/SPT 更新或 Module 注册；外部截断按删失处理。所有结论都是
冻结轨迹上的相关性描述，不能当作干预的因果效果。

## 审计边界

- `approach` 定义为：当前尚未 ready、也没有清醒的相邻树转向机会，但存在至少一个
  局部可通行的安全移动动作。
- 接近进度使用当前无遮挡可见树的最近 Manhattan 距离是否减少判断。此指标只描述
  一步局部移动，不是全局导航距离。
- 本批原有干预只包含 `ready->do` 和可选的 `turn->ready->do`。approach 时的动作
  仍是原 RGB policy 的采样，因此 `sampled_valid_approach` 与
  `executed_valid_approach` 相同；本轮没有偷偷加入 approach oracle。
- 失败归因在该 episode 最后一个 wood 阶段上做互斥分类：没有观察到无遮挡树、可见
  但没有安全接近动作、存在接近机会却从未选择、选择后没有观察到距离进展、发生
  进展但没有到达局部采集机会、以及到达局部机会后仍失败。
- 死亡时点同时记录局部状态、阶段、资源、附近 hostile 数和 adapter 的伤害提示。
  伤害提示是诊断字段，不等于环境因果标签。

## 完整任务汇总

| head | 条件 | 成功 | 死亡 | external truncation | 失败集 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-only | `ready_do` | 24/90 = 26.7% | 56 | 10 | 66 |
| baseline-only | `turn_ready_do` | 38/90 = 42.2% | 43 | 9 | 52 |
| mixed coverage | `ready_do` | 43/90 = 47.8% | 44 | 3 | 47 |
| mixed coverage | `turn_ready_do` | 49/90 = 54.4% | 38 | 3 | 41 |

两种 head 的 turn 条件仍然比对应 ready 条件成功更多，但本审计不重新估计该干预的
因果效应，只复用此前已经完成独立回放和输入哈希检查的 trace。

## 接近动作审计

下表的“有效选择”分母是所有存在安全接近动作的时步；“距离减少”分母是已经执行
的有效接近动作；“下一步采到 wood”只在下一事件存在时计数，因此只是短期衔接描述。

| head / 条件 | 接近机会 | 有效选择 | 位置改变 | 距离减少 | 下一事件采到 wood | 同步死亡 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / `ready_do` | 7152 | 1291/7152 = 18.1% | 1291/1291 = 100% | 1290/1291 = 99.9% | 77/7119 = 1.1% | 28/1291 = 2.2% |
| baseline / `turn_ready_do` | 6120 | 1113/6120 = 18.2% | 1113/1113 = 100% | 1112/1113 = 99.9% | 61/6092 = 1.0% | 24/1113 = 2.2% |
| mixed / `ready_do` | 7036 | 1481/7036 = 21.0% | 1481/1481 = 100% | 1477/1481 = 99.7% | 122/7015 = 1.7% | 20/1481 = 1.4% |
| mixed / `turn_ready_do` | 6476 | 1337/6476 = 20.6% | 1337/1337 = 100% | 1336/1337 = 99.9% | 78/6456 = 1.2% | 20/1337 = 1.5% |

这里的主要缺口是**选择**：大约四分之三到五分之四的接近机会没有被原策略选中。
一旦选择有效动作，Crafter 的局部执行本身几乎总能改变位置并减少最近树距离。因此
“动作执行器不能移动”不是当前第一瓶颈；需要继续区分 RGB policy 是否能稳定识别
目标、保持目标和在较长接近过程中存活。下一事件没有立即采到 wood 是正常的，因为
接近通常需要多步，不能用该列替代完整采集资格。

## 失败阶段归因

四个 arm 合计 206 个失败 episode：

| 最后失败阶段类别 | 计数 | 占失败集 |
| --- | ---: | ---: |
| 没有观察到无遮挡可见树 | 11 | 5.3% |
| 可见树但没有安全接近动作 | 1 | 0.5% |
| 有接近机会但从未选择 | 31 | 15.0% |
| 选择接近但没有观察到进展 | 0 | 0.0% |
| 有接近进展但没有到达局部采集机会 | 137 | 66.5% |
| 到达局部采集机会后仍失败 | 26 | 12.6% |

按 arm 展开如下；类别在每个失败 episode 的最后 wood 阶段上互斥。

| head / 条件 | 无树 | 无安全动作 | 未选择 | 无进展 | 进展但未到达局部机会 | 到达后失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / `ready_do` | 4 | 0 | 8 | 0 | 37 | 17 |
| baseline / `turn_ready_do` | 3 | 0 | 7 | 0 | 42 | 0 |
| mixed / `ready_do` | 2 | 1 | 8 | 0 | 27 | 9 |
| mixed / `turn_ready_do` | 2 | 0 | 8 | 0 | 31 | 0 |

turn 条件下 73 个失败 episode 属于“已经有接近进展但仍未到达局部机会”，而不是
“到达相邻树后无法转向”。这说明补齐 turn->ready->do 后，剩余失败主要在更早的
目标保持、较长接近路径或存活过程中发生。

## 死亡时点与资源上下文

共记录 181 次死亡。死亡前最后一个局部状态为：

| 死亡前状态 | 次数 | 占死亡 |
| --- | ---: | ---: |
| `approach` | 92 | 50.8% |
| 没有无遮挡可见树 | 50 | 27.6% |
| 有可见树但无安全接近动作 | 28 | 15.5% |
| `sleep_override` | 7 | 3.9% |
| `turn` | 4 | 2.2% |
| `ready` | 0 | 0.0% |

死亡发生在 wood 阶段 0/1/2 的次数为 `93/49/39`。adapter 伤害提示为
`hostile_or_projectile` 122 次（67.4%）、`mixed_or_unknown` 43 次（23.8%）和
`depletion` 16 次（8.8%）。死亡前 health 的均值/中位数为 `2.12/2`，food、drink、
energy 均值分别为 `3.82/3.67/4.73`；附近 hostile 数均值为 `1.41`、中位数为 `1`。
这些数值支持优先检查敌人接触和移动过程中的生存，但没有证明 hostile 是每次死亡的
唯一原因，也没有证明资源奖励或死亡惩罚就能解决问题。

分 arm 的死亡前状态计数为：

| head / 条件 | approach | 无可见树 | 可见但无安全动作 | turn | sleep |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / `ready_do` | 28 | 18 | 3 | 4 | 3 |
| baseline / `turn_ready_do` | 24 | 16 | 1 | 0 | 2 |
| mixed / `ready_do` | 20 | 9 | 14 | 0 | 1 |
| mixed / `turn_ready_do` | 20 | 7 | 10 | 0 | 1 |

## 诊断结论

1. 当前首要问题是 RGB policy 的**接近动作选择和长程接近过程**。局部安全动作一旦
   被选中，执行和单步距离减少都稳定；因此继续只修正原生动作执行不会触及主要缺口。
2. `turn_ready_do` 已经消除了相邻树处的转向衔接缺口，但大量失败发生在到达局部
   采集机会之前，且死亡多发生在 approach 或没有可见目标的状态。
3. mixed coverage 的有效接近选择率高于 baseline（约 20.6%–21.0% 对 18.1%–18.2%），
   但这仍是相关的 head 对照，不能把它解释成 GRU 记忆或某一表示因素的因果证明。
4. wood>=3 仍未达到候选 `0.8` Module 门槛；本轮也没有重新验证 stone、coal、
   pickaxe，因此不能把结果推广到挖矿或持续任务链。

## 下一项与 SPT/SPI 的关系

下一项固定同一 encoder、head、稳定 object-order wrapper 和 PPO/资格边界，在新的
独立 seed 段做最小 `approach` 对照：以 `turn_ready_do` 为共同基础，只增加“存在安全
接近动作时选择局部距离减少动作”的 oracle 条件，保留原始 RGB policy 的采样和 RNG，
不加入生存、探索、全局地图或资源奖励干预。比较完整 wood>=3 成功率、死亡时点、
最后阶段归因和 approach 后续轨迹，检查“选择缺口”是否足以解释成功率差距。

该 oracle 只用于定位 RGB executor 的能力边界，不能注册正式 Module，也不更新
Knowledge/SPT。只有在不依赖 oracle 的 RGB policy 在独立资格 episode 中稳定达到
候选门槛，并通过完整 Module 契约检查后，才进入 SPT/SPI 的技能复用；届时仍需逐项
验证 stone、coal、pickaxe 以及跨技能持续任务链。

## 产物与复核

- [固定配置](../configs/crafter_wood3_approach_death_audit_v1.yaml)
- [只读分析器](../experiments/analyze_crafter_wood3_approach_death.py)
- [针对性测试](../tests/test_crafter_wood3_approach_death.py)
- [episode 指标](../results/crafter_wood3_approach_death_audit_20261013_v1/episode_metrics.csv)
- [失败归因](../results/crafter_wood3_approach_death_audit_20261013_v1/failure_attribution.csv)
- [死亡时点](../results/crafter_wood3_approach_death_audit_20261013_v1/death_timing.csv)
- [接近动作结果](../results/crafter_wood3_approach_death_audit_20261013_v1/approach_outcomes.csv)
- [按 arm 汇总](../results/crafter_wood3_approach_death_audit_20261013_v1/diagnosis_summary.csv)
- [验证清单](../results/crafter_wood3_approach_death_audit_20261013_v1/verification.json)

完整 trace、冻结 head 与 checkpoint 保留在本机结果目录；Git 只保存配置、代码、
报告和紧凑 CSV/JSON。复核依赖原始输入，紧凑产物不能替代独立环境和策略回放。

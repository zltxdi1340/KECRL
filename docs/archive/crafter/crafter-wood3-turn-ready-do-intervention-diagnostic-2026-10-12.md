# Crafter wood>=3 局部转向增量干预诊断（2026-10-12）

本轮冻结同一组 100k CNN-only encoder 与 matched head，在两臂都已补齐 ready->do
的条件下，仅增加局部有效转向，检查“已经邻近树但未对准”的执行缺口。

同批 fresh episode 中，baseline-only 从 `24/90 = 26.7%` 提高到
`38/90 = 42.2%`，mixed coverage 从 `43/90 = 47.8%` 提高到 `49/90 = 54.4%`。
三个 seed 的成功数均增加，但完整任务仍失败很多；局部转向是瓶颈之一，补齐
turn->ready->do 还不足以形成可靠 wood executor。两臂都使用 oracle，不能注册 Module。

## 固定协议

- 输入为既有 `matched_baseline_only`、`matched_mixed_coverage` 标准化线性 head；
  baseline-only 不是原始 PPO actor。本轮没有拟合、选择或训练。
- 三个 policy seed，每个 head 每臂 30 集，共 360 集、180 对；初始 wood=0，目标
  wood>=3，每集最多 256 步。环境/action seed 起点为 `161000000/165000000`，
  policy seed 间 stride 为 1,000,000，与上一轮主评估及本轮 preflight 的 seed 段分离。
- 两臂均每步先从原 RGB policy 分布采样一次，再读取 collection oracle。在 ready 时
  均执行 do；`ready_do` 在其余状态保留原采样，`turn_ready_do` 仅在清醒、容量允许、
  存在无 object 遮挡的相邻树且尚未 ready 时执行有效转向。
- 原采样已属于有效转向集合就保留；否则选择集合中原策略概率最高的方向，概率并列
  时选择最小 action id。规则不增加随机抽样，也不使用全局地图。
- 转向使用 Crafter 原生方向动作，朝树按方向键会改变朝向而留在原位。转向后的下一
  状态仍须重新满足 ready 条件才能执行 do；不假定转向期间不会死亡或树不会被
  object 挡住。
- 不干预 approach、探索或生存。不把局部可采机会或距离减少当作全局导航安全保证。
  两臂的环境轨迹及实际步数可以随动作、成功和死亡而不同。
- 配对从相同环境/action seed、初始 RGB/snapshot 开始。两臂共有的 ready 替换不
  结束共同轨迹前缀，只有新增转向造成的首次实际动作差异才结束前缀；共同步仍检查
  相同的抽样后 RNG digest。
- 使用既有稳定 object-order wrapper 和真实 CUDA，不修改安装包。所有 policy、
  encoder、optimizer 状态冻结，梯度为空，没有 Knowledge/SPT 更新或 Module 注册。

## 验证协议

新增 24 项针对性测试与全量回归已通过（`316 passed`）。测试覆盖四个方向的原生
turn->ready->do、保留原有效采样、固定概率并列规则、ready 优先、睡眠覆盖、不干预
approach，以及配对中共有 ready 替换和首次新增转向的区分。

20 集 CUDA preflight 使用独立的 `151000000/155000000` seed 段，已完成独立回放。
控制臂与上一版 ready_do runner 的轨迹一致，策略采样、RNG、干预规则和环境转移均
相符；preflight 不并入主结果。

360 个主 episode 共 44,662 步，全部经独立进程重放：重新从 RGB 计算分布及原动作
采样，最大概率差为 `0.0`，采样后 RNG、before/after snapshot、RGB、奖励、wood、
原生 done 全部相符。还核对采集规则、health delta、连续性、milestone、终止和外部
截断边界，以及 ready/turn gate 与实际替换计数。

180 对 episode 的初始状态、共同步抽样 RNG、首次新增转向前的轨迹前缀均通过。
seed 0 四条 trace 在独立进程完整重跑，文件重新加载后的 canonical digest 与主运行
一致。主运行六次控制臂参考审计与旧 runner 一致（另有 658 次参考交互）。
模型/optimizer 的前后状态比较、梯度为空及真实 CUDA 检查通过；checkpoint、head、
数据、源码与环境安装包前后哈希一致。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.run_crafter_wood3_turn_ready_do_intervention \
  --config configs/crafter_wood3_turn_ready_do_intervention_cuda_v1.yaml \
  --output results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.analyze_crafter_wood3_turn_ready_do_intervention \
  --input results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1 \
  --output results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified
```

运行与分析拒绝覆盖已有目录；复现使用新目录。collection oracle 不输入 RGB policy，
但会改变执行动作，因此两种诊断条件都不是 RGB-only executor，不能进入正式 Module
资格。`0.8` 仍是待验证候选门槛，本轮不验证 stone、coal、pickaxe 或持续任务链。

## 完整任务结果

| head | 条件 | wood>=3 成功 | 死亡 | external truncation | 交互步数 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-only | ready_do | 24/90 = 26.7% | 56 | 10 | 12,613 |
| baseline-only | turn_ready_do | 38/90 = 42.2% | 43 | 9 | 10,318 |
| mixed coverage | ready_do | 43/90 = 47.8% | 44 | 3 | 11,495 |
| mixed coverage | turn_ready_do | 49/90 = 54.4% | 38 | 3 | 10,236 |

这是新 seed 段内两臂的配对比较；不能用本轮 ready_do 与上一轮另一 seed 段的
33.3%/51.1% 直接计算转向收益。两种 head 的增量分别为 14/90 和 6/90 个成功
episode，实际步数也因路径改变、成功或死亡提前终止而变化。本轮不作等交互预算
训练结论。

分 seed 成功数（各格分母均为 30）：

| seed | baseline ready_do | baseline turn_ready_do | mixed ready_do | mixed turn_ready_do |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 11 | 14 | 17 |
| 1 | 14 | 17 | 15 | 16 |
| 2 | 6 | 10 | 14 | 16 |

| head | ready_do 独有成功 | turn_ready_do 独有成功 | 两者都成功 | 两者都失败 |
| --- | ---: | ---: | ---: | ---: |
| baseline-only | 2 | 16 | 22 | 50 |
| mixed coverage | 2 | 8 | 41 | 39 |

两种 head 各有两集在控制臂成功、转向臂失败。改变转向会改变后续环境路径、随机
过程和生存时长；本轮支持该固定转向干预对本批完整任务有净增益，不把收益拆成
单步转向、时间节省和生存变化的独立效应。

## 转向后果与阶段漏斗

| head / 条件 | turn 状态步数 | 实际有效转向 | 转向后 ready | 下一步采到 wood | turn gate / 实际转向替换 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline ready_do | 1,015 | 31（3.1%） | 31/31 | 31/31 | 0 / 0 |
| baseline turn_ready_do | 82 | 82（100%） | 82/82 | 82/82 | 82 / 70 |
| mixed ready_do | 740 | 31（4.2%） | 31/31 | 31/31 | 0 / 0 |
| mixed turn_ready_do | 99 | 99（100%） | 99/99 | 99/99 | 99 / 86 |

turn 状态为未 ready 的 awake adjacent opportunity。本批所有有效转向当步存活，
after snapshot 均 ready，下一动作均为 do 并采到一份 wood；没有转向后死亡、仍不
ready 或下一事件因外部截断缺失的记录。这是局部衔接结果，不是生存保证。

阶段漏斗按“进入/完成”记录：

| head / 条件 | wood=0 | wood=1 | wood=2 |
| --- | ---: | ---: | ---: |
| baseline ready_do | 90/57 | 57/38 | 38/24 |
| baseline turn_ready_do | 90/61 | 61/44 | 44/38 |
| mixed ready_do | 90/67 | 67/55 | 55/43 |
| mixed turn_ready_do | 90/70 | 70/58 | 58/49 |

失败阶段中，baseline ready_do 的“未观察到可用相邻树”计数为 wood=0/1/2 的
`29/13/7`，另有 `4/6/7` 集有相邻机会但从未进入 ready；转向臂分别为
`29/17/6` 和 `0/0/0`。mixed ready_do 的两类计数为 `20/10/8` 和 `3/2/4`，
转向臂为 `20/12/9` 和 `0/0/0`。转向臂剩余 52/41 次失败在最后失败阶段都没有
观察到可用相邻树，其中死亡 43/38 次、外部截断 9/3 次。没有“阶段无动作观察”或
“仅睡眠/容量受限”类别。

“未观察到可用相邻树”不等于整个地图没有树，也不能直接归因为探索不足：树可能
可见但未靠近，局部移动可能受阻，或策略在接近前死亡。该类别不确认伤害来源或
失败的因果贡献。

## 下一项与 SPI 关系

上述只读审计已完成，见[接近与死亡时点审计报告](crafter-wood3-approach-death-audit-2026-10-13.md)。
冻结轨迹中 18.1%–21.0% 的 approach 机会由原策略选出有效动作，但选中后位置改变率
为 100%、距离减少率为 99.7%–100%；181 次死亡中 92 次发生在 approach 状态，50 次
发生在没有可见无遮挡树的状态。206 个失败中有 137 个已经出现接近进展却没有到达
局部采集机会。

下一项固定 encoder、head 和稳定 wrapper，只增加 approach 的最小 oracle 对照，继续
保留原采样、RNG、终止和截断边界，不同时新增生存或探索干预。该动作集合只描述
局部通行和距离减少，不提供全局路径、敌人规避或生存保证；oracle 仍不能注册
Module 或更新 Knowledge/SPT。

目前的 SPI 证据是：到达可用相邻树后，turn->ready->do 能完成局部采集，但真实
RGB policy 尚未稳定学会选择与衔接；到达该机会前仍大量失败。后续需要把必要的
执行环节学到 RGB executor，并通过独立成功率和完整 Module 契约检查；oracle 副本
不能直接接入正式 Module，SPT/Knowledge 更新边界不变。

## 产物

- [固定配置](../../../configs/crafter_wood3_turn_ready_do_intervention_cuda_v1.yaml)
- [冻结干预 runner](../../../experiments/run_crafter_wood3_turn_ready_do_intervention.py)
- [独立回放分析器](../../../experiments/analyze_crafter_wood3_turn_ready_do_intervention.py)
- [针对性测试](../../../tests/test_crafter_turn_ready_do_intervention.py)
- [验证清单](../../../results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified/verification.json)
- [汇总结果](../../../results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified/pooled_outcomes.csv)
- [分 seed 结果](../../../results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified/outcomes.csv)
- [阶段漏斗](../../../results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified/stage_funnel.csv)
- [局部转向后果](../../../results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified/turn_outcomes.csv)
- [episode 配对](../../../results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1/analysis_verified/paired_episodes.csv)

Git 保存配置、代码、报告和紧凑分析产物；完整 trace、冻结 head 与 checkpoint 保留
在本机结果目录。完整复核依赖原始输入，紧凑 CSV 不能替代环境和策略回放。

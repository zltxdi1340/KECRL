# Crafter wood>=3 ready->do 干预诊断（2026-10-12）

本轮冻结已有 CNN encoder 和两种 matched head，只在已经可采集时将实际执行动作改为
`do`，检验上一轮观察到的 ready->do 衔接缺口是否影响完整自然任务。

同批 fresh episode 中，baseline-only 成功率从 `21/90 = 23.3%` 提高到
`30/90 = 33.3%`，mixed coverage 从 `28/90 = 31.1%` 提高到 `46/90 = 51.1%`。
这支持 ready->do 是执行瓶颈之一，但补齐它仍不能稳定完成任务。干预使用 collection
oracle，不能算作已学会的 RGB-only SPI executor，也不能注册 Module。

## 固定协议与比较边界

- 输入为冻结的 100k CNN-only checkpoint 和上一轮已锁定的
  `matched_baseline_only`、`matched_mixed_coverage` 标准化线性 head；前者不是原始
  PPO actor。本轮没有拟合或重新选择 head。
- 三个 policy seed，每个 head 每种条件 30 集，共 360 个主评估 episode、180 对。
  初始 wood=0，目标 wood>=3，每集最多 256 步；主评估共 49,032 次交互。
- 每个 head 内的 `native` / `ready_do` 使用相同环境 seed、action RNG seed 和初始
  RGB/snapshot。环境 seed 起点为 `141000000`，action seed 起点为 `145000000`，
  policy seed 间 stride 为 1,000,000。
- 每步先从原 RGB policy 分布采样一次并保存动作、概率与抽样后 RNG digest，再读取
  诊断 snapshot。`native` 原样执行；`ready_do` 仅在 `ready_to_collect=true` 时执行
  do，其他状态执行原采样动作。原采样已为 do 时只计 gate，不计替换。
- ready 使用原生条件：面对 tree、目标格无 object、wood 容量未满、没有睡眠覆盖。
  不干预 turn、approach、探索或生存行为。
- 使用既有稳定 object-order wrapper 和真实 CUDA，未修改安装的 Crafter 包。
  主运行前有独立 seed 段的 20 集 preflight；seed 0 的四条主 trace 另在独立进程
  完整重跑，digest 全部一致。这些检查不并入主成功率分母。
- 初始状态和抽样规则受控，第一次实际动作替换之后允许环境路径分歧；每臂的实际
  步数随成功/死亡提前终止而变化。本轮不是等交互预算的训练对照，也不是旧批次
  成功率的直接重测。
- policy/encoder/optimizer 无更新，梯度为空，没有 Knowledge/SPT 更新、正式训练
  或 Module 注册。

## 完整任务结果

| head | 条件 | 成功 | 死亡 | external truncation | 交互步数 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-only | native | 21/90 = 23.3% | 66 | 3 | 13,163 |
| baseline-only | ready_do | 30/90 = 33.3% | 57 | 3 | 11,388 |
| mixed coverage | native | 28/90 = 31.1% | 60 | 2 | 13,762 |
| mixed coverage | ready_do | 46/90 = 51.1% | 44 | 0 | 10,719 |

分 seed 结果（每格分母均为 30）：

| seed | baseline native | baseline ready_do | mixed native | mixed ready_do |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 4 | 5 | 14 |
| 1 | 11 | 15 | 14 | 13 |
| 2 | 8 | 11 | 9 | 19 |

baseline 三个 seed 的成功数都提高，mixed 的 seed 1 从 14 降到 13；不能把 pooled
收益写成所有 seed 都改善。最高单 seed 条件也只有 `19/30 = 63.3%`。

配对结果：

| head | native 独有成功 | ready_do 独有成功 | 两者都成功 | 两者都失败 |
| --- | ---: | ---: | ---: | ---: |
| baseline-only | 2 | 11 | 19 | 58 |
| mixed coverage | 3 | 21 | 25 | 41 |

干预会改变后续位置、环境随机过程和生存时长，因此某些 native 成功的 episode 会在
干预下失败。这里只报告固定诊断协议下的完整轨迹结果，不把净成功差分解成单步 do、
节省时间或生存改善的独立效应，也不以相关步级观察代替独立资格 episode。

## 衔接与阶段漏斗

| head / 条件 | ready 步数 | ready 时 do | ready 时方向动作但位置不变 | gate / 实际替换 |
| --- | ---: | ---: | ---: | ---: |
| baseline native | 848 | 103（12.1%） | 556 | 0 / 0 |
| baseline ready_do | 121 | 121（100%） | 0 | 121 / 86 |
| mixed native | 1,269 | 121（9.5%） | 927 | 0 / 0 |
| mixed ready_do | 171 | 171（100%） | 0 | 171 / 122 |

gate 条件下每次 ready 都立即采到一份 wood，因此 ready 步数下降不能解释成采集
机会减少；native 中的重复 ready 访问包括原地阻塞。两种 gate 条件共 292 次 ready+do
全部采集成功；本批没有成功与原生死亡同一步发生的 episode，分析仍分别记录这两类
事件。步级比例仅为描述统计。

阶段漏斗以下按“进入/完成”记录，wood=0/1/2 分别对应下一份 wood：

| head / 条件 | wood=0 | wood=1 | wood=2 |
| --- | ---: | ---: | ---: |
| baseline native | 90/51 | 51/31 | 31/21 |
| baseline ready_do | 90/52 | 52/39 | 39/30 |
| mixed native | 90/54 | 54/39 | 39/28 |
| mixed ready_do | 90/70 | 70/55 | 55/46 |

失败类别只使用失败的最后一个 wood 阶段内的观察，按“未观察到无 object 遮挡的相邻
树 → 睡眠/容量受限 → 从未进入 ready 朝向 → 已 ready 但未采集”的顺序互斥归类。
它们描述在哪个执行环节停止，不直接识别死亡的伤害来源或失败的因果贡献。

| head / 条件 | 未观察到可用相邻树 | 有相邻机会但从未 ready | 已 ready 但未采集 | 总失败 |
| --- | ---: | ---: | ---: | ---: |
| baseline native | 34 | 24 | 11 | 69 |
| baseline ready_do | 34 | 26 | 0 | 60 |
| mixed native | 22 | 16 | 24 | 62 |
| mixed ready_do | 31 | 13 | 0 | 44 |

本批没有“阶段无动作观察”或“仅睡眠/容量受限”类别。gate 后的 60/44 次失败全部发生
在该阶段第一次 ready 之前，分别有 57/44 次死亡。baseline 的主要损失仍在第一份
wood 前（38 次失败）；mixed 还有 20/15/9 次失败分别停在 wood=0/1/2。
“未观察到可用相邻树”不能直接等同于地图没有树或仅为探索不足；局部接近、路径阻塞
和生存时长都可能影响这个类别。

## 验证与复现

独立分析进程重放全部 360 个主 episode 的 49,032 步，重新从 RGB 计算策略分布和
原动作采样，最大概率差为 `0.0`；抽样后 RNG、before/after snapshot、RGB、奖励、
wood、原生 done 全部相符。还校验连续性、health delta、采集规则、milestone、终止
与截断边界、gate 及实际替换计数。

180 对 episode 的初始状态、共同步的抽样 RNG 和首次替换前的轨迹前缀全部通过。
主运行六次 native 参考审计与旧 runner 在不读取 oracle 时的轨迹一致；独立分析器
再次加载 seed 0 的四条跨进程 trace，确认与主 trace 完全一致。模型/optimizer 的
前后状态比较，以及 checkpoint、head、数据、代码和环境安装包的前后哈希检查全部
通过。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.run_crafter_wood3_ready_do_intervention \
  --config configs/crafter_wood3_ready_do_intervention_cuda_v1.yaml \
  --output results/crafter_wood3_ready_do_intervention_cuda_20261012_v1

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.analyze_crafter_wood3_ready_do_intervention \
  --input results/crafter_wood3_ready_do_intervention_cuda_20261012_v1 \
  --output results/crafter_wood3_ready_do_intervention_cuda_20261012_v1/analysis_verified
```

运行与分析都拒绝覆盖已有目录；复现时使用新输出目录。新增 12 项针对性测试覆盖
gate 边界、原采样保留、native 与旧 runner 一致、配对前缀与 RNG、采集和死亡分别
计数，以及锁定协议拒绝范围变更。全量回归为 `292 passed`。

## 下一项与 SPI 关系

下一项冻结同一组 head/encoder/wrapper，用新 seed 段配对比较 `ready_do` 与只增加
局部有效转向的 `turn->ready->do` 诊断副本；继续每步先采样、记录替换、核对首次
替换前前缀与阶段漏斗。它定位的是“已经邻近树但未对准”的增量贡献，不同时干预
approach 或生存，也不将局部距离减少解释成全局导航安全。

如果转向干预仍留下大量未接近树的失败，再检查 approach 的阻塞、可见树距离和
死亡前路径；若转向有稳定增益，再据此收紧只用 RGB 学习选择与衔接的目标。本轮
还未实现或验证 RGB-only ready gate，不能直接把 oracle 规则接入正式 SPI 执行器。

局部转向增量诊断已完成，见[turn->ready->do 报告](crafter-wood3-turn-ready-do-intervention-diagnostic-2026-10-12.md)。
它显示转向能进一步改善本批完整 wood 任务，但剩余失败集中在到达相邻树之前；下一项
转为只读审计 approach 与死亡时点。

这轮工作为 wood SPI 的执行与学习规格提供可区分的证据。SPT/SPI 的理论边界不变：
只有真实 RGB executor 通过独立成功率评估及完整 Module 契约检查，才可注册和复用。
`0.8` 仍是待验证候选门槛，本轮不是 Module 资格评估；stone、coal、pickaxe 和持续
任务链没有新结论。

## 产物

- [运行配置](../configs/crafter_wood3_ready_do_intervention_cuda_v1.yaml)
- [冻结干预 runner](../experiments/run_crafter_wood3_ready_do_intervention.py)
- [独立重放分析器](../experiments/analyze_crafter_wood3_ready_do_intervention.py)
- [针对性测试](../tests/test_crafter_ready_do_intervention.py)
- [验证清单](../results/crafter_wood3_ready_do_intervention_cuda_20261012_v1/analysis_verified/verification.json)
- [完整结果](../results/crafter_wood3_ready_do_intervention_cuda_20261012_v1/analysis_verified/pooled_outcomes.csv)
- [分 seed 结果](../results/crafter_wood3_ready_do_intervention_cuda_20261012_v1/analysis_verified/outcomes.csv)
- [阶段漏斗](../results/crafter_wood3_ready_do_intervention_cuda_20261012_v1/analysis_verified/stage_funnel.csv)
- [episode 配对](../results/crafter_wood3_ready_do_intervention_cuda_20261012_v1/analysis_verified/paired_episodes.csv)

Git 保存配置、代码、报告及紧凑分析产物；原始 JSONL trace、冻结 artifact 和 checkpoint
留在本机结果目录。复核依赖报告所列的原始输入，紧凑 CSV 不能替代完整重放。

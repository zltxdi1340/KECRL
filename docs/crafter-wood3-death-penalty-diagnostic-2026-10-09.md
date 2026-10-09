# Crafter wood>=3 terminal death penalty 诊断（2026-10-09）

本轮是正式实验前的可行性 pilot，只验证一个预先固定的奖励改动。所有结果均为
`formal_result=false`，不注册 Module，不更新 Knowledge Bank、SPT，也不启动正式训练。
结论仅适用于 `wood >= 3`。

固定 `-1.0` 死亡惩罚在当前 3 seed、100k 预算下没有收益：最终资格成功率均值从
`0.533` 降为 `0.317`，三个 seed 的差值均为负，死亡和采集未完成比例也增加。
本轮不采用该惩罚，后续继续以零惩罚 deterministic CNN-only 为基线。

## 实验协议

- 对照为 deterministic v3 的 CNN-only：两层空间 CNN、128 维 embedding，episode PPO。
- 实验 arm 只在**训练 episode 的死亡终止步**额外加入 `-1.0`，再计算 GAE/returns。
  系数固定为与 success bonus 相同的绝对值，本轮不搜索系数。
- 原生环境奖励、progress bonus `0.5`、success bonus `1.0` 与基线一致；成功、环境
  horizon、外部截断和所有评估 episode 均不加 death penalty。
- 训练 seed 为 `[0, 1, 2]`，每个 seed 精确使用 `100000` 次交互，在 `25000` 和
  `100000` 步保存策略、优化器和 RNG checkpoint。
- 训练 horizon 为 `256`，环境 length 为 `10000`；外部截断 bootstrap，死亡不 bootstrap。
- 稳定环境 wrapper 为 `balance-object-insertion-order-v1`，安装包未修改。
- `PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，PyTorch deterministic
  algorithms 与 cuDNN deterministic 开启，benchmark 和 TF32 关闭。
- 三个实验 seed 在独立进程中运行，使用同一 RTX 4090（GPU 1）；每进程 CPU 线程数为 1。
- 开发评估在每个 checkpoint 使用每 seed 20 个 episode；最终资格评估也为每 seed 20 个
  episode，沿用已有的诊断 seed 和动作 RNG manifest。

主要指标为最终资格成功率；同时报告训练和资格死亡、截断、wood1/2/3 到达情况。
训练统计先计算各 seed 的 episode 比例，再对 seed 求均值。死亡总数受 episode 长度
影响，不能直接作为改进证据。奖励尺度改变后 value loss 也不能直接跨 arm 评价优劣。

相同 seed manifest 不保证两个 arm 的训练轨迹或经历的世界数量一致；本轮约束的是
环境/动作 seed 生成规则、网络、训练器和实际交互预算。资格评估按相同 seed 和 episode
配对。资格 seed 虽与训练角色分离，但已被多轮诊断复用，不能视为新的正式保留测试集。

## 复现与奖励边界

实验前先用零惩罚重跑 seed 0 的 25k 训练。以下内容与历史基线完全一致：policy、
optimizer、Python/NumPy/Torch/CUDA RNG、episode 数、交互步数、训练记录中的行为与 PPO
统计、训练曲线和开发评估记录。life 字段曾做诊断修复，因此仅这些诊断字段未纳入
历史记录一致性比较。复现使用固定 entropy coefficient，25k 审计停止点不影响其数值。

审计文件：
`results/crafter_wood3_death_penalty_cuda_pilot_20261009_v1/zero_control/audit.json`。

测试覆盖死亡奖励仅加一次及其 GAE 传播、成功/截断/评估排除、非法系数拒绝、其他训练
配置变化拒绝，以及不同 episode 数下的 seed 比例统计。全量测试：`161 passed`。

## 运行与产物

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m \
experiments.run_crafter_wood3_death_penalty_diagnostic \
--config configs/crafter_wood3_death_penalty_cuda_pilot_v1.yaml \
--output results/crafter_wood3_death_penalty_cuda_pilot_20261009_v1
```

基线：
`results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1/`。
基线资格轨迹的完整 life 诊断：
`results/crafter_wood3_survival_checkpoint_diagnostic_20261008_v2/result.json`。
本轮配置、源码快照和哈希位于结果目录的 `config.json`、`source_snapshot/`、
`provenance.json`；每 seed 原始轨迹在 `cnn_only/seed_*/result.json`。

## 结果

汇总：`results/crafter_wood3_death_penalty_cuda_pilot_20261009_v1/summary.json`。
可复核的统计表与图：结果目录下的 `analysis/`。对照图中圆点为各 seed，菱形为均值。

| 设置 | seed 0 | seed 1 | seed 2 | 资格均值 | 达到候选 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN-only，零惩罚基线 | 0.50 | 0.45 | 0.65 | 0.533 | 0/3 |
| CNN-only，死亡惩罚 -1.0 | 0.45 | 0.10 | 0.40 | 0.317 | 0/3 |
| 惩罚减基线 | -0.05 | -0.35 | -0.25 | -0.217 | — |

每 seed 的 20 个资格 episode 使用相同环境/动作 seed manifest。合并 60 组配对结果中，
双方都成功 8 组、只有基线成功 24 组、只有惩罚成功 11 组、双方都失败 17 组。
这是诊断样本的描述统计，不作为正式显著性或总体成功率结论。

| 设置 | 训练区间 | 区间训练成功率 | 死亡比例 | 外部截断比例 | 开发成功率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 基线 | 0–25k | 0.300 | 0.636 | 0.065 | 0.400 |
| 惩罚 -1.0 | 0–25k | 0.308 | 0.640 | 0.051 | 0.417 |
| 基线 | 25k–100k | 0.408 | 0.538 | 0.054 | 0.400 |
| 惩罚 -1.0 | 25k–100k | 0.364 | 0.589 | 0.048 | 0.383 |

100k 实际训练 episode 数为基线 `[760, 705, 703]`，惩罚 `[689, 697, 734]`；两个 arm
的交互步数都精确为每 seed 100000。25k 开发均值的小幅变化没有在最终资格评估中保留。
后 75k 区间的训练成功率与死亡比例也没有改善；训练和资格评估的均值变化方向相同。
本轮没有出现“增加生存却减少采集”的可保留收益。仅 3 个 seed 仍不能排除随机变异，
不据此作总体性能或显著性结论。

后 75k 的 PPO 均值：基线 KL `0.00217`、clip fraction `0.0225`、explained variance
`0.581`；惩罚 KL `0.00208`、clip fraction `0.0215`、explained variance `0.524`。
逐 episode 的所有 PPO 统计均为有限值，没有观测到数值爆炸。value loss 从 `0.144`
变为 `0.191`，由于奖励目标已改变，不以这项数值的差异单独判断训练器质量。

三个惩罚 seed 均验证了真实 CUDA tensor；其确定性配置与历史基线相同。逐训练 episode
核对 death penalty 只在死亡时出现且数值为 `-1.0`，三个 seed 的奖励边界错配都为 0。
全部开发/资格 episode 的死亡惩罚数量也都为 0。本轮完成 325k 训练交互，其中 25k
用于零惩罚复现，300k 用于实验 arm；基线 300k 交互来自已有运行。

## 失败阶段与执行行为

| 设置 | 成功 | 死亡 | 外部截断 | wood1 | wood2 | wood3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线资格集 | 32/60 | 26/60 | 2/60 | 49/60 | 39/60 | 32/60 |
| 惩罚资格集 | 19/60 | 38/60 | 3/60 | 43/60 | 28/60 | 19/60 |

从 wood1 到 wood2 的条件完成比例为 `39/49=0.796` 对 `28/43=0.651`；从 wood2 到
wood3 为 `32/39=0.821` 对 `19/28=0.679`。死亡阶段从基线的 `[10, 10, 6]` 增加到
`[16, 13, 9]`，分别对应 wood1 前、wood1–wood2、wood2–wood3。下降贯穿首次寻找与
后续采集，固定的终止惩罚没有解决这些执行问题。

| 设置 | seed 0 noop 比例 | seed 1 noop 比例 | seed 2 noop 比例 |
| --- | ---: | ---: | ---: |
| 基线资格轨迹 | 0.1% | 1.9% | 1.9% |
| 惩罚资格轨迹 | 0.4% | 57.5% | 3.7% |

各列按该 seed 的实际资格交互步统计。seed 1 的惩罚策略出现明显停滞，其成功率为
0.10、死亡 18/20。合并交互步后，noop 从 1.3% 变为 22.6%，do 从 44.7% 变为 32.3%；
这一合并统计受 seed 1 的长轨迹影响，不能代替逐 seed 分析。也不能仅从动作分布认定
noop 是其他 seed 性能下降的原因。

惩罚资格死亡中，30/38 次在 food/drink/energy 全部仍大于 0 时发生；其余 8 次 drink
为 0。终止步伤害规则提示为敌人/投射物 22 次、熔岩 1 次、资源耗尽 3 次、混合或
未知 12 次。这些标签由 health 差值、资源和地形推断，不是直接伤害来源插桩，也不是
确证的因果归因。它们仍支持优先诊断危险导航与持续执行，而不能排除并发资源耗尽。

## 判断与下一项

当前结果排除了“在该 CNN-only 基线、100k 预算上增加 -1.0 死亡终止奖励就能解决
瓶颈”这一具体方案；不能推广为所有惩罚系数、更长预算或其他架构都无效。
基线和惩罚 arm 均未达到候选门槛，正式训练与技能复用链继续暂停。

下一项宜保持零惩罚 CNN-only 与现有 PPO 边界，先建立 RGB、动作与下一步采集/掉血事件
的关联诊断，再只加入一种面向局部执行的辅助目标做 25k/100k 对照。短距离交互与危险
反馈能否被学会，是下一轮要验证的假设；本轮数据还不能证明延迟信用分配就是唯一
原因。辅助标签若取自环境诊断，应只用于训练监督，评估 Policy 输入仍保持 RGB。
不同时修改架构、奖励、动作空间或训练器；本轮没有启动这项实验。

资格 0.8 仍是候选门槛，达到后仍须完整 Module 契约检查。stone、coal 和 pickaxe 需
分别验证，不能继承本轮结论。代码和文档当前为工作区修改，产物保存了对应源码快照。

后续已完成局部事件诊断和固定系数 0.1 的采集辅助损失 pilot，见
[局部事件与采集辅助损失记录](crafter-wood3-local-event-auxiliary-diagnostic-2026-10-09.md)。

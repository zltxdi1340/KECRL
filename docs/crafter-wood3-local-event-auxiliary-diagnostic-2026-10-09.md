# Crafter wood>=3 局部事件与采集辅助损失诊断（2026-10-09）

本轮已完成局部事件诊断和固定系数的采集辅助损失 pilot。100k 最终资格成功率均值
从基线的 `0.533` 降为 `0.333`，三个 seed 各下降 `0.20`，不采用本轮系数 0.1 的方案。
所有产物均为 `formal_result=false`，不注册 Module，
不更新 Knowledge Bank 或 SPT，不启动正式训练。结论只适用于 `wood >= 3`。

## 局部事件诊断

使用 deterministic v3 的 CNN-only 25k/100k checkpoint，每个 checkpoint、每个
policy seed 运行 20 个新 episode，共 120 个 episode、16205 次诊断交互。
环境 seed 从 7000000 开始、动作 seed 从 9000000 开始，按 seed index 加 1000000；
这些范围与已有训练、开发、资格角色不重叠。没有更新基线 Policy。

每步记录动作前的 RGB embedding、动作和动作后的 wood 增量、health 差值。初始
wood 未知的第一步不标作负例。掉血标签表示 health 差值为负，不表示已确证的伤害来源。
每个 seed 单独按 episode 划分前 15 个训练、后 5 个测试，避免同一 episode 的邻近帧
跨 split。离线线性 probe 固定初始化 seed 0，Adam lr=0.03、160 次更新，标准化只拟合
训练 split，BCE 正例权重为 sqrt(负例数/正例数)，限定在 [1, 10]。

| checkpoint | 已知 wood 标签 | wood 增量 | `do` 样本 | `do` 后 wood 增量比例 | 掉血事件 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25k | 8894 | 97 | 3479 | 2.79% | 184/8954 |
| 100k | 7191 | 105 | 3242 | 3.24% | 158/7251 |

所有已观察到的 wood 增量均来自 `do`。若混合所有动作，仅靠动作类别就能获得较高
AUC：它区分了可能采集的 `do` 与其他动作，不能证明 RGB 表示识别了采集机会。
因此主要诊断改为只保留 `do` 样本；这时 action-only 的 AUC 恒为 0.5。

| checkpoint | policy seed | 测试 `do` 样本 | 测试正例 | RGB embedding probe AUC | balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25k | 0 | 221 | 11 | 0.579 | 0.510 |
| 25k | 1 | 190 | 6 | 0.421 | 0.500 |
| 25k | 2 | 588 | 9 | 0.507 | 0.499 |
| 100k | 0 | 333 | 9 | 0.546 | 0.551 |
| 100k | 1 | 168 | 6 | 0.755 | 0.639 |
| 100k | 2 | 216 | 8 | 0.692 | 0.618 |

100k 的部分 seed 有采集可预测性，但 seed 0 接近随机排序，测试正例只有 6--9 个。
这些结果只提供测试一个局部采集辅助目标的依据，不能证明表示不足或信用分配是唯一
原因，也不能从较高总体 accuracy 推断模型已学会采集。health probe 同样随 seed
变化，本轮不再加入第二个辅助目标。加权 BCE 的输出也不视为校准后的事件概率。

原始采样：`results/crafter_wood3_local_event_diagnostic_cuda_20261009_v1/`。
条件 probe 的固定初始化重分析：
`results/crafter_wood3_local_event_conditional_probe_20261009_v1/summary.json`。
后者保存 dataset SHA256 和 probe 源码 SHA256，沿用同一份 NPZ，不增加环境交互。
初次采样产物中的旧 probe 不包含条件于 `do` 的比较，且未固定 probe 初始化；
本记录采用重分析结果。

## 辅助损失协议

保持零死亡惩罚的 deterministic v3 CNN-only：相同 CNN、actor/critic、episode PPO、
奖励、动作 allowlist、seed manifest、256 步 horizon 和精确 25k/100k 交互预算。
原生奖励、训练 progress bonus 0.5、success bonus 1.0 均沿用基线。

已核对本机 Crafter 的 `Player.update/_do_material/texture` 和 `data.yaml`：采集目标
为当前位置加 facing；朝向相邻树、无对象阻挡且未睡眠时，一次 `do` 收到 1 wood，
没有连续敲击多次才收到 wood 的要求。facing 对应不同玩家纹理。因此该事件目标对应
局部视觉机会与动作执行，低 wood 增量频率不能解释为所有负例都是采集过程的中间步。

只增加一个训练目标 `do_wood_gain`：由动作前 RGB embedding 预测本次 `do` 是否
使 wood 增加，使用独立线性 head。损失为 `PPO loss + 0.1 * weighted BCE`，正例权重
沿用上述平方根规则和 cap=10，按当前 episode 的已知 `do` 标签计算，不搜索系数。
辅助 head 的初始化隔离 RNG，保持原 actor/critic 初始化和采样 RNG 不变。

标签来自训练时动作前后已知的库存差值，仅用于监督，不作为 actor、critic 或辅助
head 的输入。非 `do`、未知库存及缺失 inventory 的样本均屏蔽。动作后的信息与
动作前 RGB 对齐；不使用下一帧作为预测输入。离线 probe 数据不用于训练 Policy。
资格与开发评估只执行 RGB Policy，不优化、不应用辅助损失。

三个 seed `[0, 1, 2]` 各训练 100000 次交互，保存两个 checkpoint，开发与最终资格
评估每 seed 各 20 个 episode。资格 seed 与训练分离，但被多轮诊断复用，不能当作
新的正式保留测试集。跨 arm 固定的是 seed 生成规则和实际交互预算；策略不同仍会
导致不同轨迹、episode 数和遇到的世界数量。

主要指标为 100k 最终资格成功率，辅助报告 wood1/2/3、死亡阶段、动作分布和 PPO
统计。训练比例先按 seed 计算再求均值，避免 episode 长度导致计数误读。
辅助预测损失下降本身不作为任务性能改善的证据。

## 审计与验证

运行顺序为零辅助损失 seed 0 的 25k 基线复现，通过后启动三个 100k seed 和一个
独立进程中的辅助 seed 0 25k 重复。零控制核对历史 policy、optimizer、全部 RNG、
行为/PPO 训练记录、训练曲线和开发结果。历史 life 字段曾修复，仅这些诊断字段
从历史记录一致性比较中排除。辅助重复比较完整新记录和 checkpoint 状态。

零控制 25k 审计已通过：policy、optimizer、Python/NumPy/Torch/CUDA RNG、episode
数与交互步数全部相同，训练行为/PPO 记录、训练曲线和开发评估记录也相同。
辅助 seed 0 的跨进程 25k 重复也通过：全部 checkpoint 比较字段和完整训练/开发
记录一致。三个 100k checkpoint 的 60 个资格 episode 另行重放，成功、步数、终止
原因全部一致。重放正确重建包含辅助 head 的策略；评估仍只有 RGB 输入。

两种 arm 均为每 seed 精确 100000 步，全部训练 PPO/辅助损失统计为有限值；六个
辅助 checkpoint 的 policy 与 optimizer tensor 也全部为有限值。训练源码与快照
哈希相同，历史基线配置、结果和 checkpoint 哈希未变化。
全部开发/资格 episode 的辅助样本数为 0、PPO update 为 None，训练和评估均没有
增加死亡奖励。

测试已覆盖辅助 head RNG 隔离、零系数更新等价、共享 RGB encoder 梯度、未知/
非 `do` 标签屏蔽、动作前 RGB 与标签对齐、评估无更新、非法系数拒绝及基线配置匹配。
全量回归：`175 passed`；读取优化与旧 runner 兼容调整后相关测试：`25 passed`。

## 运行与产物

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m \
experiments.run_crafter_wood3_do_wood_gain_auxiliary_diagnostic \
--config configs/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_v1.yaml \
--output results/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_20261009_v1
```

输出目录保存配置、源码快照、基线 checkpoint/结果哈希、逐 seed 轨迹与策略/
优化器/RNG checkpoint。汇总写入 `summary.json`；零控制与辅助重复分别写入
`zero_control/audit.json`、`auxiliary_repeat_25k/audit.json`。

重放结果：`results/crafter_wood3_do_wood_gain_auxiliary_checkpoint_replay_20261009_v1/result.json`。
结果目录 `analysis/` 包含资格、训练曲线、辅助监督和条件 probe CSV，以及
`verification.json`、`comparison.png/svg`、`conditional_probe.png/svg`。
图显示各 seed 和均值，不添加基于 3 seed 的总体置信区间。

```bash
/opt/anaconda3/bin/python -m experiments.analyze_crafter_wood3_local_event_auxiliary \
--input results/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_20261009_v1 \
--probe-input results/crafter_wood3_local_event_conditional_probe_20261009_v1 \
--output results/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_20261009_v1/analysis
```

图表导出使用有 matplotlib 的分析 Python；不改变训练环境依赖。结果目录存在时
各 runner 和导出器均拒绝覆盖，应为重复实验使用新目录。

## 100k 结果

| 设置 | seed 0 | seed 1 | seed 2 | 资格均值 | 达到候选 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN-only 基线 | 0.50 | 0.45 | 0.65 | 0.533 | 0/3 |
| `do_wood_gain` 辅助损失，系数 0.1 | 0.30 | 0.25 | 0.45 | 0.333 | 0/3 |
| 辅助减基线 | -0.20 | -0.20 | -0.20 | -0.200 | - |

60 组配对资格结果中，双方都成功 11 组、只有基线成功 21 组、只有辅助成功 9 组、
双方都失败 19 组。沿用多轮诊断 seed 的描述统计不作为正式显著性或总体成功率结论。

| 设置 | 训练区间 | 区间训练成功率 | 死亡比例 | 外部截断比例 | 开发成功率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 基线 | 0--25k | 0.300 | 0.636 | 0.065 | 0.400 |
| 辅助 | 0--25k | 0.283 | 0.658 | 0.059 | 0.283 |
| 基线 | 25k--100k | 0.408 | 0.538 | 0.054 | 0.400 |
| 辅助 | 25k--100k | 0.341 | 0.601 | 0.058 | 0.367 |

最终训练 episode 数为基线 `[760, 705, 703]`、辅助 `[702, 680, 726]`。
训练后 75k 的成功率与资格成功率变化方向相同。辅助的开发结果从 25k 到 100k
提高，但最终仍低于基线，不能用开发曲线提高代替资格收益。

后 75k 的 PPO 均值：基线 KL `0.00217`、clip fraction `0.0225`、value loss `0.144`、
explained variance `0.581`；辅助分别为 `0.00189`、`0.0160`、`0.147`、`0.595`。
EV 略高同时任务完成率更低，critic 拟合指标不能替代执行能力判断。

辅助训练共获得 123775 个已知 `do` 样本，其中 3146 个 wood 增量，正例比例约 2.54%。
按 seed 分别为 `992/42544`、`991/39170`、`1163/42061`。各 seed 的已标注 episode
平均辅助损失从早期到后期分别为 `0.439 -> 0.407`、`0.383 -> 0.407`、
`0.420 -> 0.439`，没有一致下降趋势。episode 的正例权重和轨迹分布会变化，不能
从这项损失单独判断辅助头的泛化效果。本轮没有对训练后的辅助头做新保留集预测评估。

本轮训练交互共 350k：零控制 25k、辅助重复 25k、三个实验 seed 各 100k；基线
300k 来自已有运行。局部诊断的 16205 次交互与开发/资格评估另计，不混入训练预算。

## 失败阶段与下一项

| 设置 | 成功 | 死亡 | 外部截断 | wood1 | wood2 | wood3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线资格集 | 32/60 | 26/60 | 2/60 | 49/60 | 39/60 | 32/60 |
| 辅助资格集 | 20/60 | 37/60 | 3/60 | 49/60 | 33/60 | 20/60 |

到达 wood1 的总数相同，但 wood1 到 wood2 的条件完成率从 `39/49=0.796` 变为
`33/49=0.673`，wood2 到 wood3 从 `32/39=0.821` 变为 `20/33=0.606`。
三个死亡阶段从 `[10, 10, 6]` 变为 `[11, 15, 11]`，分别对应 wood1 前、wood1--wood2、
wood2--wood3。新增死亡主要出现在首次采集之后，支持下一步重点诊断持续寻找、转向与
危险条件下的重复执行；这些终止计数不能单独证明具体导航行为是因果来源。

| 设置 | seed 0 noop 比例 | seed 1 noop 比例 | seed 2 noop 比例 |
| --- | ---: | ---: | ---: |
| 基线 | 0.1% | 1.9% | 1.9% |
| 辅助 | 4.0% | 2.1% | 0.6% |

各列按对应 seed 的实际资格步数计算，未出现先前死亡惩罚 pilot 中的大幅 noop 停滞。
`do` 比例分别从 `[64.9%, 32.8%, 33.3%]` 变为 `[32.0%, 38.5%, 46.5%]`：动作频率
变化方向不一致，增加采集动作比例也未保证重复采集完成。

辅助资格死亡中 33/37 次在 food/drink/energy 仍全部大于 0 时发生，其余 4 次 drink
耗尽。终止步伤害提示为敌人/投射物 18 次、资源耗尽 2 次、混合或未知 17 次。
这些提示仍是启发式标签，不是直接伤害来源插桩，不能据此排除并发原因。

保留零辅助损失、零死亡惩罚的 CNN-only 为基线。本轮只否定这一个系数、目标、
训练预算组合的采用价值，不能推广到所有辅助目标或系数。正式训练与技能复用链仍暂停。

下一项先做现有 checkpoint 的**采集机会与执行拆分诊断**：在动作前记录邻接树、
朝向、目标格对象阻挡、睡眠状态及附近敌人，核对“有可采机会时是否执行 do”与
“没有机会时如何移动/转向”，并按 wood0/1/2 阶段汇总。在同一稳定 wrapper 和新的
诊断 seed 上采样，不更新 Policy；位置/世界状态仅作诊断标签，Policy 输入仍为 RGB。
这能区分机会未到达、机会未识别和机会存在时执行失败，再决定下一种训练目标。
本轮没有启动该后续诊断。

候选门槛仍为 0.8，达到也不替代完整 Module 契约检查。stone、coal 和 pickaxe 没有
在本轮重新验证。代码和文档为工作区修改，训练产物已归档实际源码快照。

后续采集机会与执行拆分诊断已完成，详见
[采集机会诊断记录](crafter-wood3-collection-opportunity-diagnostic-2026-10-09.md)。

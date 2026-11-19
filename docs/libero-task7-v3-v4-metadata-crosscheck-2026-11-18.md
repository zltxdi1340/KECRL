# task7共享v3 / 视角专用v4：只读元数据交叉核查（2026-11-18）

**控制记录零漂移；两候选交集9、仅v3成功2、仅v4成功1、并集12、双双失败10。** v4在seed07保留v3全部3条成功，再恢复flip10-unchanged；seed06丢失2条v3成功。9个共同成功来源中v4全部慢1–8个净控制步。并集12是事后结果集合，不是已执行组合策略成绩，不能用来通过继续门或解锁120条。

本次仅读取docs/evidence与results下16份已审计JSON/CSV，检查记录的SHA、完成/审计/结果链接及JSON与CSV一致性；没有读取configs绑定正文、HDF5、checkpoint、NPZ/RGB或私有坐标，无新数组解码、模型构造/前向、训练、环境或物理步。未更改既有归档、协议、绑定或checkpoint。本报告与新JSON/CSV是本次派生结果，不是新闭环样本。

## 1. 控制一致性与来源

v3/v4的flat_zero_xyz均8/22：全部22条成功布尔值和terminal_step完全相等，源source_id/seed/initial_state_index/condition/anchor及原前缀描述符完全相同。三个seed训练权重张量hash、sampling hash和500个loss值一致。元数据记录的控制NPZ SHA及全公开观察摘要序列也22/22一致。

这些是已审计记录层面的严格一致性；本轮未重新计算原NPZ/checkpoint SHA、未解码逐像素或物理状态，不能冒充新一次位级物理复演。双双成功仅表示这两份归档都成功，不是多次运行稳定性估计。

## 2. 四格集合与基线覆盖

| 类别 | 数量 | source_id |
|---|---:|---|
| 双双成功 | 9 | `flip00-unchanged`、`flip00-confirm5`、`flip03-unchanged`、`flip03-confirm5`、`flip05-unchanged`、`flip06-unchanged`、`flip09-unchanged`、`flip11-unchanged`、`flip11-confirm5` |
| 仅v3成功 | 2 | `flip04-unchanged`、`flip05-confirm5` |
| 仅v4成功 | 1 | `flip10-unchanged` |
| 双双失败 | 10 | `flip01-unchanged`、`flip01-confirm5`、`flip04-confirm5`、`flip06-confirm5`、`flip08-unchanged`、`flip08-confirm5`、`flip09-confirm5`、`flip10-confirm5`、`flip12-unchanged`、`flip12-confirm5` |

并集12/22。交集9中只含原8条基线成功的5条，缺`flip08-unchanged`、`flip08-confirm5`、`flip10-unchanged`；并集中含基线6条，仍缺两个flip08。因此“核心交集包含全部基线成功”不成立。双双失败10条中的两个flip08可由flat成功，不能称任务本身不可解；这里只表示当前两种空间读出在这些归档条件下均未成功。

`flip05-confirm5`确实在v3独有项中，v3于173步成功（净73步），v4于300步失败；另一项`flip04-unchanged`为v3于212步成功（净137步），v4失败。唯一v4独有为`flip10-unchanged`，295步成功、交接100、净195步。

## 3. 逐seed翻转及seed07解释

| seed | 状态数 | 控制 | v3 | v4 | v3→v4新增 | v3→v4丢失 |
|---|---:|---:|---:|---:|---|---|
| 20261106 | 12 | 4 | 8 | 6 | — | flip04-unchanged, flip05-confirm5 |
| 20261107 | 8 | 4 | 3 | 4 | flip10-unchanged | — |
| 20261108 | 2 | 0 | 0 | 0 | — | — |

seed06的翻转：`flip04-unchanged`（anchor75）和`flip05-confirm5`（anchor100）从成功变失败，没有新增。seed07唯一翻转：`flip10-unchanged`（anchor100）从失败变成功；原v3成功`flip09-unchanged`、`flip11-unchanged`、`flip11-confirm5`全部保留，没有用flip10替换其中另一条。

相对flat控制，seed07的v4仍丢失两条flip08；保留控制成功中的flip09-unchanged、flip10-unchanged两条，并新增控制失败的两条flip11，故总数4/8不退化。不能把“成功数不退化”解释成“保住原来全部4条控制成功”。

## 4. 共同成功子集的净控制步数

定义`净步数=terminal_step−anchor_step`，不计reset、稳定化与旧前缀；只比较9个共同成功来源。失败的300步截断不计为成功耗时。

| source_id | anchor | v3净步 | v4净步 | v4−v3 |
|---|---:|---:|---:|---:|
| flip00-unchanged | 75 | 41 | 45 | +4 |
| flip00-confirm5 | 75 | 59 | 65 | +6 |
| flip03-unchanged | 75 | 67 | 75 | +8 |
| flip03-confirm5 | 75 | 57 | 65 | +8 |
| flip05-unchanged | 100 | 45 | 47 | +2 |
| flip06-unchanged | 75 | 51 | 55 | +4 |
| flip09-unchanged | 125 | 8 | 9 | +1 |
| flip11-unchanged | 200 | 22 | 23 | +1 |
| flip11-confirm5 | 200 | 25 | 26 | +1 |

v3均值41.667步、中位数45；v4均值45.556步、中位数47，平均多3.889步。v4快/平/慢=0/0/9；这些共同成功终点最晚226步，不属于295步的压线新增。flip10的净195步单独报告，不混入9个共同成功均值。该条件子集统计不代表全22状态的无条件收敛速度或重复运行分布。

## 5. 同口径熵与成功计数：描述性比较

v3共享的是参数，日志仍有agentview与eye_in_hand各自熵。采用已审计训练trace，同seed两轮500批sampling hash和有效行数序列相等；按有效帧/通道加权，`H_view=Σ逐步entropy_nats_sum / Σheatmaps_per_view`，每步分母为valid_rows×256。整体为两视图等权，归一化再除ln16。下表用固定末50更新（451..500）；全500步同口径值保留在analysis.json，不按成败挑窗口。

本口径与此前v4 training-summary中“50个逐步均值的简单平均”略有舍入/权重差异，此处统一重新加权，不改旧summary或结果。闭环两策略访问状态和后缀长度不同，本次不将闭环平均熵混作同输入的架构对照。

| seed（成功v3→v4） | v3 agent H | v4 agent H | Δagent | v3 wrist H | v4 wrist H | Δwrist |
|---|---:|---:|---:|---:|---:|---:|
| 20261106（8→6） | 2.561878 | 2.551827 | -0.010051 | 2.533501 | 2.485292 | -0.048209 |
| 20261107（3→4） | 2.555877 | 2.537983 | -0.017895 | 2.526201 | 2.499259 | -0.026942 |
| 20261108（0→0） | 2.566445 | 2.544671 | -0.021774 | 2.526853 | 2.487497 | -0.039355 |

单位均为nats。v4两视图熵在三个seed上均较低，腕部下降幅度更大；但成功计数分别8→6、3→4、0→0，并不随熵降低一致改善。不能由此归因于注意力对准、腕部专门化、几何分辨率提升或去共享机制；宽度和初始化也同时变化。本次不重演任何潜在热图或推断它们对应碗/盘。

## 结论与交付

覆盖并非完全相同，但互补仅为v3独有2条/v4独有1条；v4也没有在共同成功子集上更快。并集12即使在已知数据上达到某个计数门，也是假设有事后择优者的上限，不能按source_id、seed或最终成败事后切换策略并报成已执行成绩；更不据此自动训练组合或启动120条。v3/v4原继续门失败的封存结论保持。

完整22行及检查标记：`results/libero-task7-v3-v4-metadata-crosscheck-20261118-v1/paired-states.csv`；计算定义、9/2/1/10列表、时序与全500/末50熵、16份来源元数据SHA和零操作账本见同目录`analysis.json`。仅做标准库元数据运算与一致性断言，未改程序、未跑pytest或新环境检查。

本轮保持500更新/稳定化5/现loss及封存角色不变，未修改既有协议、绑定或原始结果；没有新的训练、模型前向、物理步或数值归档解码。

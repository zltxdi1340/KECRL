# Crafter wood>=3 采集机会与执行拆分诊断（2026-10-09）

本轮只读取冻结 checkpoint 的行为，训练交互步数为 0。比较 deterministic v3
CNN-only 基线与上一轮 `do_wood_gain` 辅助损失系数 0.1 的策略，使用新诊断 seed。
所有产物均为 `formal_result=false`，不注册 Module，不更新 Knowledge Bank/SPT，
不启动正式训练。结论仅适用于 `wood >= 3`。

主诊断与审计均已完成。100k 基线在邻接未对准时只以 9.4% 的比例选择朝树转向，
可采时以 42.5% 的比例执行 `do`；附近没有 radius-3 敌人时这些缺口仍存在。
当前问题更明确地落在把空间机会转换为接近、转向和采集动作的执行能力，尚不能
由本轮进一步断定是 encoder、actor 学习还是信用分配中的某一个单独因素。

## 固定协议

- 两个 arm、三个 policy seed `[0, 1, 2]`、25k/100k 两个 checkpoint；每组 20 个 episode，
  共 240 个主诊断 episode。每 episode 最多 256 步，环境 length=10000。
- 环境 seed 从 13000000、动作 RNG seed 从 17000000 开始，按 seed index 加 1000000。
  配置校验排除与已有训练、开发、资格及上轮局部事件诊断的同类 seed 范围重叠。
- 相同环境/动作 manifest 在 arm 和 checkpoint 间配对；它不保证不同策略的完整轨迹一致。
- 稳定 wrapper 为 `balance-object-insertion-order-v1`，使用 RTX 4090 GPU 1，严格
  Torch/cuDNN 确定性配置、`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- actor/critic 仍只接收 64x64 RGB。先由策略采样动作，再读取动作前世界状态作诊断标签。
  不改变 adapter 的公共 observation/state，不用这些标签选择或修改动作。
- checkpoint 的参数和 optimizer 状态加载后保持不变；执行 `eval()` 与 `no_grad()`。

## 标签与解释边界

记录动作前位置、facing、wood、资源、睡眠、四邻格地形及对象阻挡、视窗内树的位置、
附近 Zombie/Skeleton/Arrow，动作概率和实际动作，随后记录 wood/health 变化及位置/朝向。
树的搜索只限本机实际 RGB 地形视窗 9x7；“视窗内”是地理范围标签，不能保证夜间像素
容易辨识。hostile radius 固定为 Manhattan 距离 3，只是邻近标签，不是直接伤害归因。

三层机会定义分别为：

1. 邻接且没有对象挡住的树，表明已到达局部树机会。
2. 存在这样的邻接树且睡眠不覆盖动作，表明可以通过移动动作调整朝向。
3. 已朝向这样的树、未受睡眠覆盖且库存未满，表明执行 `do` 能获得 wood。

Crafter 在 `_move` 中先改变 facing，再尝试移动，所以朝树移动虽然被树挡住，仍是
有效转向。睡眠且 energy<9 时动作被覆盖；energy=9 时会先醒来再执行选择的动作。
本机树采集无工具要求、一次获得 1 wood、概率为 1；Player 在其他对象之前更新。
每个记录步都检查 `(action == do AND ready_to_collect) == (wood 增量 > 0)`，且增量
只能为 0 或 1。错配直接停止，不把错误标签纳入后续判断。

转向率以“存在未对准的、可转向邻接树”的步数为分母。接近动作率以视窗内有未阻挡
树、存在沿 Manhattan 距离接近的合法移动、且未被睡眠覆盖的步数为分母；排除 lava，
不使用全局路径规划，也不把距离下降称作确定的安全导航。

ready visit 指同一 wood 阶段内连续处于第三层机会的窗口；采到 wood、机会消失或
episode 终止后关闭。机会期间非 `do` 不直接视为策略错误：敌害回避和资源动作可能
有合理目的，需要结合邻近敌人和后续结果解释。

各 episode 失败按最终 wood 阶段归类，并区分：没有邻接机会、只有睡眠中的机会、
清醒邻接但从未对准、曾 ready 但没有执行 do，以及进入阶段后尚无下一决策步就终止。
最后一类保留“刚采到 wood 同一步死亡”的情况，避免把零决策暴露误判为搜索失败。

## 审计与测试

每个 policy/checkpoint 的首个 episode 在原 `_rollout` 路径与新记录路径各执行一次，
比较成功、步数、原生奖励、终止原因、wood 首次步数、动作计数、动作熵和 Torch/CUDA RNG。
另以独立 Python 进程重复基线 seed 0 的 100k checkpoint 全部 20 个诊断 episode，
比较完整逐步记录的 canonical digest 与汇总。记录前后核对 policy/optimizer 未改变，
并保存和核对 checkpoint、源码与本机 Crafter 文件哈希。

全量测试 `184 passed`。新增测试覆盖真实 Crafter 的朝树转向、满能量醒来、对象阻挡、
视窗范围、附近敌人半径、熔岩排除、诊断 opt-in、公共状态隔离、读取不改变 world RNG/
RGB/固定动作轨迹、阶段末零决策暴露，以及新 seed 与旧角色的冲突拒绝。

12 组 observer 对照全部通过，比较字段及 Torch/CUDA RNG 全部相同。独立进程的
20 个完整 episode 逐步 canonical digest 和汇总也相同。全部冻结 policy/optimizer
状态未改变，真实 CUDA tensor 已验证。主诊断的 33191 步、353 次 wood 增量均满足
采集规则，错配为 0；可采条件下选中 do 的每一次都获得了 1 wood。

独立重复完成 2419 步，13 个 observer control 完成 1375 步，本轮总诊断交互 36985 步，
训练交互为 0。主诊断 240 个 episode、独立重复 20 个、observer control 13 个，共 273 个。
重复完成后再次核对全部 25 个源码/已安装 Crafter/checkpoint/配置文件哈希，全部相同。

## 运行与产物

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m \
experiments.run_crafter_wood3_collection_opportunity \
--config configs/crafter_wood3_collection_opportunity_cuda_v1.yaml \
--output results/crafter_wood3_collection_opportunity_cuda_20261009_v1
```

配置、源码快照和 provenance 位于输出根目录；逐步轨迹在
`{arm}/seed_{seed}/checkpoint_{steps}/episodes.jsonl`，每组及全局汇总为 `summary.json`。
独立进程重复保存在 `cross_process_repeat/`。所有 runner 均拒绝覆盖现有输出目录。

统计表、图和逐步 digest 验证在 `analysis/`：`outcomes.csv`、`stage_funnel.csv`、
`failure_classes.csv`、`decisions.csv`、`decision_conditions.csv`、`verification.json`、
`final_file_verification.json` 及 `opportunity_comparison.png/svg`。每组和 pooled 行均保留；
pooled 决策率按实际暴露步数加权，不能视为 seed 均值，更不能把逐步样本当作独立样本。

```bash
/opt/anaconda3/bin/python -m experiments.analyze_crafter_wood3_collection_opportunity \
--input results/crafter_wood3_collection_opportunity_cuda_20261009_v1 \
--output results/crafter_wood3_collection_opportunity_cuda_20261009_v1/analysis
```

## 新诊断集合结果

| 策略 | checkpoint | seed 0 | seed 1 | seed 2 | 成功率均值 | 死亡 | 外部截断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 25k | 0.50 | 0.60 | 0.10 | 0.400 | 34/60 | 2/60 |
| 基线 | 100k | 0.55 | 0.45 | 0.20 | 0.400 | 33/60 | 3/60 |
| 辅助 | 25k | 0.35 | 0.35 | 0.30 | 0.333 | 37/60 | 3/60 |
| 辅助 | 100k | 0.35 | 0.10 | 0.30 | 0.250 | 43/60 | 2/60 |

这是新诊断集合，不能与上一轮资格成功率均值 0.533/0.333 混为同一个指标。基线
seed 2 在新集合仅 0.20，也显示表现依赖环境/动作 seed。每 seed 20 个 episode
不足以单独估计总体泛化率；所有策略仍未达到候选 0.8，本轮没有执行 Module 资格检查。

| 策略 | checkpoint | 邻接未对准时朝树转向率 | ready 时 do 率 | 可局部接近时的接近动作率 |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 25k | 61/396 = 15.4% | 93/236 = 39.4% | 948/4267 = 22.2% |
| 基线 | 100k | 44/468 = 9.4% | 102/240 = 42.5% | 778/3922 = 19.8% |
| 辅助 | 25k | 47/551 = 8.5% | 84/242 = 34.7% | 959/4264 = 22.5% |
| 辅助 | 100k | 34/528 = 6.4% | 74/177 = 41.8% | 850/4100 = 20.7% |

100k 基线按 seed 的转向率为 `[7.1%, 15.5%, 7.7%]`，ready do 率为
`[57.6%, 36.5%, 37.1%]`。转向缺口并非只由单个 seed 导致。25k 到 100k 的
ready do pooled 率略升，却没有形成更稳定的接近或转向行为。

| 100k 策略与空间条件 | ready do | 邻接未对准时转向 | 可局部接近时接近 |
| --- | ---: | ---: | ---: |
| 基线，radius 3 内无敌人 | 82/201 = 40.8% | 29/388 = 7.5% | 634/3262 = 19.4% |
| 基线，radius 3 内有敌人 | 20/39 = 51.3% | 15/80 = 18.8% | 144/660 = 21.8% |
| 辅助，radius 3 内无敌人 | 68/155 = 43.9% | 30/442 = 6.8% | 741/3536 = 21.0% |
| 辅助，radius 3 内有敌人 | 6/22 = 27.3% | 4/86 = 4.7% | 109/564 = 19.3% |

无近邻敌人也有低转向、低接近与机会流失，不能将所有非采集动作解释为躲避。
但这些轨迹没有配对到完全相同的状态，敌人距离也不是完整危险信息；不从这张表
推断敌人出现对动作的因果效应。

100k 基线 ready 的平均 `p(do)` 为 0.402，邻接但未对准时为 0.487。按 seed 分别是
`0.611 vs 0.636`、`0.324 vs 0.322`、`0.322 vs 0.454`。没有稳定的“真正可采时提高
do 概率”的选择性。辅助策略 pooled 为 `0.403 vs 0.446`，也未形成清晰改善。
这一观察涉及当前策略输出，不能证明 RGB 中缺少信息或 encoder 无法表达机会。

100k 基线的 3493 次 do 中，102 次增加 wood；其他上下文为邻接树但朝向不对 214 次、
没有无阻挡邻接树 2826 次、睡眠覆盖 249 次、目标为对象 102 次。目标对象可能对应
战斗或食物动作，不把所有未增加 wood 的 do 视为无意义行为。

## 最终失败阶段

| 100k 策略 | wood1 | wood2 | wood3 | wood1→wood2 | wood2→wood3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基线 | 44/60 | 34/60 | 24/60 | 34/44 = 77.3% | 24/34 = 70.6% |
| 辅助 | 37/60 | 22/60 | 15/60 | 22/37 = 59.5% | 15/22 = 68.2% |

| 最终失败阶段的机会历史 | 基线失败 36 次 | 辅助失败 45 次 |
| --- | ---: | ---: |
| 没有到达无阻挡邻接树 | 22 | 24 |
| 清醒时曾邻接，但从未 ready | 5 | 8 |
| 曾 ready，但该阶段未执行 do | 8 | 13 |
| 刚进入该阶段同一步终止，无下一决策步 | 1 | 0 |

“没有到达”只指最终失败阶段，episode 在前一个阶段可能成功采过 wood；这一分类
不是说整个 episode 从未遇到树。没有出现只在睡眠中邻接的最终失败类别。
基线三个阶段的失败数为 `[16, 10, 10]`，辅助为 `[23, 15, 7]`。

100k 基线共有 150 个 ready visit，其中 102 个采到 wood，48 个离开或终止未采，
每 visit 平均 1.60 个 ready 步；辅助为 126 个 visit、74 个采到 wood、52 个未采，
平均 1.40 个 ready 步。短机会窗口下，接近后转向与紧接着采集都需要稳定执行。
睡眠覆盖分别占 878/7772=11.3%、1266/8952=14.1% 的诊断步；不能把睡眠中的
采样动作与实际执行等同，但睡眠不足以解释清醒时的低转向率。

## 判断与下一项

最直接的执行规则正常：可采且选 do 时，环境每次都给 wood。尚未稳定学会的是到达
局部机会、调整朝向和机会存在时选择采集。危险与死亡继续限制过程，但无近邻敌人
的子集也有明显动作缺口。本轮不能进一步区分 encoder/actor/信用分配的单独贡献。
scalar wood-gain 辅助目标未改善这些输出，继续保留零辅助损失、零死亡惩罚 CNN-only。

下一项应先做**短距离转向与采集的受控能力测试**：固定现有 Policy，把“树已邻接且
已朝向”与“树已邻接但需转向”分开，保持 RGB 输入，在安静、短窗口的构造场景中
测量 action probability 与 turn→do 完成率；加入同位置无树对照，区分空间条件反应
与固定动作偏好。这类场景只是能力诊断，不能用于宣称 Crafter 任务成功或 Module
资格。先检验局部动作选择的能力，再决定是否采用方向监督、局部课程或其他训练改动。
本轮没有启动该受控测试，也没有增加新的训练 arm。

后续已完成这项受控能力诊断，记录见
[局部转向与采集能力诊断](crafter-wood3-controlled-collection-diagnostic-2026-10-09.md)。

代码和文档目前为工作区修改，未提交。运行产物已保存实际源码与已安装 Crafter
规则快照；安装包没有修改。

候选资格门槛仍为 0.8。本轮新诊断集合不替代正式保留测试集与完整 Module 契约检查。
stone、coal、pickaxe 未在本轮重新验证。

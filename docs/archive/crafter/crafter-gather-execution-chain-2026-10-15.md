# Crafter gather 动作链审计与学习率修正

日期：2026-10-16。状态：非正式诊断已完成，正式 E1/E2 未启动。

## 本轮问题与范围

本轮检查受控采集 SPI 的低成功率是否来自原生规则不可执行、32 步上限过短，或策略没有随动作阶段改变选择。检查服务于后续 SPT 更新的迁移比较，不要求先把自然 wood 训练到 0.8。

沿用 `quiet-material-gather-v1`：三种材料距离中心 1--3，统一木镐、充足初始生存资源、无敌人/水/熔岩，原生 64×64 RGB、移动、转向和采集规则。此范围不验证自然 Crafter 生存、工具制作或 Module 资格。

代码为 [`run_crafter_gather_execution_chain_diagnostic.py`](../../../experiments/run_crafter_gather_execution_chain_diagnostic.py)。先在 `0746753` 的 v1 run8、`908eb4c` 的 v2 run1、再在 `e43d678` 的 v3 run1 和 `155b204` 的 v3 run2 上保存六个固定预算末端 coal Policy 并做只读检查。四个版本独立保留，不合并为新增训练 seed。

## 原生动作链机械正对照

32 个诊断布局（seed 0--31）× tree/stone/coal，共 96 个场景/目标组合。几何脚本在位置和朝向联合状态上做 BFS，计入初始朝向、被材料阻挡时原地转向和最终 `do`，保留所有并列最短首动作。实际动作逐步调用 `env.step`；没有直接移动玩家或修改库存来完成采集。

| 每集上限 | 脚本成功数 |
|---|---:|
| 1 步 | 7/96 |
| 2 步 | 40/96 |
| 4 步 | 92/96 |
| 8 步 | 96/96 |
| 16 步 | 96/96 |
| 32 步 | 96/96 |

最短链为 1--6 步，平均 2.667 步。在 32 步组中，approach 117/117、turn 43/43、do 96/96 均符合预期位置、朝向和库存增量。六个上限共执行 576 集机械正对照。

因此，抽查场景的有效动作链可以在 32 步内完成。这不证明 CNN/PPO 一定能在当前训练预算内学会，也不排除随机探索耗尽 32 步。96/96 属于脚本环境审计，不计入 A/B/C 方法成绩。

## 冻结 Policy 在共同 RGB 上的读数

每个版本固定 A/B/C × seed 11/12 的最终 coal Policy（4,096 target support 步）。脚本产生同一组 32 个 coal 动作链，共 83 个状态：35 个 approach、16 个 turn、32 个 do。六个策略共读取 498 次分布，不采样策略动作、不更新参数。

指标为该状态下“能进入最短原生采集链的动作集合”的概率质量，允许并列动作。它不惩罚等长替代路线，但可能排除最终仍能成功的较长路线；它不是自主轨迹成功率。角色为已查看的诊断状态，不是正式独立测试集。

修复版 v2 的结果如下：

| 固定 Policy | approach 集合概率 | turn 集合概率 | 可采集时 do 概率 |
|---|---:|---:|---:|
| A / seed 11 | 14.57% | 12.19% | 34.91% |
| B / seed 11 | 10.97% | 10.22% | 54.76% |
| C / seed 11 | 6.53% | 6.42% | 68.98% |
| A / seed 12 | 24.74% | 17.73% | 8.61% |
| B / seed 12 | 10.21% | 7.11% | 51.88% |
| C / seed 12 | 20.92% | 22.18% | 6.83% |
| 7 个允许动作均匀分布参照 | 17.14% | 14.29% | 14.29% |

B/seed 11 在 approach、turn、do 状态的平均 `p(do)` 分别为 54.58%、54.77%、54.76%；B/seed 12 为 51.32%、52.01%、51.88%。这些读数显示，B 的 `do` 偏好没有明显随可采集条件切换，接近和转向集合概率仍低。A/C 的 seed 12 到可采集状态也很少选择 `do`。

历史 v1 的 B/seed 11 集合概率为 5.39%/3.37%/68.29%，B/seed 12 为 22.68%/16.76%/5.13%。修复学习率后 B 的表现改变，旧版的 seed 差异不能直接推广到当前版本。

当前能定位的是动作阶段条件化不足的表现；RGB 表示读出、动作头、状态覆盖、优化和奖励/信用分配的责任仍未分开。检查只读取 coal Policy；stone 只参加机械正对照，未验证其学习成功率。

## 来源 scene pool 可比性审计

带 manifest 的 v2 run2 显示，source 角色本身没有与 target/evaluation 交叉污染，但 B/C 的实际 scene seed 集合不相同：seed 11 的 B/C 实际 scene 数为 289/283，seed 12 为 288/289；B 的 query scene 没有进入 C。新的严格审计将此判定为失败，而不是只报告总步数相等。

v3 为每个 source slot 声明 64 个 support 和 64 个 query 候选 scene seed，support/query 池彼此不重叠，B 与 C 每个 replica 的声明池均为 2,048 个 scene seed，交集和并集相等。B/C 仍独立采样和提前结束，所以实际访问子集不同（seed 11 为 299/312，seed 12 为 275/276）；这属于策略轨迹差异，不是 candidate pool 不一致。v3 的场景池审计为 `valid=true`、`provenance_valid=true`、`strict_source_scene_pool_equal=true`。

v2 的 B `25/80` 终点优势在 v3 对齐池后变为 B `7/80`、C `19/80`；这说明 v2 信号对 source pool 边界敏感，不能作为方法优势证据。

## 发现并修复的学习率错误

v1 在创建 FOMAML optimizer 后，无条件用普通 PPO 的 `policy_learning_rate=0.0003` 覆盖两个来源 optimizer，导致声明的 `meta_learning_rate=0.001` 未实际生效。v1 的预算/接口审计没有检查实际学习率，因此旧 `valid=true` 不能证明这一配置边界。

`908eb4c` 按 arm 构建来源 optimizer：B 外循环为 0.001，C 普通更新为 0.0003，内循环和 target PPO 仍为 0.0003。结果新增实际学习率账本，独立 analyzer 会拒绝记录值与配置不符的运行；旧记录缺少该字段时明确标注未验证。回归测试覆盖了这次错误。

修正配置为 [`crafter_gather_spt_transfer_pilot_v2.yaml`](../../../configs/crafter_gather_spt_transfer_pilot_v2.yaml)，场景池校准配置为 [`crafter_gather_spt_transfer_pilot_v3.yaml`](../../../configs/crafter_gather_spt_transfer_pilot_v3.yaml)。v3 的数值设置、任务、两个 seed、来源 8,192 步和目标 4,096 步保持不变，只增加 paired candidate pool 并让 C 合并两个 256 步块后更新。

| 版本 | A 终点 | B 终点 | C 终点 |
|---|---:|---:|---:|
| v1：实际 B 外循环 0.0003 | 15/80 | 7/80 | 8/80 |
| v2：B 外循环按配置 0.001 | 15/80 | 25/80 | 8/80 |

完整曲线见[集成 pilot 记录](crafter-gather-spt-transfer-pilot-2026-10-15.md)。v2 终点共 240 集：48 成功、192 external truncation、0 死亡。B 的小样本改善是进一步检查的信号；它不建立正式 SPT 优势，也不能把全部失败归因于该错误。没有增加预算来选择获胜方法。

## 验证和封存

- 定向测试 `19 passed`，全量回归 `437 passed`。
- v1 run8 相对 run7，六臂/seed 的全部原有训练与评价记录完全一致；这是已记录指标的独立进程复现，不是完整动作轨迹回放。
- v2 独立 analyzer：`valid=true`、`source_optimizer_rates_verified=true`、warnings 为空；v2 run2 数值复现与 run1 一致，但严格 source pool 审计失败。
- v3 run1/run2 独立 analyzer 均为 `valid=true`、`source_optimizer_rates_verified=true`；两次场景池审计均为 `valid=true`、`strict_source_scene_pool_equal=true`。数值训练记录、episode/scene/RGB manifest 和优化器设置逐项一致。
- 两版动作链结果均在另一个进程核对机械案例数、概率和动作集合账本、checkpoint 文件/参数哈希、4,096 步标签和共享 RGB 状态。各策略 probe 前后参数不变，六策略读取同一状态集合，核对结果 `valid=true`。
- 结果位于 `results/crafter_gather_spt_transfer_pilot_v1_run8/`、`results/crafter_gather_spt_transfer_pilot_v2_run1/` 和 v3 的 `run1`/`run2` 目录；v3 run2 另有 `reproduction_verification.json`，各动作链目录包含 `diagnostic.json` 和 `verification.json`，保存于实验主机。

几何 oracle 的使用范围仅为机械正对照动作和诊断标签；A/B/C 训练、目标适应和自主评价仍使用固定动作 allowlist，不接收几何标签，也不替换动作。本轮没有生存干预、活动 SPT 切换、Module 注册或 Knowledge 更新。

## 下一项

E2 runner 已将 v3 pool 分配扩展到三个 leave-one-binding-out fold，并在一个 seed 的 bounded smoke 中通过 resolved binding、RGB provenance、paired pool 和角色隔离审计。candidate manifest 的成本/provenance 预检也通过，跨 replica 角色 seed 区间无碰撞；下一步确认正式 hash 封存和统计边界。

这些边界明确后，再判断预定预算是否提供足够学习信号并冻结正式 E1/E2。不会因本次 B 终点较高就直接启动正式训练，也不自动转入新 head、奖励或表示消融。

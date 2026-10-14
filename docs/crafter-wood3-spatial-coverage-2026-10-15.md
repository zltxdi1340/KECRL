# Crafter `wood >= 3`：episode 隔离的状态覆盖消融

本轮诊断针对一个具体问题：在固定同一个 100k 步 CNN-only Policy 和同一个 `conv2_max2x2` 空间动作头后，低成功率是否主要来自训练状态覆盖不足。

## 实验边界

- 源 Policy 固定为已验证的 100k 步 CNN-only checkpoint，卷积编码器、critic、原始 actor 和 optimizer 都不更新。
- 三个 head 训练臂使用相同的动作集合监督、相同的特征和相同的 head 优化设置：
  - `old_native_216`：既有 native 状态训练集，216 行；
  - `new_native_216`：新采集的 40 个独立 environment episode 中抽取的 216 行；
  - `merged_native_432`：前两者合并，432 行。
- 新采集按 environment episode 划分为 train/validation/held-out，分别为 40/20/20 个 episode。相同 episode 不会跨 split；RGB 重复也不能跨 split。
- validation 只用于 head 的 epoch/LR 选择。held-out 在 selection lock 写入后才读取，不能参与拟合或选择。
- 采集和 fresh 评估都保留自然死亡，没有动作 oracle、生存干预或死亡 episode 排除；`do`、转向和接近动作都由策略采样。
- 每个 fresh episode 使用四个 arm 共同的环境 seed/action seed，分别独立执行；只比较真实轨迹和结果。

## 读数和解释

独立 analyzer 会重放全部新采集 episode，重建数据集与 action-set 标签，重算三个 head 的特征、参数和 held-out readout，并重新执行 fresh 轨迹。只有 runner 的 cross-process repetition、输入封存、数据集重放、held-out 只读评估和 fresh 轨迹复核全部通过，结果才进入诊断摘要。

- 若 `new_native_216` 明显优于 `old_native_216`，说明既有状态覆盖不足是重要瓶颈。
- 若 `merged_native_432` 继续明显优于两者，说明覆盖量仍不足；后续应增加自然 episode 覆盖，而不是先修改 SPT/SPI 接口。
- 若三个 head 的 held-out readout 改善但 fresh 成功率不改善，瓶颈在 on-policy 生存、导航或信用分配，而不是局部动作头容量。
- 若 fresh 成功率随覆盖变化而改善，同时死亡阶段和动作链指标改善，才有理由把状态覆盖作为当前主要干预方向。

本轮仍只诊断 `wood >= 3`，不能把结论直接推广到 stone、coal 或 pickaxe。

## Pilot 结果（2026-10-15）

80 个新 source episode 共得到 6,926 个唯一 RGB 状态；train/validation/held-out 共 40/20/20 个 episode。四个 fresh arm 各 40 个配对环境/action seed。runner 跨进程重复、采集环境重放、source 概率与动作重采样、head 独立拟合、held-out 重算和 fresh 轨迹重放均通过；无 action oracle、生存干预或死亡样本排除。

| arm | wood≥3 成功 | death | wood1 / wood2 / wood3 | 死于 wood1 前 |
|---|---:|---:|---:|---:|
| source CNN-only | 25/40 (0.625) | 15 | 35 / 25 / 25 | 5 |
| old native 216 | 8/40 (0.200) | 30 | 20 / 13 / 8 | 19 |
| new native 216 | 8/40 (0.200) | 30 | 20 / 13 / 8 | 19 |
| merged native 432 | 16/40 (0.400) | 21 | 35 / 27 / 16 | 5 |

配对精确 McNemar 检验中，source 相对 old/new 各为 19 对 2 个不一致成功，`p=0.00022`；source 对 merged 为 14 对 5，`p=0.0636`；merged 对任一 216 臂为 12 对 4，`p=0.0768`。这些是单 seed、每臂 40 集、未做多重比较校正的 pilot 统计。merged 比 216 臂多 8 次成功是方向性信号，尚不足以宣布稳定提升。old/new 的 40 个配对成功/失败标签完全相同；它们的拟合参数不同，但 fresh 成败结果相同。

validation 的 category-macro greedy accuracy 四臂训练后都停在 0.333。old/new 两个 head 都选 epoch 0，继续优化使 validation set-NLL 从约 1.892 恶化到 3.72/3.67；merged 选 epoch 50，validation NLL 降到 1.758，独立 held-out NLL 为 1.627（old/new 约 1.792）。greedy accuracy 仍为 0.333，改进来自转向/接近动作集合概率质量，而不是 top-1 类别率。held-out turn 类的有效动作概率从 source 的 0.072 提升到 merged 的 0.140；approach 从 0.180 到 0.192。

这说明“新分布的 216 行”没有优于既有 216 行，而“旧+新合并至 432 行”让优化能在独立 validation/held-out 上改善动作集合概率，并减少首个 wood 前的死亡。不过 merged 仍只有 16/40 成功，且有 9 次在取得 2 个 wood 后死亡；source 有 25/40 成功且没有这一阶段死亡。死亡提示为 source 的 12 次 hostile/projectile、1 次 depletion、2 次 mixed/unknown；old/new 各 18/2/10；merged 为 10/2/9。因而覆盖量有信号，但动作头仍弱于原策略，且后半链与生存问题仍未解决。

下一步应先做一项**等训练量对照**来拆分“更多样本”与“来源/状态覆盖更广”：固定 432 行和相同 validation/held-out，比较 432 行旧分布子集、432 行新分布子集与 216+216 混合；同时报告 validation 曲线，避免只看最优 checkpoint。fresh 评估应扩大到至少 100 个配对 episode，并保留当前 source control。若旧分布没有足够 episode 隔离的 432 行，则先扩采旧分布，再拟合；不要直接把当前 merged 的收益归因于 coverage diversity。

紧凑结果和独立验证记录在 `results/crafter_wood3_spatial_coverage_cuda_pilot_20261015_v1/analysis_verified/`。这些结果只是 `wood >= 3` 的诊断，未证明 Module 资格，也没有验证其他技能任务。


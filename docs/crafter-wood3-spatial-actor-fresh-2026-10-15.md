# Crafter wood>=3 RGB 空间动作头 fresh 诊断（2026-10-15）

本轮检验：在同一个已经训练好的 CNN-only Policy 上，用相同 native 状态与相同监督预算，比较最终 embedding 动作头、保留 conv2 空间布局的动作头，以及未修改的 source Policy，能否在 fresh episode 中自主完成 `wood >= 3` 动作链。它是单 source seed 的诊断 pilot，不是正式训练或 Module 资格结果。

## 实验边界

- Source 是 seed 0、100k interaction-step 的 CNN-only checkpoint。source encoder、actor、critic 和 PPO optimizer 均保持冻结；fresh 每集采样 RNG 由预先锁定的 action seed 单独设置。没有 PPO 更新、环境策略训练、Knowledge/SPT 更新或 Module 注册。
- 两个新动作头使用完全相同的 216 条 native RGB 训练状态：ready 90、turn 30、approach 96；使用独立的 115 条 native validation 状态选择 epoch/LR。局部几何 action-set 标签用于训练状态拟合，也用于 validation 选择；heldout 未参与拟合或选择。两个头都是 128 输入、6 输出、774 个参数，学习率候选为 0.003/0.03，每候选 1500 次离线更新。
- 协议字段 `teacher_scope` 的原始文字写成“train rows only”，范围表达不完整：validation action-set 标签也参与 epoch/LR 选择。报告以上述实际执行过程为准；heldout 标签未参与任何拟合或选择，后续协议会明确区分 train fit、validation selection 与 fresh evaluation。
- 最终 embedding head 选中 epoch 750，conv2 `adaptive max pool 2x2` head 选中 epoch 200；两者学习率均为 0.03。Source CNN-only arm 不拟合新 head。
- fresh 执行只看 RGB，不使用动作替换或状态 oracle。所有臂使用相同的 40 个环境 seed 和动作 seed；每个 seed 对每臂都是独立策略轨迹，初始 RGB 一致。动作 allowlist 固定为 0–6，采样执行，最多 256 步。
- 两个拟合臂保留 source Policy 在 allowlist 下逐状态原有的 `p(do)`，只把剩余非-do 概率质量重新分给六个移动/休息动作。逐步概率审计的最大绝对差分别为 final embedding `1.19e-7`、conv2 `5.96e-8`，均低于锁定的 `5e-7` 浮点容差。action 5 行及 source 模型本身没有更新。
- 死亡、外部截断和未达目标 episode 全部保留；没有生存干预，也没有按具体死亡原因排除统计。本轮没有做 fall/drowning 等原因分层。

## Fresh episode 结果

| 策略 | 成功 | 成功率（Wilson 95% CI） | 死亡 | 外部截断 | 交互步数 | 到达 wood 1/2/3 的 episode 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 source CNN-only | 24/40 | 60.0% (44.6–73.7%) | 15 | 1 | 5190 | 35 / 29 / 24 |
| 最终 embedding head | 13/40 | 32.5% (20.1–48.0%) | 27 | 0 | 6198 | 22 / 14 / 13 |
| conv2 空间 head | 22/40 | 55.0% (39.8–69.3%) | 18 | 0 | 4162 | 32 / 25 / 22 |

相同 episode seed 配对后的成功结果如下。精确 McNemar 检验是未校正的探索性比较，不能替代多个 source Policy seed 的复现：

| 配对 | 两者均成功 | 仅左侧成功 | 仅右侧成功 | 两者均失败 | 右侧减左侧 | 双侧精确 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source vs 最终 embedding | 11 | 13 | 2 | 14 | -27.5 个百分点 | 0.0074 |
| source vs conv2 | 16 | 8 | 6 | 10 | -5.0 个百分点 | 0.7905 |
| conv2 vs 最终 embedding | 8 | 14 | 5 | 13 | -22.5 个百分点 | 0.0636 |

单 pilot 的证据显示，最终 embedding head 明显弱于未修改 source；conv2 head 与 source 接近，但当前样本没有证明它更好。它改善了最终 embedding head 的结果，却未超过原始策略。

## 失败定位

验证集 action-set 的平均有效动作概率质量显示，两个拟合头在 native validation 状态上能把更多概率分给 turn/approach 有效动作：

| 策略 | validation turn 有效概率质量 | validation approach 有效概率质量 | fresh 自访 turn 有效概率质量 / 实际命中率 | fresh 自访 approach 有效概率质量 / 实际命中率 |
| --- | ---: | ---: | ---: | ---: |
| 原 source CNN-only | 3.8% | 18.4% | 7.4% / 7.6% | 18.5% / 18.5% |
| 最终 embedding head | 22.3% | 19.3% | 3.0% / 3.6% | 14.9% / 15.7% |
| conv2 空间 head | 19.1% | 28.3% | 6.6% / 7.4% | 13.8% / 13.7% |

validation 与 fresh 自访状态的方向不一致：两个 head 在 validation 上的 turn 概率质量明显升高，但在各自 fresh 轨迹上都低于 source；approach 也没有转化成更高命中率。现有证据因此更支持优先检查状态覆盖和 on-policy 分布泛化，而不是直接把问题归结为“CNN 不含空间信息”。这仍是定位线索：fresh 三臂访问了不同状态，当前记录不能单独区分训练行数量、标签读出、概率重分配和访问分布变化各自的因果贡献。

死亡阶段计数也与此相符：最终 embedding 臂在获得第一根 wood 前死亡 18 集，source 为 4 集，conv2 为 8 集；总死亡分别为 27、15、18。该结果说明失败常发生在链条早段，但它没有分析具体伤害原因，不能据此开始 survival intervention。

## 验证状态与解释边界

- 主运行、三臂完整跨进程重复全部通过；训练行、验证行、拟合特征、head 状态和 120 个 fresh episode 完整轨迹摘要一致。
- 独立 analyzer 重新加载 source checkpoint，重建 train/validation 特征，复播全部 120 个 fresh episode，并验证配对初始 RGB、概率约束、输入哈希和 source Policy/optimizer 未改变。
- 全量测试回归：`411 passed`。
- 第一次 v1 执行在严格 `1e-7` 最大误差门槛处停止：观测到的 `1.19e-7` 是 `Categorical` 浮点归一化产生的单精度舍入量级差异。该次不作为结果；完整结果来自独立锁定的 v2 配置，容差为 `5e-7`，实测最大误差仍仅 `1.19e-7`。
- 只有一个 source Policy seed，每臂 40 集；结果用于诊断而非正式效果声明。0.8 仍是待验证候选门槛，完整 Module 契约资格检查尚未执行。
- `wood >= 3` 之外的 stone、coal、pickaxe、前置工具关系、组合调用和持续任务链均未验证。没有正式 PPO 训练、Module 注册或 Knowledge/SPT 更新。

## 后续方向

同状态 cross-readout 已完成，结果见[独立诊断报告](crafter-wood3-spatial-cross-readout-2026-10-15.md)。下一步按 episode 隔离扩展 native on-policy 状态覆盖，检验相同容量 head 是否能在 turn/approach 状态上稳定泛化，并用完全独立的 heldout/fresh seed 评估。自主动作链仍未达到稳定复现；因此暂不进入正式训练或共享 survival intervention。wood 之后还需验证 stone、coal、pickaxe 和工具前置条件。

## 产物

- [锁定配置](../configs/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2.yaml)
- [动作头与概率约束](../experiments/crafter_spatial_action_head.py)
- [运行器](../experiments/run_crafter_wood3_spatial_actor_fresh.py)
- [独立分析器](../experiments/analyze_crafter_wood3_spatial_actor_fresh.py)
- [fresh 分项汇总](../results/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2_20261015_v1/analysis_verified/fresh_arm_metrics.csv)
- [配对 episode 结果](../results/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2_20261015_v1/analysis_verified/paired_fresh_outcomes.csv)
- [validation 与 on-policy 动作概率对照](../results/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2_20261015_v1/analysis_verified/validation_vs_onpolicy_action_mass.csv)
- [公开汇总](../results/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2_20261015_v1/analysis_verified/public_summary.json)
- [独立审计](../results/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2_20261015_v1/analysis_verified/verification.json)

原始 RGB、checkpoint、head artifact 和逐步 episode trace 保留在本机结果目录，没有推送到 GitHub。

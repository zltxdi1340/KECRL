# Crafter 可行性诊断状态（2026-10-15）

## 当前结论

当前仍在正式训练前的可行性诊断阶段。最新的 seed-0、100k-step source CNN-only 对照中，原始策略 40 集成功 24 集；同状态 native 几何监督的最终 embedding head 成功 13 集；conv2 spatial head 成功 22 集。主运行、完整跨进程重复、独立 checkpoint/特征重建和 fresh episode 复播均通过，全量回归 `411 passed`。

这个结果不支持直接开始正式训练：final embedding head 退化，conv2 head 虽接近 source，却没有超过 source 的证据。validation 上 head 的 turn/approach 有效动作概率质量提高，但 fresh 自访状态上两种 head 都低于 source。随后完成的同状态 cross-readout 进一步显示，final head 在自己的访问状态上的 turn 有效概率质量只有约 6.0%，而 source/conv2 在同一状态池上约为 13.9%/12.7%；conv2 在 source 状态池的 turn 质量可达约 14.8%，但没有稳定转化为更高 fresh 成功率。当前瓶颈更像状态覆盖与条件泛化的耦合，尚不能归因于单一表示或生存因素。

本轮的两个 head 只学习六个非-do 动作；原 CNN、actor/critic 和优化器冻结，`p(do)` 在每帧复用 source 策略概率，最大实测差 `1.19e-7`。局部几何 action-set 标签用于 216 条 train 状态拟合，也用于 115 条 validation 状态选择；fresh 执行没有 oracle，没有动作替换。死亡和截断全部保留，未执行生存干预或死亡原因分层。配置中的 `teacher_scope` 字段没有明确包含 validation 选择，详细报告已更正此范围说明。

## 已完成的同状态检查

已将三臂 120 集 fresh 轨迹逐步恢复为 15,550 个 RGB 状态，并让三个冻结策略在同一状态池上交叉输出 46,650 条动作概率。恢复使用封存动作，不采样或执行新动作；主运行、跨进程重复、独立 analyzer 和输入哈希检查全部通过。详细结果见[同状态 cross-readout](crafter-wood3-spatial-cross-readout-2026-10-15.md)。

阶段/死亡前窗口的 episode 宏平均显示，final head 的 27 次死亡里 18 次在第一根 wood 前，source/conv2 分别为 4/8 次。冻结 source critic 在 source path 上的 value 随 wood 阶段降低，并在死亡前 approach 状态转负；与 `gamma=0.99` Monte Carlo return 的绝对误差仍约 0.23–0.61，且死亡前 turn 样本少。因此 value 有粗粒度阶段信号，但不足以排除 reward/credit 问题，也不能把 survival 归因为唯一瓶颈。

## 下一项诊断

做按 episode 隔离的新状态覆盖对照：从独立 seed 的自然 RGB 轨迹增加 turn/approach 覆盖，比较原 216 条状态训练 head 与加入扩展覆盖后的同容量 head，再用独立 heldout 状态和全新 fresh seed 检验；heldout 不参与训练或 epoch 选择。若 coverage 能稳定改善自主链，继续 wood；若动作链稳定而死亡仍主导，再按既定顺序研究共享生存能力。

目前没有正式 PPO 训练、Module 注册、Knowledge/SPT 更新，也没有 Module 资格结论。`wood >= 3` 修通前不开始后续生存机制诊断；wood 后还需要逐项验证 stone、coal、pickaxe、工具前置条件和持续任务链。0.8 仍是待验证候选门槛。

详细协议与配对结果见[空间动作头 fresh 诊断](crafter-wood3-spatial-actor-fresh-2026-10-15.md)。

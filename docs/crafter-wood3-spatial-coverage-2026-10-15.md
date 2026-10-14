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


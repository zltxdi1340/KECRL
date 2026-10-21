# LIBERO Spatial 示教轨迹阶段结构审计（2026-10-21）

这一步检查候选 `PickPlace(Bowl, Plate)` Skill Family 的演示是否提供了“接近—抓取—搬运—释放”程序抽象可以依靠的可观测时序证据。它是进入 Policy/SPI 学习前的数据诊断，不是训练、checkpoint 选择、Module 资格检查或正式实验。

## 审计边界

- LIBERO checkout、数据 revision 和 10 个任务沿用已冻结的 dataset manifest；角色沿用候选 transfer split，仅用于按角色汇总描述。
- 读取每条 HDF5 演示的 `actions`、`robot_states`、`rewards` 和 `dones`。没有读取 RGB 来给阶段打标签，也没有用隐藏物体位姿或完整 `states` 推断接触事件。
- 夹爪动作最后一维以 `-1` 记为打开、`+1` 记为闭合，去除零值保持后统计符号运行。`robot_states[:, 2:5]` 只用于描述末端位置位移，不是阶段标签。
- “第一次闭合”与其后的“第一次重新打开”是动作边界证据，不等于已经观测到抓取成功或物体接触。

可重复入口为 [`scripts/audit_libero_demonstration_structure.py`](../scripts/audit_libero_demonstration_structure.py)，接口单元测试为 [`tests/test_libero_demonstration_structure.py`](../tests/test_libero_demonstration_structure.py)。

## 总体结果

审计覆盖 10 个任务的 500 条演示，共 62,250 个 transition：

| 指标 | 结果 |
|---|---:|
| 从打开命令开始 | 500/500 |
| 观察到第一次闭合 | 500/500 |
| 观察到第一次重新打开 | 493/500 |
| 前三段命令为“开—闭—开” | 493/500（98.6%） |
| 严格只有一次“开—闭—开”循环 | 443/500（88.6%） |
| 轨迹结束时夹爪打开 | 486/500 |
| 轨迹结束时夹爪闭合 | 14/500 |

最常见的夹爪命令骨架为 `-1, +1, -1`（443 条），其次是带额外循环的 `-1, +1, -1, +1, -1`（41 条）。另有 7 条在第一次闭合后直接结束，说明“释放”不是每条成功演示都以一个独立的重新打开命令显式出现。

第一次闭合前、中间闭合段、第一次重新打开后的步数比例中位数分别为 `0.361`、`0.521`、`0.097`。这支持“闭合后长距离搬运”的粗粒度解释，但不能证明闭合期间始终抓住目标，也不能把任意一步划为接近、抓取、搬运或释放。

## 按实验角色汇总

| 角色 | 任务数 | 演示数 | “开—闭—开” | 严格单循环 | 第一次重新打开 |
|---|---:|---:|---:|---:|---:|
| source SPI | 5 | 250 | 246 | 219 | 246 |
| SPT validation SPI | 2 | 100 | 99 | 91 | 99 |
| held-out query SPI | 3 | 150 | 148 | 133 | 148 |

三个角色的粗骨架比例都接近，未见某一角色缺失基本夹爪时序。但是这些是已固定数据的描述统计，不能用来选择角色，也不能替代冻结后的策略评估。

## 对 SPT/SPI 设计的含义

1. 候选 SPT 的四阶段程序可以保留为**人工定义的接口抽象**：接近源对象、抓取、搬运到目标、释放。演示中的夹爪开合为这一抽象提供了高层时序支持。
2. 当前数据没有阶段标注、接触标签或成功抓取标签。`open-close-open` 只能作为粗粒度事件边界，不能直接构造监督 phase label，也不能据此声称 Policy 学到了可复用的四阶段 Module。
3. 额外开合和末端闭合终止在不同任务均存在。后续 Policy 训练应允许动作时序有弹性；评价应使用环境成功谓词和完整 Module contract，而不是精确匹配一条演示动作序列。
4. 不同初始几何仍造成显著运动差异：例如高位 fixture 与开放抽屉任务的闭合段位移更长。SPT 上下文至少要保留 `source_predicate`、`source_anchor` 和 `source_geometry`，不能只学习一个不带实例绑定的固定动作模板。
5. 本审计没有验证 RGB 是否足以读出源/目标，也没有验证 head、记忆、奖励或元学习是否是当前瓶颈；这些必须在 CUDA 学习 smoke 和后续 A/B/C 诊断中单独测量。
6. 后续闭环归因确认，数据生成脚本把执行 `action[t]` 后返回的 observation 与 `action[t]` 存在相同索引。因此本页的夹爪动作骨架统计仍成立，但 Policy 监督不能直接使用 HDF5 same-index `obs[t] -> action[t]`；必须重建 causal pre-action 数据视图。证据见 [`libero-bc-closed-loop-attribution-2026-10-21.md`](libero-bc-closed-loop-attribution-2026-10-21.md)。

## 当前结论与下一步

当前结论是：LIBERO Spatial 的 10 个任务具备一致的高层夹爪时序骨架，足以继续做候选 SPT/SPI 的受限学习；四阶段低层程序仍是待学习和待资格验证的假设。该结果不构成正式实验结果，也不产生合格 Module。

后续已完成 CUDA 运行时、候选行为克隆 Policy 的有界 pipeline 检查和单 SPI 闭环归因，结果见 [`libero-cuda-training-path-smoke-2026-10-21.md`](libero-cuda-training-path-smoke-2026-10-21.md)、[`libero-bc-policy-pipeline-check-2026-10-21.md`](libero-bc-policy-pipeline-check-2026-10-21.md) 与 [`libero-bc-closed-loop-attribution-2026-10-21.md`](libero-bc-closed-loop-attribution-2026-10-21.md)。下一项是在 source 边界内验证因果对齐数据视图，再决定多 source learning pilot 和固定预算 A/B/C 迁移诊断。

## 复现命令

```bash
/opt/anaconda3/bin/conda run --no-capture-output -n libero-kecrl \
  python scripts/audit_libero_demonstration_structure.py \
  --dataset-root /home/zl202621/third_party/LIBERO/datasets \
  --split-manifest configs/libero_spatial_transfer_split_candidate_v1.json
```

本命令只读外部固定数据，不训练模型，不写入仓库生成物。

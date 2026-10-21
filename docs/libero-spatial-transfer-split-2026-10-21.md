# LIBERO Spatial 迁移划分与泄漏审计（2026-10-21）

本记录冻结学习 pilot 前使用的候选任务角色和数据索引。它的作用是让 A/B/C 后续比较可复现，并确保 source 更新、SPT validation、Module qualification 和最终 held-out 评价不会读取同一批样本。本记录仍为候选协议：没有冻结模型、优化器、预算或统计门槛，也没有启动训练。任务族检查只验证 BDDL 语义、对象和接口一致；演示的粗粒度夹爪时序已在 [`libero-demonstration-phase-structure-audit-2026-10-21.md`](libero-demonstration-phase-structure-audit-2026-10-21.md) 中单独审计，仍不构成低层四阶段标签。

## 关键发现：两套初态不能按编号配对

每个 `libero_spatial` 任务同时提供：

1. HDF5 中 50 条 demonstration，各自带一个 92 维 `init_state`；
2. 官方 `.pruned_init` 中 50 个用于在线 benchmark 的 92 维固定初态。

逐任务比较表明，这两套集合都各有 50 个唯一有限值状态，但没有任何完全相同的 raw state；相同编号也不是对应样本。把 `demo_7` 和 `.pruned_init[7]` 当作同一场景会制造错误的可比性声明。

因此候选协议将它们设为两个独立命名空间：

- demonstration index 只划分离线 support/query 数据；
- evaluation initial-state index 只划分在线 checkpoint selection、candidate acceptance、qualification 和 final evaluation；
- 任何代码不得通过相同整数索引把两者连接起来。

“零个完全相同 raw state”只排除了逐字节状态复用，不证明两套分布在语义或几何上独立。

## 任务角色

任务角色保存在 [`configs/libero_spatial_transfer_split_candidate_v1.json`](../configs/libero_spatial_transfer_split_candidate_v1.json)：

| 角色 | task id | 几何覆盖 | 允许用途 |
|---|---|---|---|
| source SPI | 0, 1, 3, 4, 7 | table、movable surface、open container、fixture | B 的 FOMAML 更新；C 的等预算普通预训练；source 非退化检查 |
| SPT validation SPI | 2, 6 | table | checkpoint/适应设置选择与候选 SPT 接受，不进入最终效应估计 |
| held-out query SPI | 5, 8, 9 | movable surface、table、fixture | 方法冻结后的 qualification 和最终 A/B/C 迁移比较 |

唯一 open-container 任务 4 放在 source。这样最终 held-out 不会把“从未出现过的初态几何类别”与“SPT 更新是否有帮助”混在一起。held-out 同时保留 movable、table 和 fixture 三种已经在 source 出现过的几何类型。

## Demonstration 索引

每个 SPI 的 50 条演示按任务角色解释：

| 任务角色 | indices 0--29 | indices 30--49 |
|---|---|---|
| source | `meta_support` | `meta_query` |
| SPT validation | `adaptation_support` | `selection_query` |
| held-out query | `adaptation_support` | `sealed_query` |

held-out 的 `sealed_query` 在 A/B/C、适应预算、checkpoint 规则和实现代码冻结前不可读取。A、B、C 在 validation/held-out SPI 上必须使用完全相同的 adaptation support 前缀和更新次数。

## 在线固定初态索引

`.pruned_init` 的 50 个状态独立划分：

| 任务角色 | 在线索引用途 |
|---|---|
| source | 0--19 candidate non-regression；20--39 qualification；40--49 保留不读 |
| SPT validation | 0--9 checkpoint selection；10--29 candidate acceptance；30--49 保留不读 |
| held-out query | 0--19 qualification；20--49 final evaluation |

qualification 前必须冻结 Policy 和适应过程；qualification 结果只能决定是否形成 Module，不能反向选择 checkpoint 或修改预算。最终评价只读取一次；无论 qualification 是否通过，A/B/C 的效应分析都保留，失败不能触发调参。

## A/B/C 数据边界

- **A：原始 SPT。** 不读取 source 数据来更新 SPT，只用各 query SPI 允许的 adaptation support 生成并适应实例 Policy。
- **B：SPT 更新。** 只从 source SPI 的 `meta_support` 和 `meta_query` 接收 FOMAML 反馈，产生候选 SPT；validation 数据只能选择/接受，不能进入元梯度。
- **C：普通预训练。** 读取与 B 完全相同的 source task、demonstration 和更新预算，模型容量相同，但不使用元目标。

B 优于 A 才说明来源 SPI 经验对后续实例有帮助；B 优于 C 才能把收益与普通预训练区分。当前划分只建立了可检验边界，没有提供任何一项效果证据。

## 自动审计

执行：

```bash
PYTHONPATH=. conda run -n libero-kecrl \
  python scripts/audit_libero_transfer_split.py \
  --init-root /path/to/LIBERO/libero/libero/init_files \
  --dataset-root /path/to/LIBERO/datasets
```

审计器会拒绝：task role 重叠或遗漏、索引重叠或遗漏、几何覆盖漂移、revision/task map 不一致、初态文件大小或 SHA-256 改变、状态维度错误、同一命名空间内重复状态，以及 demonstration 与在线评价集合中的完全相同 raw state。LIBERO 的 `.pruned_init` 是固定 revision 提供的 NumPy legacy pickle；审计器只在哈希校验通过后读取它，并显式使用 `weights_only=false` 兼容该官方格式，不接受用户提供的 checkpoint 或可执行对象。

## 仍未冻结

- Policy 架构和 SPI context encoder；
- BC/FOMAML 的 loss、batch、学习率和更新次数；
- adaptation support 预算曲线与停止规则；
- SPT candidate acceptance 和 Module qualification 阈值；
- A/B/C 的严格等更新量核算；
- 候选正式模型的显存和吞吐；CUDA 运行时连通性已通过有界 smoke。

这些项目只能使用 source 与 SPT validation 角色做 pilot。若候选划分在 pilot 前需要修改，应创建 `v2` 并保留本版本；一旦读取 held-out query 或 final evaluation，本版本不得原地修改。

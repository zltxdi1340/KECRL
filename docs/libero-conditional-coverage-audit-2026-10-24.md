# LIBERO 条件覆盖审计（2026-10-24）

## 目的与范围

本次审计检查源 SPI 的 `meta_support` 与 `meta_query` 是否在 task relation、source geometry 和动作阶段上存在明显的条件缺口。它是机制诊断，不是 Policy 训练、checkpoint 选择、SPT/Knowledge 更新或 Module 资格检查。

审计读取源任务 `0, 1, 3, 4, 7` 的 50 个示教分区：每个任务的 `0–29` 为 `meta_support`，`30–49` 为 `meta_query`。只读动作、`obs/ee_pos`、`obs/ee_ori`、joint/gripper 状态和示教元数据；没有读取 RGB、`states`、`init_state`、`model_file`，没有访问 official candidate、validation、held-out 或 qualification 状态。

动作阶段由 gripper action 的符号确定：首次正值前为 `approach`，首次正值到其后的首次负值为 `transport`，之后为 `release`。`ee_pos` 只作为可观测末端轨迹代理，不是目标相对位姿，也没有加入 Policy 输入。

## 可复现信息

- 配置：[libero conditional coverage audit](../configs/libero_conditional_coverage_audit_v1.json)，SHA-256 `6b8446a06cd5d4b2cf642a83009d4f1ceb983c2bb3f6d5c566f5810179d9d098`。
- 结果文件：`/tmp/libero-conditional-coverage-20261024-v2.json`，SHA-256 `adb9b054e35ba140b4c2c56b6905120662e642d348429d3fe5e98065f53cf6da`。
- 运行提交：`a78577b4424388b6d9912a761f02062dd2484345`。
- 五个 HDF5 文件均通过 dataset manifest 的尺寸和 SHA-256 校验。
- `formal_result=false`；`training_run=false`；`checkpoint_selected=false`；`spt_updated=false`；`knowledge_updated=false`；`module_qualified=false`。

## 结果

### 关系与几何集合

| 条件 | meta-support | meta-query |
|---|---|---|
| task relation | `between`, `next_to`, `on_support`, `inside_container` | 完全相同 |
| source geometry | `table_region`, `movable_object_surface`, `open_container`, `fixture_surface` | 完全相同 |
| relation × geometry 组合 | 5 个 | 完全相同 |
| query 缺少 support 组合 | 0 |  |
| support 缺少 query 组合 | 0 |  |

五个组合为：`between × table_region`、`next_to × table_region`、`on_support × movable_object_surface`、`inside_container × open_container`、`on_support × fixture_surface`。因此没有发现“查询集完全没有某个关系或几何类别”的简单覆盖断裂。

### 动作阶段覆盖

| 角色 | 总有效步 | approach | transport | release | unknown |
|---|---:|---:|---:|---:|---:|
| meta-support | 18,798 | 6,467 | 9,739 | 2,592 | 0 |
| meta-query | 12,619 | 4,377 | 6,559 | 1,683 | 0 |

每个任务、每个角色都包含三个实际阶段。末端轨迹路径长度中位数如下，support/query 没有整体数量级断裂：

| task | relation / geometry | support 路径中位数 | query 路径中位数 |
|---:|---|---:|---:|
| 0 | between / table_region | 0.8648 | 0.8682 |
| 1 | next_to / table_region | 1.1609 | 1.1348 |
| 3 | on_support / movable_object_surface | 0.8926 | 0.8921 |
| 4 | inside_container / open_container | 1.1921 | 1.1827 |
| 7 | on_support / fixture_surface | 1.0452 | 1.0437 |

阶段比例仍有条件偏移。例如 task 1 的 release 中位比例为 support `0.0963`、query `0.1060`；task 4 为 `0.0809`、`0.0748`；task 7 为 `0.0783`、`0.0878`。这表示数据不是完全同分布，但当前审计没有把偏移与 Policy 动作误差或闭环失败做因果连接。

### 几何边界

LIBERO 的 HDF5 含有 simulator state，但目标物体位姿属于环境私有状态。本审计没有使用 `states` 或模型 XML 推导“末端到目标”的真实相对距离；`ee_pos` 只能说明末端轨迹的绝对位置和位移代理。因此仍不能判断图像中目标相对位置是否在 support/query 间充分覆盖，也不能据此判断 RGB 表示是否丢失了目标关系。

## 诊断结论

1. 当前 source-SPI support/query 已覆盖相同的关系类别、几何类别和五个关系×几何组合，简单的类别缺失不是动作头诊断失败的充分解释。
2. 三个动作阶段在两侧都存在，但阶段比例和动作/末端位移分布有任务条件偏移；这更适合做分层误差分析或阶段平衡对照，不能直接推出需要重采样。
3. task 0 的 action-head 实验本身只覆盖 `between × table_region`；本次跨五任务审计补足了类别层面的检查，但没有补足目标相对几何标签，也没有验证跨关系的 Policy 泛化。
4. 这次审计没有改变 SPT、Knowledge、SPI、Policy 或 Module 资格状态，不能作为正式 SPI 结果。

## 下一项

固定当前 causal pre-action Policy 和数据分区，按 `task relation × source geometry × action phase` 分层汇总 expert-state action/translation/gripper 误差，并比较各层 support/query 差异。若误差集中在某一阶段或几何组合，再做预先声明的阶段平衡或表示输入诊断；若各层误差相近，则继续检查闭环反馈和执行误差。真实目标相对几何只有在明确设计不越过 Policy 边界的观测代理后才能加入，不能直接读取 simulator object pose。

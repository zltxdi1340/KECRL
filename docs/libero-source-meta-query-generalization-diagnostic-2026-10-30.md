# LIBERO task 4/7 source 示教泛化诊断

## 目的与协议

前一轮冻结层级读出只在 `meta_support` 上报告拟合误差。本诊断检验同一 source SPI 的不同示教上是否保留该误差排序，不把结果解释为跨任务迁移或 Module 资格。

- 任务：source SPI task 4、task 7；每任务独立训练。
- 训练示教：`meta_support` demonstrations `0–29`。
- 查询示教：不重叠的 `meta_query` demonstrations `30–49`。
- 输入：HDF5 `causal_pre_action` 视图。初始 RGB 由各示教 `init_state` 在 LIBERO 中重建；固定观测采集 seed 为 `20261022`。
- Policy：CNN-only，统一七维 action MSE；训练 seed 为 `20261022/20261023/20261024`，每模型 Adam 500 updates。
- 层级：full Policy、冻结 visual encoder、冻结 visual encoder 与 fusion、仅 action head 可训练。每个任务×seed 的四个 arm 共用同一初始模型和 batch 顺序。
- 查询隔离：每个任务的 12 个 support fits 全部结束后才创建并读取 `meta_query` dataset。query 未参与反向传播或 checkpoint 选择。
- 规模：24 个模型、共 12,000 optimizer updates。没有执行 Policy rollout，也没有读取 candidate observation、validation、held-out、qualification 或官方评价初态。

冻结配置和 runner：[`libero-runtime-native-task4-task7-support-query-hdf5-diagnostic_v1.json`](../configs/libero-runtime-native-task4-task7-support-query-hdf5-diagnostic_v1.json)、[`run_libero_source_meta_query_generalization_diagnostic.py`](../experiments/run_libero_source_meta_query_generalization_diagnostic.py)。

## 结果

表内为三个训练 seed 的均值 ± sample SD；MSE 越低越好。support 与 query 均按所有有效七维 action 元素汇总。

| Task | 可训练范围 | Support MSE | Query MSE | Query − support |
|---|---|---:|---:|---:|
| 4 | Full Policy | 0.034854 ± 0.000262 | 0.038517 ± 0.000965 | 0.003664 ± 0.000956 |
| 4 | Freeze visual | 0.065068 ± 0.000906 | 0.075045 ± 0.000321 | 0.009977 ± 0.001113 |
| 4 | Freeze visual + fusion | 0.097473 ± 0.006325 | 0.112094 ± 0.006074 | 0.014621 ± 0.001168 |
| 4 | Action head only | 0.208326 ± 0.002066 | 0.214727 ± 0.002281 | 0.006401 ± 0.000217 |
| 7 | Full Policy | 0.038463 ± 0.000837 | 0.042630 ± 0.001165 | 0.004167 ± 0.001026 |
| 7 | Freeze visual | 0.065916 ± 0.001515 | 0.067789 ± 0.001532 | 0.001873 ± 0.000564 |
| 7 | Freeze visual + fusion | 0.093313 ± 0.002748 | 0.092971 ± 0.002498 | -0.000342 ± 0.001470 |
| 7 | Action head only | 0.185758 ± 0.000537 | 0.183048 ± 0.000498 | -0.002710 ± 0.000040 |

Query 排序 `full < freeze_visual < freeze_visual_fusion < action_head_only` 在 task×seed 的 `6/6` 组中成立，与 support 排序一致。Full Policy 的 query MSE 为 task 4 `0.0385`、task 7 `0.0426`；仅训练 action head 时分别为 `0.2147`、`0.1830`。query gripper MSE 也呈现较大差距：task 4 为 `0.1395` 对 `0.9505`，task 7 为 `0.1709` 对 `0.8488`（Full Policy 对 action-head-only）。

task 4 的 query 误差在四个 arm 都高于 support；task 7 的冻结 fusion 与 action-head-only query 误差接近或略低于 support。因此数据不支持“所有策略都明显过拟合”的结论。seed SD 只反映初始化变化，不能代表 query 示教抽样不确定性。

## 解释边界

该结果支持一个有限判断：在这两个已选 source SPI、固定训练预算和离线 BC 指标下，视觉表示与后续映射参与训练，比只重训 action head 更有利于拟合未参与训练的示教。它不能证明 visual encoder 或 fusion 的因果贡献，因为冻结 arm 同时改变可训练参数量和优化自由度。

CNN-only 下各 arm 的实际可训练参数数为：Full Policy `821,704`，Freeze visual `464,648`，Freeze visual + fusion `396,552`，Action head only `1,799`。最后一种只更新 action head；Freeze visual + fusion 仍更新 per-frame projection 和 action head。该参数差异也是层级对照的解释限制。

query 是同一任务的不同示教，不是新任务实例；action MSE 不是闭环成功率。没有创建或注册 Module，没有更新 SPT 或 Knowledge，结果不是正式 SPI 实验，也不能用于资格判定。task 4/7 和 3 个 seed 的规模仍属诊断性证据。

## 复现与验证

- 结果 JSON：`/tmp/libero-source-meta-query-generalization-task4-task7-20261030-v1.json`
- 结果 SHA-256：`091bbb49669d2f0c4196f856ec5f9618e51a08cbc08709dbc844fb747880ac27`
- 配置 SHA-256：`9d355ffa017b4b32b87336cecdbd89a33a67a1089cd72e37110043673b78793c`
- Runner SHA-256：`995e7618949e290f9314b9898054ee1decb82892ef3f762c49d5c9d625320b4f`
- 执行 commit：`3780739c980e507f6443433e228843224e5c9175`
- Runtime：Python `3.10.22`、PyTorch `2.4.1+cu124`、CUDA `12.4`、RTX 4090 `cuda:0`；24 模型耗时 `261.45 s`。
- 新增专项测试：`5 passed`。排除 Crafter 依赖测试文件后，其余测试为 `283 passed, 4 skipped`。Crafter 测试在 `libero-kecrl` 中因缺少可选 `crafter` 包无法运行。

## 下一步

离线 query 排序稳定后，下一诊断应检验它是否对应真实控制能力：重新拟合并保存 Full Policy checkpoint，在预先固定的 source 非资格初态上做配对闭环评估，记录成功率与逐步状态偏移。该诊断需单独冻结 rollout 初态、episode/step 预算和停止规则；不得把 query MSE 当成 Module qualification 的替代指标。

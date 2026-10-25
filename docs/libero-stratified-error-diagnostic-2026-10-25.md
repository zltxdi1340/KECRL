# LIBERO 分层 expert-state / free-run 误差诊断（2026-10-25）

## 目的和边界

本次诊断把已有 causal pre-action Policy 的逐步动作误差和 free-run 状态漂移按 `task relation × source geometry × action phase` 分层。它是单 seed 机制 pilot，不是正式训练、checkpoint 选择、SPT 更新或 Module qualification。

固定条件：source SPI 任务 `0, 1, 3, 4, 7`；每个任务训练 causal baseline Policy 500 次 Adam update；seed `20261021`；每个任务使用 meta-support `0,1,2` 训练/回放和 meta-query `30,31,32` 回放；设备为 `cuda:1` / RTX 4090。阶段由专家 gripper action 符号定义，统计只在专家 replay 成功的 episode 上汇总。专家 replay 失败的 episode 保留在结果中并单独计数，不进入误差均值。

- 配置：[stratified error diagnostic](../configs/libero_stratified_error_diagnostic_v1.json)，SHA-256 `9c5bf0d102aaad1feb3fa711f7d828e5efddc94a8cfb168096475f938aabdba1`。
- 结果：`/tmp/libero-stratified-error-20261025-v2.json`，SHA-256 `7216ec6dbd74a4262e33aee595c019c811f2c63c033cf343eca262b17a9eb444`。
- 运行提交：`a3d4debddec9869444515ae1e3d992af4daee5f6`；耗时 `453.96s`；峰值 CUDA allocation `293,302,272` bytes。
- `formal_result=false`；未访问 validation、held-out 或 qualification 状态；未更新 SPT/Knowledge，未注册 Module。

## 专家轨迹资格

| task | relation / geometry | support 专家成功 | query 专家成功 |
|---:|---|---:|---:|
| 0 | between / table_region | 3/3 | 3/3 |
| 1 | next_to / table_region | 3/3 | 3/3 |
| 3 | on_support / movable_object_surface | 2/3 | 3/3 |
| 4 | inside_container / open_container | 3/3 | 3/3 |
| 7 | on_support / fixture_surface | 2/3 | 3/3 |

task 3 和 task 7 的 support 各有一条示教动作在当前固定环境中未达到 success。它们没有被当作成功专家控制样本进入 phase 误差均值；这说明部分 HDF5 示教轨迹在当前 replay 边界下需要单独复核，不能把所有离线动作都当作同等有效的专家闭环标签。

## 分层结果

表中 `n` 是专家成功轨迹实际执行窗口中的有效步数；`joint` 是 free-run 与专家 joint-state 的 L2 差；`eye` 是 eye-in-hand RGB MAE。release 阶段样本少或为零，表示专家在 success 判定前已经结束，不能解释为训练数据完全没有 release。

| task | role | phase | n | action MAE | translation MAE | gripper MAE | joint L2 | eye MAE |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | support | approach | 110 | .1091 | .1662 | .1744 | .1205 | 17.26 |
| 0 | support | transport | 124 | .1236 | .1901 | .1368 | .5668 | 37.42 |
| 0 | query | approach | 109 | .0879 | .1596 | .0559 | .1313 | 13.52 |
| 0 | query | transport | 94 | .1421 | .2057 | .2488 | .4492 | 33.76 |
| 0 | query | release | 62 | .2318 | .2968 | .5770 | .4624 | 41.02 |
| 1 | support | approach | 159 | .0774 | .1419 | .0338 | .3039 | 15.41 |
| 1 | support | transport | 162 | .1405 | .2353 | .1292 | .5552 | 32.02 |
| 1 | query | approach | 169 | .1047 | .1619 | .1668 | .1985 | 17.17 |
| 1 | query | transport | 167 | .1631 | .2736 | .1207 | .6527 | 37.91 |
| 3 | support | approach | 84 | .0621 | .1045 | .0373 | .2212 | 8.77 |
| 3 | support | transport | 80 | .1058 | .1619 | .1749 | 1.0582 | 23.53 |
| 3 | support | release | 2 | .3544 | .2673 | 1.6180 | 1.5789 | 26.85 |
| 3 | query | approach | 125 | .0801 | .1380 | .0538 | .2064 | 11.36 |
| 3 | query | transport | 111 | .0972 | .1488 | .1315 | .9490 | 23.03 |
| 4 | support | approach | 149 | .0980 | .1862 | .0325 | .5066 | 32.26 |
| 4 | support | transport | 222 | .1214 | .2018 | .1114 | .9137 | 49.40 |
| 4 | query | approach | 171 | .1006 | .1804 | .0580 | .5379 | 36.09 |
| 4 | query | transport | 268 | .1080 | .1786 | .1053 | 1.3098 | 45.46 |
| 4 | query | release | 3 | .2336 | .1168 | 1.2543 | 2.2849 | 36.94 |
| 7 | support | approach | 58 | .0802 | .1394 | .0798 | .1758 | 13.89 |
| 7 | support | transport | 169 | .0933 | .1640 | .0678 | .6116 | 32.36 |
| 7 | query | approach | 120 | .0956 | .1637 | .1232 | .2386 | 18.08 |
| 7 | query | transport | 252 | .0925 | .1590 | .0580 | .8223 | 30.54 |

所有 30 条 Policy rollout 都未成功，均在 300 步 time limit 终止。该结果只能说明当前 500-update、单 seed baseline 在这些开放初态上仍未形成稳定闭环，不能作为正式 Policy 性能结果。

## 诊断判断

1. **没有单一的全局阶段根因。** approach 阶段已经有明显动作误差，transport 阶段 joint drift 通常显著扩大；release 阶段在有足够样本的 task 0、task 3、task 4 中夹爪误差明显增大，但 release 样本很少。
2. **open-container 条件最难。** task 4 的 approach/transport joint drift 和 eye-in-hand RGB MAE 在 support/query 都偏高，尤其 query transport joint L2 `1.3098`、eye MAE `45.46`。这支持“几何条件会放大状态反馈问题”的工作假设，但不是单 seed 的因果证明。
3. **query 阶段存在条件差异。** task 1 query transport translation MAE `0.2736` 高于 support `0.2353`；task 0 query release gripper MAE `0.5770`，说明同一关系/几何类别内部也有动作阶段分布变化。
4. **task 3/7 的 support 专家 replay 资格不稳定。** 这首先是数据—环境 replay 边界问题，需要单独复核；不能用这些失败轨迹支持“Policy 学不会”的结论。
5. **当前误差链仍符合早期动作误差到状态反馈放大的描述。** action MAE 在 approach 已非零，joint drift 在 transport 变大，eye-in-hand 误差同步升高；但本诊断没有独立干预视觉、动作 head 或控制器，不能判定最终根因。

## 下一步

先不启动 A/B/C。下一步对 task 3/task 7 的失败专家 replay 做固定环境、固定初态、固定动作序列的重复性审计，区分数据轨迹本身不成功、状态/动作时序问题和环境 replay 敏感性。随后在专家 replay 资格稳定的任务上，做一个预先声明的单变量阶段平衡或 gripper loss pilot；仍只使用 source support/query，保持 causal 对齐和相同 500-update budget。只有基础 Policy 在 source/query 上形成可重复闭环后，才进入 SPT 更新和 A/B/C。

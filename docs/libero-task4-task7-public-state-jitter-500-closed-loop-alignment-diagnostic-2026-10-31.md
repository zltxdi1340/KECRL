# LIBERO task 4/7 baseline 与 500-update jitter 闭环轨迹对齐诊断（2026-10-31）

## 范围与方法

本项只读取两份已经完成的公开轨迹 JSON：pinned 500-update Full Policy baseline reference 和 500-update public-state-jitter rollout。没有重新训练、没有重放环境、没有读取 qualification/validation/held-out/final evaluation，也没有读取 private simulator state。

配对键为 task、training seed 和 candidate nonregression initial state `0–19`。两份 artifact 共 `120` 对 episode，reset observation digest `120/120` 对齐。分析从 `stepwise_public_state_trace` 重建 action norm、首次 close、close 持续、首次 release、gripper sign switch、joint/gripper offset 和指定步点指标。

## Provenance

- baseline：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。
- jitter：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json`，SHA-256 `8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2`。
- 分析结果：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-alignment-20261031-v1.json`，SHA-256 `408d9071c808665c71e2578f248f247d474e36a8ff04f2ec7a19819bce1b6704`。
- 分析脚本：[analyze_libero_task4_task7_public_state_jitter_500_closed_loop_alignment.py](../experiments/analyze_libero_task4_task7_public_state_jitter_500_closed_loop_alignment.py)，SHA-256 `3f8f525f17906e247cb8923942f7d86becd407573318130010107faf89ceb9c8`。

## 总体对齐结果

| 指标 | baseline | 500-update jitter |
|---|---:|---:|
| 成功 | `7/120` | `6/120` |
| 有 close 的 episode | `88/120` | `77/120` |
| 首次 close 步数均值（有 close 样本） | `68.5` | `80.8` |
| close step 数均值 | `88.9` | `71.9` |
| 有 release 的 episode | `70/120` | `58/120` |
| gripper sign switch 均值 | `2.21` | `1.99` |
| 首动作 norm 均值 | `0.967` | `0.918` |
| 首动作 gripper 均值 | `-0.827` | `-0.740` |
| 全轨迹 action norm 均值 | `0.954` | `0.935` |
| 最大 joint offset L2 均值 | `2.413` | `2.202` |
| 最大 gripper offset L2 均值 | `0.02788` | `0.02780` |

paired success transition 为 baseline-only `4`、jitter-only `3`、both success `3`、neither success `110`。jitter-only 的 3 条轨迹中，jitter 首次 close 步数均值为 `38.0`，baseline reference 有 close 的两条均值为 `40.0`；baseline-only 的 4 条轨迹中，baseline 首次 close 均值为 `55.5`，jitter 有 close 的三条均值为 `63.0`，且一条 jitter 轨迹没有 close。样本很小，这些只作分层描述。

## 指定步点的 public state 与 action 对齐

下表是所有可达到该步点的 episode 均值；成功 episode 提前结束会从更晚步点的分母中排除。

| step | baseline joint offset | jitter joint offset | baseline gripper offset | jitter gripper offset | baseline action norm | jitter action norm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.00858` | `0.00856` | `0.000393` | `0.000405` | `0.967` | `0.918` |
| 5 | `0.07584` | `0.07719` | `0.01860` | `0.01880` | `1.119` | `1.123` |
| 10 | `0.20700` | `0.21115` | `0.02554` | `0.02558` | `1.225` | `1.231` |
| 20 | `0.59024` | `0.59920` | `0.02628` | `0.02631` | `1.260` | `1.254` |
| 50 | `1.19058` | `1.37236` | `0.02358` | `0.02431` | `0.923` | `0.932` |
| 100 | `1.29114` | `1.42675` | `0.02453` | `0.02450` | `0.924` | `0.928` |
| 200 | `1.87035` | `1.72047` | `0.02539` | `0.02536` | `0.915` | `0.895` |
| 300 | `2.27300` | `2.08748` | `0.02618` | `0.02571` | `0.967` | `0.924` |

早期 step 1–20 的 joint/gripper offset 很接近；jitter 在 step 50/100 的 joint offset 更大，step 200/300 又低于 baseline。gripper offset 在所有指定步点都接近，差异主要体现在动作时序和 joint trajectory，而不是公开 gripper 状态幅度。这个方向反转说明混合总体均值不能直接对应单一漂移机制。

## 解释边界

这组配对轨迹支持一个有限描述：500-update jitter 改变了首动作幅度、close/release 时序和中段 joint drift 的形状，但没有提高总体闭环成功率。它不能证明 jitter 导致失败，也不能把 joint drift、视觉、action head 或数据覆盖中的任一项认定为因果根因；candidate nonregression 初态不是 unseen，成功转移数量也不支持显著性或正式性能结论。

下一步如果继续，应只围绕 `jitter-only` 与 `baseline-only` 的 7 条转移轨迹做更细的 phase/首步对齐，保持 500-update checkpoint 和当前闭环协议不变。不要据此更新 SPT/Knowledge、注册 Module 或访问 qualification。

## 验证证据

- alignment analysis 专项测试：`9 passed`。
- 分析输出的 120 对 episode、reset digest、transition counts 和 step-point trace 均通过独立复核。
- `py_compile` 与 `git diff --check` 通过。

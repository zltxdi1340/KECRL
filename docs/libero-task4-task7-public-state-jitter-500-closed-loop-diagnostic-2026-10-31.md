# LIBERO task 4/7 public-state jitter 500-update 闭环诊断（2026-10-31）

## 目的与边界

本项固定 budget-sweep 中的 `500 updates` public-state-jitter checkpoint，检验它是否改变已完成的 Full Policy candidate nonregression 闭环诊断。baseline 使用已完成并 pinned 的 500-update Full Policy 闭环结果作 reference；本次只新增执行 jitter arm 的 120 个 episode，因此没有重新训练 baseline，也没有改变 checkpoint 规则。

评估使用 source `candidate_nonregression` 初态 `0–19`，这些初态已被此前诊断使用，不能称为 unseen。task 为 `4,7`，training seed 为 `20261022/20261023/20261024`，环境 seed `20261022`，stabilization `5`，每 episode 最多 `300` 步，deterministic policy action。没有访问 qualification、validation、held-out 或 final evaluation，没有读取 private simulator state，没有更新 SPT/Knowledge，也没有注册 Module。

## 配置与 provenance

- 协议：[libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic_v1.json](../configs/libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic_v1.json)，SHA-256 `9d7e3cb06cf8d97888f3fb8e9a1007f048c4210625133fa4dde129c3f2d9d1ac`。
- validator：[validate_libero_task4_task7_public_state_jitter_500_closed_loop_protocol.py](../experiments/validate_libero_task4_task7_public_state_jitter_500_closed_loop_protocol.py)。
- runner：[run_libero_task4_task7_public_state_jitter_500_closed_loop_diagnostic.py](../experiments/run_libero_task4_task7_public_state_jitter_500_closed_loop_diagnostic.py)，SHA-256 `87a789d5f3e43feab117beb721b410d18888a01b8a16bf0004c140e9be6630d9`。
- 结果：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json`，SHA-256 `8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2`。
- 运行设备：RTX 4090 `cuda:1`，Python `3.10.22`、PyTorch `2.4.1+cu124`；6 个 jitter model、120 episodes，elapsed `856.78s`。
- 500-update jitter checkpoint 来自 budget-sweep 结果 `3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110`；6 个 checkpoint file/weight hash 在运行时逐一核对。
- baseline reference：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。

120/120 个 jitter episode 的 reset observation digest 与 pinned baseline reference 对齐；paired keys 为 task、training seed 和 initial state index。

## 结果

| task | seed | pinned baseline | 500-update jitter |
|---|---:|---:|---:|
| 4 | 20261022 | `0/20` | `0/20` |
| 4 | 20261023 | `2/20` | `3/20` |
| 4 | 20261024 | `1/20` | `0/20` |
| 7 | 20261022 | `0/20` | `0/20` |
| 7 | 20261023 | `1/20` | `0/20` |
| 7 | 20261024 | `3/20` | `3/20` |
| **合计** |  | **`7/120 = 5.83%`** | **`6/120 = 5.00%`** |

paired outcome 为：baseline-only success `4`、jitter-only success `3`、both success `3`、both failure `110`。发生 success 状态转移的具体初态为：task 4 seed `20261023` state `9`（jitter-only），task 4 seed `20261024` state `17`（baseline-only），task 7 seed `20261023` state `3`（baseline-only），task 7 seed `20261024` states `4,5`（baseline-only）和 `6,18`（jitter-only）。114 个 jitter episode 以 `time_limit` 结束，6 个以 `success` 结束。

因此，在固定的 500-update budget 和这组已使用过的 candidate nonregression 初态上，jitter 没有提高总体闭环成功率，反而从 `7/120` 变为 `6/120`；task 4 seed `20261023` 的局部增加被 task 4 seed `20261024` 与 task 7 seed `20261023` 的减少抵消。这个结果不支持把 500-update public-state jitter 作为闭环修复。

## 解释边界与下一步

该结果只支持一个有限判断：500-update jitter 在当前 candidate nonregression 闭环诊断上没有总体控制收益。它不能证明 jitter 一定有害，也不能把差异归因于视觉、action head、状态漂移或数据覆盖中的单一机制；样本是诊断规模，且 baseline 是 pinned reference 而非本次重新执行的 baseline arm。

下一步应使用这次 jitter trace 与已有 baseline trace 做只读机制对齐：首个动作、首次 close、close 持续、release、joint/gripper offset 和 action norm，判断 3 个 jitter-only 与 4 个 baseline-only 转移是否有共同模式。不要因为离线 query MSE 或单个 seed 成功数选择 checkpoint，也不要直接进入 Module qualification。

## 验证证据

- 500-update 闭环协议专项测试：`7 passed`。
- 120 episodes、6 checkpoint hash、120 reset digest 和 paired summary 独立复核通过。
- `py_compile` 与 `git diff --check` 通过。

# LIBERO task 4/7 public-state jitter 预算对照诊断（2026-10-31）

## 目的与冻结边界

本次预算对照检验 500、1000、2000 optimizer updates 是否改变 public joint/gripper state jitter 的离线 source-demo 泛化表现。每个 task、seed 和 arm 只训练一次到 2000 updates，在 500/1000/2000 三个固定节点记录 support/query 曲线；曲线 checkpoint 只用于记录，不用于选择。

实验仍属于 LIBERO 机制诊断，不是正式 SPI 结果。训练使用 causal HDF5 `meta_support` demonstrations `0–29`，评价使用独立 `meta_query` demonstrations `30–49`；每个 task 的全部 baseline/jitter support fits 到 2000 updates 后，才创建 query dataset。没有 rollout，没有访问 candidate、qualification、validation、held-out 或 final evaluation，没有更新 SPT/Knowledge，也没有注册 Module。

## 配置与实际运行 provenance

- 冻结配置：[libero-task4-task7-public-state-jitter-augmentation-budget-sweep-diagnostic_v1.json](../configs/libero-task4-task7-public-state-jitter-augmentation-budget-sweep-diagnostic_v1.json)，SHA-256 `755f06f018687f1599fc9c2db1aff0f2f15a490d1ccd2d83abfe54f544ef40fb`。
- validator：[validate_libero_task4_task7_public_state_jitter_budget_sweep_protocol.py](../experiments/validate_libero_task4_task7_public_state_jitter_budget_sweep_protocol.py)，SHA-256 `516d8e15f69578e2ab2ad6471c043f9a55305b95b0f5aaedc0596527910657b4`。
- runner：[run_libero_task4_task7_public_state_jitter_budget_sweep_diagnostic.py](../experiments/run_libero_task4_task7_public_state_jitter_budget_sweep_diagnostic.py)，SHA-256 `a9247888709198be435886f9f2073886d65d72812d2081389ecddbf8a01e2969`。
- 授权变化：上一轮为 6 个 500-update jitter endpoint；本协议为 baseline/jitter 两臂各覆盖 task `4,7` 和三个 seed，共 12 条 endpoint trajectory，每条到 2000 updates，固定曲线点为 `500/1000/2000`。
- 成功结果：`/tmp/libero-task4-task7-public-state-jitter-budget-sweep-20261031-v3.json`，SHA-256 `3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110`。
- checkpoint 目录：`/tmp/libero-task4-task7-public-state-jitter-budget-sweep-checkpoints-20261031-v3`，36 个文件；manifest SHA-256 `835f6052049527cee4569a787af9928f763e41e3a1561885c0fb660f5ee3d067`。
- Runtime：Python `3.10.22`、PyTorch `2.4.1+cu124`、CUDA `12.4`、RTX 4090 `cuda:0`；总 optimizer updates `24,000`，runner elapsed `587.06s`，峰值 CUDA allocation `286,280,192` bytes。
- 500 parity：baseline 和 jitter 的 500-update weight hash、support metrics、query metrics 均与已固定的 source query/jitter 结果一致；两 arm 的 initial model hash 也一致。

第一次和第二次启动在 task 4 的 support fits 完成后、query 评价开始前因 runner batch-size 字段路径错误 fail-closed；没有写结果 JSON，partial checkpoint 未删除，分别保留在 `/tmp/libero-task4-task7-public-state-jitter-budget-sweep-checkpoints-20261031-v1` 和 `v2`。成功 v3 的 provenance JSON 中记录了这两次尝试及原因。

## Support/query 曲线

以下为三个训练 seed 的均值；结果 JSON 保存每个 task×seed×arm 的完整曲线和动作子空间指标。

### Task 4

| updates | baseline support MSE | jitter support MSE | baseline query MSE | jitter query MSE | jitter − baseline query |
|---:|---:|---:|---:|---:|---:|
| 500 | `0.034854` | `0.034144` | `0.038517` | `0.037290` | `-0.001227 ± 0.001296` |
| 1000 | `0.027441` | `0.027437` | `0.033177` | `0.033460` | `+0.000283 ± 0.001258` |
| 2000 | `0.018212` | `0.019927` | `0.032373` | `0.031328` | `-0.001045 ± 0.002946` |

### Task 7

| updates | baseline support MSE | jitter support MSE | baseline query MSE | jitter query MSE | jitter − baseline query |
|---:|---:|---:|---:|---:|---:|
| 500 | `0.038463` | `0.038341` | `0.042630` | `0.042862` | `+0.000231 ± 0.000659` |
| 1000 | `0.033575` | `0.033770` | `0.043234` | `0.043844` | `+0.000609 ± 0.001782` |
| 2000 | `0.023759` | `0.023892` | `0.043044` | `0.041146` | `-0.001898 ± 0.002500` |

六个 task×seed 组合合并的未加权 query MSE 为：500 updates baseline/jitter `0.040574/0.040076`，1000 updates `0.038205/0.038652`，2000 updates `0.037709/0.036237`。support MSE 随预算持续下降，但 query MSE 不保证单调下降；task 7 baseline 在 1000 和 2000 的均值没有优于 500，jitter 则在 2000 才出现两 task 同方向的均值改善。

## 解释边界

这组曲线支持三个有限判断：

1. 500→2000 updates 确实继续降低 support fit error，短训练时间不是因为 runner 提前停止。
2. query 泛化收益依赖 task、arm 和预算；1000 updates 没有稳定改善，2000 updates 的 jitter query 均值低于 baseline，但 task/seed 仍是小样本诊断规模。
3. 更长训练预算可能是当前离线拟合的限制因素之一，但 support 降低与 query 改善并不等价，且没有闭环 rollout 证据。

因此不把 2000-update jitter 结果写成闭环收益、视觉/状态漂移的因果修复、正式 SPI 结果或 Module qualification。当前曲线也没有用于 checkpoint 选择；若后续做闭环，必须另行冻结 rollout budget、初态和停止规则，并保持 qualification/held-out 数据隔离。

## 验证证据

- budget-sweep validator：`8 passed`。
- budget-sweep runner/协议专项测试：`8 passed`；原 public-state jitter 专项测试：`10 passed`。
- 结果结构、24,000 total updates、36 checkpoint 文件 hash、shared initialization 和 500 parity 独立复核通过。
- `py_compile` 与 `git diff --check` 通过。

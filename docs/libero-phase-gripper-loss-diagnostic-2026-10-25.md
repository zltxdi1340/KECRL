# LIBERO 阶段均衡与夹爪损失诊断（2026-10-25）

## 目的与边界

观测协议已经冻结：HDF5 causal observation 用作完整离线 BC 数据，当前 runtime 只用于闭环和对齐诊断。本次在该协议下检验两个动作损失假设：

1. `phase_balanced`：按 support 示教中 approach/transport/release 的步数做逆频率加权，让出现较少的 release 阶段在训练中获得更高权重；
2. `gripper_weighted`：保持阶段均衡不变，只把 gripper 这一维的 MSE 权重固定为 6，使其总贡献与 6 个机械臂动作维度同量级。

三臂均使用 task 3、task 7、seed `20261021`、causal pre-action HDF5 view、相同初始化、相同 DataLoader 顺序和 500 次 Adam update：`baseline`、`phase_balanced`、`gripper_weighted`。没有执行 Policy 闭环动作，没有访问 validation/held-out/qualification，没有选择 checkpoint，也没有更新 SPT/Knowledge 或注册 Module。

- 配置：[phase/gripper loss diagnostic](../configs/libero_phase_gripper_loss_diagnostic_v1.json)，SHA-256 `10d912f737cf9fd7ee47b1af42ef5a60ee8c1bb829d61694c3bc67e6874fb2c5`。
- 结果：`/tmp/libero-phase-gripper-loss-20261025-v1.json`，SHA-256 `1ea46e54ebff0bf7d101ec15e5204ddca44bf9da90f12679320e7a53efb694dd`。
- 运行时记录提交：`07fc50d67f1efb02798a7b115c2c9fbbfdcb6a01`；baseline task 3/7 最终 Policy SHA 分别为 `75c73689043b7930eb62d86caf866388b43a2d15e5e8262b67fadfad1840c9d5` 和 `beb3f8e88cb51f37d54a683001adee1b609a2a11be07b58f879804a917d346dc`，与既有 stratified diagnostic 一致。

## 数据和权重

support phase 步数与 phase-balanced 权重如下：

| task | approach | transport | release | approach 权重 | transport 权重 | release 权重 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1295 | 1304 | 495 | 0.7964 | 0.7909 | 2.0835 |
| 7 | 971 | 2648 | 599 | 1.4480 | 0.5310 | 2.3472 |

unknown phase 在这两个任务的 support/query 中均为 0。阶段标签沿用已审计的 gripper-action 规则：首次正 gripper action 之前为 approach，随后到首次负 gripper action 之前为 transport，之后为 release。

## Query 结果

| task | arm | action MAE | translation MAE | gripper MAE | gripper sign accuracy |
|---:|---|---:|---:|---:|---:|
| 3 | baseline | 0.11185 | 0.15760 | 0.21042 | 0.9198 |
| 3 | phase-balanced | 0.11256 | 0.15979 | 0.20518 | 0.9132 |
| 3 | gripper-weighted | 0.13529 | 0.21261 | 0.21701 | 0.9162 |
| 7 | baseline | 0.11870 | 0.18203 | 0.18090 | 0.9347 |
| 7 | phase-balanced | 0.11371 | 0.17323 | 0.16600 | 0.9405 |
| 7 | gripper-weighted | 0.12944 | 0.20717 | 0.17940 | 0.9378 |

按两任务 query 步数加权（共 4,851 步）：

- `baseline`：action MAE `0.11593`，translation MAE `0.17217`，gripper MAE `0.19282`，gripper sign accuracy `0.9287`；
- `phase_balanced`：action MAE `0.11324`，translation MAE `0.16781`，gripper MAE `0.18181`，gripper sign accuracy `0.9295`；
- `gripper_weighted`：action MAE `0.13180`，translation MAE `0.20936`，gripper MAE `0.19458`，gripper sign accuracy `0.9291`。

阶段分层中最明显的变化是：task 3 的 release gripper MAE 从 baseline `0.54170` 降到 phase-balanced `0.16193`；task 7 的 release gripper MAE 从 `0.57951` 降到 `0.46336`。但是 task 3 的 transport gripper MAE 从 `0.26780` 升到 `0.35120`，所以阶段均衡不是所有阶段都改善。

## 诊断结论

1. **阶段覆盖/信用分配是可疑因素，但证据仍是 targeted pilot。** 逆频率阶段权重在两个任务的加权汇总中小幅降低整体 action、translation 和 gripper MAE，收益主要来自 release；task 3 的总体 action MAE 仍略升，说明它不是稳定的全局修复。
2. **单独加大 gripper 维度损失没有解决问题。** 6 倍夹爪权重在两个任务都提高了整体 action 和 translation MAE，release gripper MAE 也没有稳定下降；当前不应继续加大该系数。
3. **当前结果不能把失败归因收敛到 loss。** 输入协议分叉、当前 runtime 成功边界、视觉/本体感知漂移和 Policy 表示误差仍然分别存在。本轮只比较离线 action fit，不能推出闭环成功率收益。
4. **phase-balanced 不能直接成为正式训练默认值。** 只有两个 source task、一个 seed、固定 500 update，尚未覆盖全部 source-SPI，也未访问验证任务；它只能作为下一轮同空间表示对照的候选损失臂。

## 下一步

保持 HDF5 causal 离线协议和同一 phase-balanced 规则，比较容量匹配的 CNN-only 与 CNN+GRU：CNN-only 去掉 recurrent hidden，CNN+GRU 保留现有连续 hidden，二者共享相同双视角 CNN、数据窗口、seed、update 预算和 phase loss。先用离线 phase 分层 action 指标判断记忆是否改善 transport/release，再决定是否进入当前 runtime 闭环读取。该对照仍是机制诊断，不是正式 SPT/SPI 或 Module 资格实验。

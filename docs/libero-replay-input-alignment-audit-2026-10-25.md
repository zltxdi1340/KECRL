# LIBERO 历史 HDF5 与当前 runtime 输入对齐审计（2026-10-25）

## 目的与边界

现有 causal BC 数据视图在窗口第 0 步使用当前环境重建的初始 observation，之后使用历史 HDF5 的 `obs[t-1]` 作为 `action[t]` 的输入。状态和语义审计已经显示，历史 HDF5 与当前 runtime 在首个动作后分叉。本次逐步量化两种输入视图的差异，判断分叉是否会持续改变 Policy 实际看到的 RGB、joint 和 gripper 输入。

本次只重放 source demo 的历史动作，不训练 Policy，不读取 validation/held-out/qualification，不选择 checkpoint，不更新 SPT/Knowledge，也不注册 Module。历史 HDF5 只读；当前 observation 只在内存中做差异统计。

- 配置：[replay input alignment audit](../configs/libero_replay_input_alignment_audit_v1.json)，SHA-256 `9eef0057fcf0e452cb6581fb99a5cdbadf924414cfca6a732c474236b8ef9e07`。
- 结果：`/tmp/libero-replay-input-alignment-20261025-v1.json`，SHA-256 `5c6d93ba8ff91cc80d1bf2b91594ca59d89a8721dbaba227157c50f4416e4730`。
- 运行提交：`1254ef8973f1a781d81718fc30dd46785725bdaf`；CUDA `cuda:1` / RTX 4090；耗时约 `18.90s`。
- 固定 seed `20261021`、`stabilization=0`；四条 case 与前两项 replay 审计相同。

## 对齐定义

对每个当前 runtime 动作前 step `t`：

- `t=0` 的历史视图使用当前 runtime 初始 observation，因为 causal dataset 的实现就是这样构造首个输入；
- `t>0` 的历史视图使用 HDF5 `obs[t-1]`；
- 当前视图使用同一个 HDF5 `init_state` 和动作前缀在当前环境中实际生成的 observation；
- action 序列不改变，只比较输入，不把私有 sim state 送入 Policy。

因此 step 0 必然相同，step 1 是第一个检验历史 HDF5 与当前 runtime 是否一致的位置。

## 结果

下表的 `mean MAE` 在当前 replay 实际执行的所有动作前 step 上统计，`max MAE` 是这些 step 中的最大值；step 0 的零差异包含在平均值中。

| case | 当前 replay | paired steps | agentview MAE 均值/最大 | eye-in-hand MAE 均值/最大 | joint L∞ 均值/最大 | gripper L∞ 均值/最大 |
|---|---|---:|---:|---:|---:|---:|
| task 3 demo 1 | 101 步失败 | 101 | 5.47 / 6.47 | 15.05 / 28.32 | .0191 / .0423 | .00251 / .00815 |
| task 3 demo 0 | 88 步成功 | 88 | 3.18 / 3.50 | 5.70 / 9.73 | .00829 / .01136 | .00053 / .00816 |
| task 7 demo 0 | 155 步失败 | 155 | 7.54 / 13.80 | 15.45 / 28.53 | .0724 / .1964 | .00112 / .00815 |
| task 7 demo 1 | 112 步成功 | 112 | 5.56 / 7.08 | 11.95 / 24.83 | .0360 / .0545 | .00516 / .03643 |

四条轨迹的 step 1 都是第一个非零差异位置。首步差异为：agentview MAE `2.78–4.07`、eye-in-hand MAE `6.75–7.61`、joint L∞ `.00267–.00438`、gripper L∞ `.00527–.00529`。随后差异会沿当前 runtime 的状态轨迹积累，task 7 demo 0 的 joint 偏差最大，task 3 demo 1 的 eye-in-hand 像素差异较大。

## 诊断结论

1. 现有 causal BC 训练视图在第一个动作后就不等于当前闭环中的实际 Policy 输入；这不是只发生在 replay 失败 demo，两个成功控制 demo 也出现同样的首步分叉。
2. 差异同时存在于 RGB 与本体感知，且在部分轨迹中 joint drift 随时间明显放大。不能把它简化为单一相机问题或单一 gripper 问题。
3. 这项审计没有测量 Policy 输出对输入替换的敏感性，因此不能直接说输入分叉已经造成了多少动作误差；它只证明后续 BC/闭环诊断若混用两种 observation 语义，会把环境运行时差异和 Policy 学习误差叠在一起。
4. 当前应将历史 HDF5 observation 和当前 regenerated observation 视为两个明确的数据协议：前者用于标准离线行为拟合，后者用于当前 runtime 闭环取证；在协议未选定前，不能把 replay eligibility 或 phase 误差均值作为正式 SPI 证据。

## 下一步

固定一组已有 source-SPI causal Policy 权重，只在这四条开放 source demo 上做输入替换敏感性：连续 hidden、动作、任务上下文和动作目标全部固定，仅把历史 HDF5 causal observation 换成当前 runtime regenerated pre-action observation，逐步测量 action MAE、translation/gripper 分量和首次 sign mismatch。该诊断用于判断输入语义分叉是否足以改变 Policy 输出，不进行 checkpoint 选择，也不进入正式训练或 Module qualification。

根据这项输出敏感性结果，再决定后续 BC 采用历史 HDF5 视图、当前 runtime 重建视图，还是把二者作为两个独立诊断协议报告。

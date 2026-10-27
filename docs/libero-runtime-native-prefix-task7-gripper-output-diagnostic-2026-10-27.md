# LIBERO task 7 夹爪输出分解诊断（2026-10-27）

## 目的与边界

上一项固定轨迹诊断发现：task 7 的 HDF5 causal Policy 在两个 candidate 状态中连续 300 步没有正向 close 命令，但其 source prefix 的 close 动作占比达到 71.8%。本诊断区分两种可能：

1. 模型在 support observation 上就没有学会 close；
2. 模型在 support 上能预测 close，但进入 candidate free-run 后因早期 RGB/本体状态偏移而延迟或丢失 close。

本诊断只重建 task 7、training seed `20261022` 的两个 500-update Policy，读取 30 条 `meta_support` prefix 作为离线教师状态，另读取已核验的 candidate trace 的两个状态 `0/8` 的前 80 步。没有执行新的环境动作，没有访问 qualification、validation 或 held-out，没有选择 checkpoint、更新 SPT/Knowledge 或注册 Module。

配置：[task 7 gripper output diagnostic](../configs/libero_runtime_native_prefix_task7_gripper_output_diagnostic_v1.json)，SHA-256 `26b6af4dd27abcf039a6f469d28e2f772edb32eaf53c4728356af036748ded3f`。结果位于实验主机 `/tmp/libero-runtime-native-prefix-task7-gripper-output-20261027-v2.json`，SHA-256 `dbb70c6a4cc15d7a660727058bfd8109366f07d9489456eaf80cc5efe00d7d20`。运行提交 `9b4e5fc`，Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时约 `21.13s`。

## Support observation 结果

两臂都在 support causal prefix 上学会了相同的夹爪阶段，差异很小：

| observation arm | close sign accuracy | 预测 close 占比 | 目标 close 占比 | 首次预测 close 中位数 | 目标首次 close 中位数 |
|---|---:|---:|---:|---:|---:|
| historical HDF5 causal | 0.9646 | 0.7237 | 0.7180 | 33 | 31 |
| runtime-native | 0.9638 | 0.7287 | 0.7180 | 32.5 | 31 |

分阶段统计按每条示教单独切分后再汇总，避免不同长度 prefix 拼接造成阶段边界错误：

| support 相对阶段 | HDF5 预测 close | runtime-native 预测 close | 目标 close |
|---|---:|---:|---:|
| 每条示教步 `0–19` | 0% | 0% | 0% |
| 每条示教步 `20–39` | 34.7% | 36.7% | 39.7% |
| 每条示教步 `40–79` | 97.8% | 98.1% | 97.5% |
| 每条示教步 `80+` | 97.0% | 97.2% | 93.6% |

因此，support 上不存在“模型完全不会 close”或“训练数据没有 close 阶段”的证据。两个 observation arm 的 support 行为几乎相同，也不支持把 candidate 差异直接归因于某一 observation source 的 gripper 拟合能力。

## Candidate free-run 对照

| observation arm | 初态 | 成功 | 前 20 步首次 close | 前 40 步首次 close | 前 80 步首次 close | 前 80 步 close 占比 |
|---|---:|---|---:|---:|---:|---:|
| historical HDF5 causal | 0 | 否 | 无 | 无 | 无 | 0% |
| historical HDF5 causal | 8 | 否 | 无 | 无 | 无 | 0% |
| runtime-native | 0 | 否 | 无 | 无 | 第 65 个动作 | 18.8% |
| runtime-native | 8 | 是 | 无 | 第 37 个动作 | 第 37 个动作 | 55.0% |

candidate trace 的模型 SHA 与本诊断重建出的对应 Policy 一致；轨迹来自前一项顺序复现诊断，成功标记、终止步数和模型来源已核验。这里的“第 65 个动作”对应 0-based trace index `64`，表中用人类计数表示。

## 诊断判断

这项结果排除了一个较窄的解释：task 7 HDF5 Policy 不是因为 support 训练中缺少 close 或完全没有学到夹爪阶段。更符合现有证据的描述是：模型在 candidate 的早期自由运行中进入了与 support 不同的观察/状态分布，HDF5 臂的 close 相位未能在 80 步内触发；runtime-native 臂在两个状态也都延迟 close，只是 state 8 的延迟轨迹最终完成任务，state 0 则未完成。

这仍不能区分 RGB 表示、目标几何可观测性、动作头条件化、控制误差或 early-state drift 的相对贡献。也不能把 runtime-native 的一个成功解释成一般性 observation-source 优势；support 指标几乎相同，candidate 样本也只来自 task 7 的一个训练 seed。

## 下一步

下一项固定这两个模型和同一 candidate trace，比较每个 candidate 初态的初始 RGB/本体观察与 30 条 support 起始窗口的 Policy 输入及 gripper/action 输出：先做不改变模型的 nearest-support 表征距离和前 20 步动作差异，再做一步 teacher-forced 与 free-run 的 action divergence 对照。目的在于判断 close 延迟是在初始观察就已经出现，还是由第一步之后的状态漂移累积造成。该分析仍然不访问 qualification，不选择 checkpoint，不修改 SPI/SPT 协议。

# LIBERO 前缀策略闭环失败轨迹诊断（2026-10-27）

## 目的与边界

完整 candidate 初态覆盖显示，基础 CNN-only 策略在 task 1、3、4、7 的成功率很低。本诊断从全覆盖结果中固定抽取共同失败状态和观测臂结果分歧状态，查看失败过程中的 Policy RGB、七维动作、公开关节/夹爪状态和终止原因，检查是否存在不执行、夹爪动作缺失或明显的任务条件差异。

协议：[failure trace diagnostic v2](../configs/libero_runtime_native_prefix_failure_trace_diagnostic_v2.json)，SHA-256 `b80f59d70c25a4cf7ba815a39b303d6110ceedc4d4b1f4c8851dccd6e1aca220`。来源全覆盖结果 SHA-256 `e155a74deb2038c5aca1c99d76623f754352626f0dad7c4fee5521dbd551a4e6`。轨迹 JSON 保存在实验主机 `/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json`，SHA-256 `4b453f994df9a5a59889b0cf3d94fd6ecc4a5d65756ccc235c7f485381631085`；224 张 RGB 图像保存在同机 `/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3-snapshots/`。

运行提交 `216131878b9090c75f3c7de34cc4167372a1f477`，Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时约 `1134.42s`。重建了 4 个 task-seed 组合、两种 observation arm 共 8 个模型，各模型 SHA 与全覆盖训练结果一致。为重现原 runner 的环境生命周期，每个模型实际按同一 adapter 顺序执行候选初态 `0..19`；只对 9 个预先固定的状态/arm 配对保存详细 trace，共 18 条。所有选中轨迹的成功标记、终止步数及 RGB 文件 SHA 均通过来源核验。

本诊断只使用 source `meta_support` 和 `candidate_nonregression` 初态 `0..19`。没有访问 `qualification`、validation 或 held-out，没有选择 checkpoint、更新 SPT/Knowledge 或注册 Module。详细轨迹不包含 private simulator state；状态文件只用于固定 reset。

## 固定样本

| task | 任务实例 | training seed | candidate 初态 | HDF5 causal | runtime-native |
|---:|---|---:|---:|---|---|
| 1 | bowl 在 ramekin 旁，移至 plate | 20261023 | 0、17 | 0、300 步失败 | 0 失败；17 成功/131 步 |
| 3 | bowl 在 cookie box 上，移至 plate | 20261023 | 0、1 | 两个均失败/300 步 | 0 成功/90 步；1 失败/300 步 |
| 4 | bowl 在木柜上层抽屉，移至 plate | 20261022 | 0、2、7 | 7 成功/156 步，其余失败 | 2 成功/159 步，其余失败 |
| 7 | bowl 在 stove 上，移至 plate | 20261022 | 0、8 | 两个均失败/300 步 | 8 成功/137 步，0 失败 |

这些是按共同失败和臂间分歧选出的诊断样本，不是随机样本或成功率估计。18 条中有 5 条成功；其余 13 条均以 300 步 `time_limit` 终止。

## 观察到的执行模式

1. **策略并非普遍不输出动作。** 18 条轨迹均有明显关节变化和持续运动命令；失败不能统一解释为环境接口未执行动作或 Policy 输出全零。仅观察公开关节量不能判断末端相对目标的三维位置。
2. **task 7 的 HDF5 臂没有发出 close。** 在 state 0 和 8 两条 HDF5 rollout 中，第七维 300 步始终没有正向 close 命令；RGB 画面中目标 bowl 在 time limit 时仍留在 stove 区域。对应 runtime-native 臂在 state 8 第 37 步开始 close 并成功；state 0 第 65 步开始 close，但仍失败，因此“发出 close”本身不是充分条件。
3. **task 7 的缺失 close 不是示教没有该动作。** 30 条 source `meta_support` runtime-native 前缀合计 3,844 个 action step，其中 2,760 步（71.8%）是正向 close；全部 30 条都包含 close，首次 close 的中位数为第 31 步。HDF5 Policy 在已拟合前缀上的 gripper sign accuracy 约为 96.5%，但候选闭环状态上的这两条轨迹没有 close。这是候选状态闭环条件化/泛化断裂的证据，不足以确定是 RGB 表示、状态覆盖还是 action head 所致。
4. **夹爪时间模式并非跨任务统一。** task 3 的 runtime-native state 0 在第 53 步开始 close 并成功；同一 arm 的 state 1 在第 52 步开始 close，但随后仅有 44 个 close step 且失败。task 1 的两臂 state 17 都在约第 56 步开始 close，只有 runtime-native 成功。task 4 state 7 中 HDF5 close 后较早重新 open 并成功，runtime-native 则连续 close 更久后失败；state 2 的结果又相反。因而单独提前 close 或增加 close 时长都不能解释全部成败。
5. **观测臂改变了闭环动作序列，但本样本不构成观测来源因果效应估计。** 配对状态上的两臂通常从第一步就输出不同动作；task 4 还出现 HDF5-only 和 runtime-only 成功各一例。每个 arm 是分别训练的模型，当前只覆盖固定 seed 和挑选后的诊断状态，不能把动作差异归因成一般性的 runtime-native 优势。

## 重放边界发现

前两次轨迹尝试在每个抽样初态上单独新建环境。task 4 / seed `20261022` / state 7 的 HDF5 结果因此从来源的成功/156 步变成失败/300 步，runner 按 fail-closed 规则停止写结果。初始 RGB 相同；随后确认全覆盖 runner 对每个模型复用同一个 adapter，按 `0..19` 顺序 reset/rollout。诊断 v2 复用相同生命周期后，所有固定选中状态均重现原成功标记和终止步数。

这说明当前 candidate 评估的完整初态顺序和 adapter 生命周期是复现协议的一部分；单独运行某个 state 不能被默认当作与批量评估等价。原因尚未由此定位，可能涉及 reset/模拟器运行状态，但本次没有读取 private simulator state 来判断。后续凡是跨 run 的初态对照，都应保留该顺序边界或先单独验证 reset 独立性。

## 诊断判断与下一步

当前最强的机制线索是 task 7 HDF5 Policy 在候选闭环中没有执行训练数据大量包含的 close，而 runtime-native Policy 在一个同初态任务上能执行 close 并成功。task 1、3、4 的结果则显示，close 起始时机、持续长度和成功之间没有一条可跨状态概括的简单规则；动作链还受到任务几何和闭环状态漂移影响。

下一项做 task 7、seed `20261022` 两个固定模型的 gripper 输出分解：在 source support 前缀的专家 observation 上统计逐时刻预测/目标一致性，再对照 candidate rollout 从 initial observation 到第一次 close 前的 action/proprioception/RGB 变化，检查 HDF5 臂的 close 命令是在起始条件下就缺失，还是随 free-run 状态偏移消失。仍只用公开 observation 和 source support / candidate 状态，不碰 qualification；结果只用于定位执行机制，不据此选模型或修改 SPT/SPI 协议。

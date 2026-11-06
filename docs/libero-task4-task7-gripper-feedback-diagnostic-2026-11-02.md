# LIBERO 固定缓存表示的夹爪反馈输入诊断（2026-11-02）

源专家观察上的 open 命令有偏迟线索，而强 tanh 饱和只占少数错误。本项检验固定 Policy 的两维公开夹爪状态是否影响该错误：保留原视觉与七维 joint，仅改变夹爪输入，不改权重或训练。三种读出均对同一 source target 评分；这是模型输入敏感性，不是物理干预或新闭环成功率。

## 在读出结果之前冻结

- [配置](../configs/libero-task4-task7-gripper-feedback-diagnostic_v1.json)，SHA-256 `9f2a33a16d905e533f09cbfaf2d5902363639ec59550b811122f7bf93eec8cdf`。
- 执行时脚本 SHA-256 `a21cbabcc2f5a445ab94b8b72edfd287a633e0d0306f52477dc7378b3f0a4c3d`；该版本的完整快照保留为 `/tmp/libero-gripper-feedback-frozen-runner-20261102-v1.py`。[仓库脚本](../experiments/analyze_libero_task4_task7_gripper_feedback.py)随后仅修复 JSON 布尔类型导出，详情见下文。
- [总预算 v3](../configs/libero-task4-task7-overnight-diagnostic-plan_20261102_v3.json)：本项之后收束晨间报告，不启动新的训练/闭环或系数搜索。

固定 task 4/7 × 三 seed 的六个 baseline500，H2 source-support `t>=1` 的视觉 256 维与公开 proprio 9 维缓存。只运行 `fusion → 逐帧 projection → head → tanh`，不调用 CNN 或完整 `forward_step`；不读取 HDF5、源初态、official state 文件或新图像。总 2,700 次主读出链、77,670 有效行，外加六次首批重复/192 行，总 2,706 次链 / 77,862 行。这些是实际新增的部分 Policy 计算，不能写成“没有任何前向计算”；新增 optimizer/backward/environment steps 均为 0。

## 事先固定的替换与判断

| variant | 两维 gripper 输入 | 角色 |
|---|---|---|
| original | 原逐步观测 | 精确复现基线 |
| hold_demo_t1_gripper | 同一 demo 首个可评分 t=1 的观察，整条 demo 保持不变 | 主变体 |
| zero_gripper | 两维恒为 0 | 相反锚点的次级对照，不能替换主变体 |

t=1 锚点是已观测的因果过去，不用未来帧或专家命令选输入。RGB 的手指外观仍存在，所以没有去除全部夹爪反馈信息。替换后视觉与本体可能不符合真实状态，任何收益也不直接等于训练或部署收益。

按原 source batch manifest（32 帧、不跨 demo）读出；original 必须逐元素匹配 H2 的原完整 Policy 预测，重复读出也须精确一致。缓存视觉、joint/proprio 和权重摘要不能变化，`.grad` 为空，不调用 optimizer。任一来源/角色/预算/数值/parity 条件不符即停止，旧输出拒绝覆盖。

主线索要求 open ±5 窗口夹爪 MSE 在三个 seed 都下降且符号误差不增加；另报告全步平移每 seed 比值<=1.05、seed 均值比值<=1.02、uniform7 均值比值<=1.02 的非回归。两锚点全部保留，不能只挑表现好的一项；即使通过也不自动训练，只保留为下一次单变量学习实验的线索。

原定输出为 `/tmp/libero-task4-task7-gripper-feedback-20261102-v1.json` 及对应 artifacts 目录两份预测/目标 NPZ。本项不更新 SPT/Knowledge、不注册 Module；qualification、validation、held-out、reserved、final evaluation 保持关闭。

## 实际执行、导出失败与零重算恢复

全部六个模型、三种输入已完成，共 2,700 次主读出链/77,670 行及 6 次重复/192 行；original 与 H2 完整 Policy 输出逐元素一致，原权重与输入未改变。CPU elapsed `1.8542s`，新增完整 Policy/CNN 前向、backward、optimizer 和环境步骤均为 0。

在完成两份预测 NPZ 后，最终 JSON 导出因 `numpy.bool_` 不能序列化而失败。**v1 JSON 不完整，不能作为完成结果**；原文件和错误日志均保留。原 JSON SHA-256 `2ea4a77cd9f66c41ffaacc64ee93ee6c60db07b0218ed0c6eabd232c2d639c22`，不能因其完整前缀中写有 complete 就当作有效结果。

仓库脚本只对 nonregression 的最终布尔判断显式使用 `bool`，不改数值、阈值或变体；新脚本 SHA-256 `06e3792cfb77ddf2168b4d6b23919911cb105f2e55f4b3bd02068e4797700325`。[正常运行配置 v2](../configs/libero-task4-task7-gripper-feedback-diagnostic_v2.json)保留修复后的可复现入口，但**本夜没有用它重跑模型**。

实际使用[缓存恢复配置](../configs/libero-task4-task7-gripper-feedback-serialization-recovery_v1.json)和[恢复脚本](../experiments/recover_libero_gripper_feedback_serialization.py)，先验证原代码快照、配置、完整前缀、六个模型完成日志和两份 NPZ，再重算指标并输出新的 v2 JSON。恢复配置 SHA-256 `7df11b131acfb017714b5f1a6f68390d32ae37eea7deace07fb65131ab2e5b29`；恢复脚本 SHA-256 `e2182e2cb5e7a5fb1ba7fae8d67147e1dbfdf9a8b5fcc64295ec97f5eb8c5d22`。恢复耗时 `0.1818s`，没有模型重算、环境或训练；原运行 provenance 与恢复 provenance 分开保留，不把修复后代码冒充原执行版本。

## 结果与决定

主变体为保持同 demo t1 夹爪状态。下表 delta 为替换减 original；MSE delta 小于 0 表示下降，sign error 用百分点表示。

| task | seed 后两位 | open 夹爪 MSE delta | open sign error delta |
|---|---:|---:|---:|
| 4 | 22 | -0.007338 | +0.3003 pp |
| 4 | 23 | -0.066159 | -1.5015 pp |
| 4 | 24 | -0.057270 | -1.2012 pp |
| 7 | 22 | -0.036718 | -0.5731 pp |
| 7 | 23 | -0.003902 | +1.1461 pp |
| 7 | 24 | -0.022913 | -3.1519 pp |

三 seed 的 open gripper MSE 均值在 task 4 从 `0.931611` 降到 `0.888022`（-4.68%），task 7 从 `1.010128` 降到 `0.988950`（-2.10%）。全时序平移 MSE 均值变化分别为 -0.26%/-0.71%，七维 MSE 为 -0.30%/+0.09%，非回归判断通过。**但每个任务各有一个 seed 的 open 符号错误上升，所以两个任务均未满足完整主线索条件。**连续误差变小，不等于命令决策统一改善。

次级 zero_gripper 在六个模型均增加 open gripper MSE，不能用它替换主变体或选择有利锚点。主变体的小幅 MSE 改善保留为反馈敏感性线索；混合输入可能不符合真实状态，RGB 仍有手指信息，不能声称去掉夹爪状态可提升控制。没有据此训练、部署或恢复已关闭 BCE 分支。

## 有效结果与独立复核

- **有效结果** `/tmp/libero-task4-task7-gripper-feedback-20261102-v2.json`，SHA-256 `437f065755ef47b13a2ab972ee060a46697ab9625e6285d5eb02a9dec678f53e`。
- 原计算的 task 4/7 readout NPZ SHA-256 分别为 `7c8be60356acd1cea9c5d4d098501f3fe19bd15525ba9e73cffe762eda67746e`、`aed9ce1b6921beaa8721164fd47c1b45fc35f22eb22cffd89249564237197d1a`；路径在有效结果 archives 中。
- 独立复核 `/tmp/libero-night-last-stages-independent-audit-20261102-v1.json`，SHA-256 `1b98445e7dd3445af32b3e2d7ac125068534cd6171c90999616f51ef7c2b00cc`：核对原失败文件/代码快照/恢复输入，重新计算 108 个反馈误差分层和全部非回归/停止判断，original 预测精确一致，重算指标差为 0，无新模型调用。

新增 JSON 原生布尔类型和缓存恢复的回归检查通过；最终 CPU 回归 `532 passed`。测试范围、可选 Crafter 限制及下一项范围见[晨间汇总](libero-overnight-diagnostic-progress-2026-11-02.md)。按总预算 v3，本项后停止扩展新候选，先提交完整证据供查看。

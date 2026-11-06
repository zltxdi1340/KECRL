# LIBERO 专家观察上的夹爪命令时序诊断（2026-11-02）

本项将既有 open 切换误差分解为命令先后偏移，避免直接用闭环的 close/release 次数解释物理抓取。固定 task 4/7、seeds `20261022/20261023/20261024` 的六个 baseline500，只读 H2 source-support `t>=1` 预测缓存；不构造模型、不读取 HDF5、不训练或运行环境。

## 执行前固定的规则

[配置](../configs/libero-task4-task7-expert-command-timing-diagnostic_v1.json) SHA-256 `bbbb094b21f81c8381a74444028a7586a7e9fbe74c4c815cd975e47b8ef147a3`；[脚本](../experiments/analyze_libero_task4_task7_expert_command_timing.py) SHA-256 `78ef0868f81962d135d9a50287a7d6c0e1944137bc1cab7ceb3a134bd1427957`。主输入及其特征/目标 archive 的 pin 来自[H2 报告](libero-task4-task7-condition-separability-diagnostic-2026-11-02.md)。

lag 固定为 `-5,-2,-1,0,1,2,5`，定义为 `prediction[t]` 对 `expert[t+lag]`；所有 lag 使用每 demo 相同的内部区间，排除首尾各五个缓存时刻。负 lag 的误差更低表示预测更接近过去命令，即偏迟线索，不是发现了新的数据对齐规则，也不允许把未来标签移入训练。

专家与预测的同方向符号切换在 ±10 步内按最近、不可重复使用匹配，平局取早者，按专家事件时间依次处理。offset=`predicted_step-expert_step`，正为迟。缓存第一帧不虚构 transition；窗口跨出可观察边界时标记 censored。未匹配只表示 ±10 内未找到，不等于全程没有命令；open/close 是命令符号，不是物理 release/grasp。

## 结果与限制

| task | seed 后两位 | 描述性最低 MSE lag | open 匹配/专家事件 | open offset 中位数 | open censored 事件 | close offset 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 22 | -1 | 30/31 | +1.5 | 12/31 | 0 |
| 4 | 23 | -2 | 27/31 | +5 | 12/31 | -1 |
| 4 | 24 | -2 | 27/31 | +5 | 12/31 | +1 |
| 7 | 22 | -2 | 25/33 | +5 | 16/33 | 0 |
| 7 | 23 | -2 | 27/33 | +3 | 16/33 | +2 |
| 7 | 24 | -2 | 29/33 | +4 | 16/33 | +1.5 |

六个模型的最小误差 lag 均为负；匹配到的 open offset 中位数均为正。仅保留完整事件窗口后，open offset 中位数仍为 task 4 `+2/+5/+3`、task 7 `+5/+3/+4`。因此，在专家观察上存在比 close 更一致的 open 命令偏迟线索。

限制包括：open 窗口边界删失比例较高；未匹配和额外预测切换均另存；偏迟还可能来自连续回归平滑、命令条件混叠或输入反馈，不能由此唯一归因于采样、tanh 或 memory。这里没有自由运行状态上的正确动作，不能把 offset 当作闭环失败的因果解释，也不调标签时移或 checkpoint。

## 成本与证据

实际计算 `168,630` 个 gripper lag 标量比较，elapsed `0.02755s`，新增模型、backward、optimizer、环境步骤均为 0。结果 `/tmp/libero-task4-task7-expert-command-timing-20261102-v1.json`，SHA-256 `fce047f957e50cd34aa89b917a680ebc7875208aea245b5176b9b4ffd043e90e`，保留逐事件匹配、未匹配、删失与全部 lag，最低误差只作描述。

独立复核 `/tmp/libero-night-last-stages-independent-audit-20261102-v1.json`，SHA-256 `1b98445e7dd3445af32b3e2d7ac125068534cd6171c90999616f51ef7c2b00cc`，从缓存目标/预测重建 387 条模型—专家事件记录（包括窗口内未匹配项）并重算全部 lag 比较，数值与事件列表一致，未执行模型或环境。

随后已完成[公开夹爪输入替换](libero-task4-task7-gripper-feedback-diagnostic-2026-11-02.md)，用于检查一项具体反馈线索；最终决定见[晨间报告](libero-overnight-diagnostic-progress-2026-11-02.md)。本项不属于 formal SPI、SPT/Knowledge 更新或 Module qualification，封存角色保持关闭。

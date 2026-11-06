# LIBERO task 4/7 500-update jitter success transition 细分（2026-10-31）

## 范围

本项只读取已经完成的 paired alignment artifact，分析 7 条发生 success transition 的 episode：jitter-only `3` 条、baseline-only `4` 条。没有训练、没有环境重放、没有新增 rollout，也没有访问 qualification/held-out/final evaluation。

- 输入 alignment：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-alignment-20261031-v1.json`。
- 输入 SHA-256：`408d9071c808665c71e2578f248f247d474e36a8ff04f2ec7a19819bce1b6704`。
- 结果：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-transition-mechanisms-20261031-v1.json`。
- 结果 SHA-256：`dd25d2a3bf9a392e6a9b2306e68f0e0f52ee704f172a13364e7dac2d1dc19211`。

## 逐条 transition

| transition | task/seed/state | baseline close→release | jitter close→release | 描述性模式 |
|---|---|---|---|---|
| jitter-only | 4 / 20261023 / 9 | `42→195`，time-limit | `39→None`，success 139 步 | jitter 在 release heuristic 前完成；close 段 `101` 步，对 baseline `153` 步 |
| jitter-only | 7 / 20261024 / 6 | `38→187`，time-limit | `38→None`，success 173 步 | close 时机相同，jitter 更早结束成功；close 段 `136` 对 `151` |
| jitter-only | 7 / 20261024 / 18 | 无 close，time-limit | `37→None`，success 173 步 | jitter 新增 close 命令并完成成功 |
| baseline-only | 4 / 20261024 / 17 | `96→None`，success 174 步 | `106→127`，time-limit | 两臂都有 close；jitter 形成短 close/release schedule，未成功 |
| baseline-only | 7 / 20261023 / 3 | `50→165`，success 205 步 | 无 close，time-limit | jitter 丢失 close |
| baseline-only | 7 / 20261024 / 4 | `38→115`，success 159 步 | `42→244`，time-limit | jitter 明显延迟 release，close 段 `221` 对 `121` |
| baseline-only | 7 / 20261024 / 5 | `38→155`，success 159 步 | `41→83`，time-limit | jitter 更早 release，close 段 `43` 对 `117` |

这里的 `release` 是公开 action sign 的 heuristic，不是 simulator 内部任务阶段标签；`None` 表示轨迹中没有出现该 heuristic。jitter-only 的两条 success 在 release heuristic 之前结束，因此不能把“没有 release 命令”写成失败机制。

## 首动作与 drift

jitter-only 三条的首动作 norm 相对 baseline 的变化分别为 task 4 state 9 `+0.0168`、task 7 state 6 `-0.0948`、task 7 state 18 `-0.0973`；baseline-only 四条分别为 task 4 state 17 `+0.0274`、task 7 state 3 `-0.2203`、task 7 state 4 `-0.1217`、task 7 state 5 `-0.1302`。task 7 的 baseline-only 转移都伴随较大的首动作变化，但样本只有 3 条，不能作因果解释。

transition 分组的 jitter−baseline joint offset L2 均值如下；分母受成功提前终止影响，只作窗口内描述：

| transition | step 1 | step 5 | step 10 | step 20 | step 50 | step 100 | step 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| jitter-only | `-0.000` | `-0.001` | `-0.006` | `-0.022` | `+0.006` | `-0.008` | 无完整共同窗口 |
| baseline-only | `+0.000` | `+0.002` | `-0.003` | `-0.041` | `+0.209` | `+0.404` | `-1.562` |

gripper offset 的对应变化在 step 1–20 接近 `0`，step 50/100 只有约 `0.0005/0.0047` 的 baseline-only 分组差异；因此这 7 条转移更明显地表现为 close/release 时序和 joint trajectory 差异，而不是 gripper state 幅度的统一变化。step 200 的 baseline-only 差异受少量可达样本和成功删失影响，不能解释为整体 drift 方向。

## 解释边界

当前 7 条 transition 可分为四种描述性模式：新增 close、close 后提前成功、丢失 close、以及 release/close 时序改变。它们没有形成一个跨 task/seed 的单一共同模式；因此不能据此认定 jitter 的收益或损失来自某一确定机制，也不能把首动作差异、joint drift 或 gripper timing 写成因果结论。

下一步应保持 500-update checkpoint 和当前闭环协议不变，若继续分析，只需对这 7 条轨迹做 RGB/action 对齐或首 20 步局部比较；不重新训练、不选择 checkpoint、不注册 Module、不更新 SPT/Knowledge。

## 验证证据

- transition analysis 专项测试：`2 passed`。
- 相关 alignment/协议回归集合：`61 passed`。
- 7 条 transition、首动作、close/release、指定步点 drift 和 provenance 独立复核通过。
- `py_compile` 与 `git diff --check` 通过。

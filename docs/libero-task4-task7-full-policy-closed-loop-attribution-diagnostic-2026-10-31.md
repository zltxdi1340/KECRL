# LIBERO task 4/7 Full Policy 闭环轨迹归因诊断（2026-10-31）

## 目的与边界

本项只分析已经完成的 task 4/7 Full Policy 闭环诊断的 120 条公开轨迹，检查成功与 `time_limit` 轨迹在首个动作、夹爪 close 时序、命令阶段、公开 joint/gripper offset 和 action norm 上的描述性差异。它不重新创建环境、不执行动作、不训练、不选择 checkpoint，也不读取 qualification、validation、held-out 或 final evaluation 数据。

输入初态是此前已经使用过的 `candidate_nonregression` `0..19`，不是 unseen test。轨迹中的 joint/gripper offset 是相对 post-stabilization reset 的公开状态偏移，不是 expert-state error。轨迹没有专家逐步 action，也没有私有目标坐标，因此本项不能判断首个动作相对专家是否已经偏离，不能把 command phase 当成真实环境的 approach/transport/release 阶段，也不能计算目标接近距离。

## 来源、脚本与验证

- 输入：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。
- 只读脚本：[experiments/analyze_libero_task4_task7_full_policy_closed_loop_diagnostic.py](../experiments/analyze_libero_task4_task7_full_policy_closed_loop_diagnostic.py)；脚本只使用标准库并对输入的 task、seed、初态角色、终止原因、逐步 trace 连续性和 qualification 边界做 fail-closed 校验。
- 分析 artifact：`/tmp/libero-task4-task7-full-policy-closed-loop-attribution-20261031-v2.json`，SHA-256 `25b2099aa38226d6a6634ecd976382b9d9e92f0c445cbebded05c4baa407a2a2`。
- 输出状态为 `complete`、`formal_result=false`、`read_only_analysis=true`；没有环境重放、训练、SPT/Knowledge 更新或 Module qualification。
- 新增分析测试与既有闭环协议测试共 `11 passed`（`conda run -n libero-kecrl python -m pytest -q tests/test_analyze_libero_task4_task7_full_policy_closed_loop_diagnostic.py tests/test_libero_task4_task7_full_policy_closed_loop_diagnostic.py`）。脚本 `py_compile` 通过。

### 分析口径

- close：逐步 `action[6] > 0`；release：首次 close 之后出现 `action[6] < 0`。
- `gripper_sign_switches`：相邻 `sign(action[6])` 的变化次数，零也作为一个符号。
- `action_norm`：七维 action 的 L2；`arm_action_norm`：前六个机械臂维度的 L2。报告中的 action norm 使用七维口径。
- `longest_contiguous_close_run` 是正向 close 的最长连续段；同时保留正向 close 总步数。
- `approach / transport / release` 是 Policy 命令阶段启发式：尚未 close、已经 close 且之后未 release、已经出现 close 后又出现 release。它们不是 simulator 或 expert phase 标签。
- 指定步点没有对应状态时记为缺失；成功 episode 通常早于第 300 步结束，不能用成功轨迹的第 300 步均值填充。

## 结果

### Task × seed 成功数

| task | seed `20261022` | seed `20261023` | seed `20261024` | 合计 |
|---:|---:|---:|---:|---:|
| 4 | 0/20 | 2/20 | 1/20 | 3/60 |
| 7 | 0/20 | 1/20 | 3/20 | 4/60 |
| 合计 | 0/40 | 3/40 | 4/40 | 7/120 |

训练 seed 间成功集中在 `20261023/20261024`；评估环境 seed 仍固定为 `20261022`。这个表是 seed 分层的描述，不是 seed 的因果比较。

### 成功与 time-limit 的总体差异

下表为均值/中位数；括号中的 `n` 是有该指标的 episode 数，`缺` 是缺失数。`first_close_step` 和 `first_release_step` 只在相应命令出现时有值。

| outcome | episode | 首次 close | close 总步数 | 最长连续 close | 首次 release | sign switches | 首个七维 action norm | 最大 joint offset | 最大 gripper offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| success | 7 | 55.0/40.0 (`n=7`) | 119.0/117.0 | 112.57/115.0 | 173.0/160.0 (`n=4`, 缺3) | 2.0/2.0 | 0.9353/0.8524 | 2.4169/2.3620 | 0.0270/0.0269 |
| time-limit | 113 | 69.67/43.0 (`n=81`, 缺32) | 87.04/46.0 | 82.35/35.0 | 126.80/114.5 (`n=66`, 缺47) | 2.22/2.0 | 0.9692/1.1190 | 2.4125/2.6827 | 0.0279/0.0279 |

7 条成功轨迹全部出现 close；113 条失败中 32 条在 300 步内没有出现 close，另有 81 条出现过 close。失败轨迹的 command-phase 启发式计数为：approach `32`、transport `15`、release `66`。因此失败并不只发生在“没有 close”的轨迹；已有 close 和 release 命令的轨迹仍可能运行到 time-limit。

### 指定步点的公开状态与 action norm

表中为均值/中位数；成功列的 `n` 在第 200 步只有 2，第 300 步为 0，因为成功轨迹已提前终止。time-limit 列在所有步点均有 113 条轨迹。

| 步 | success joint | time-limit joint | success gripper | time-limit gripper | success 七维 action | time-limit 七维 action |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0075/0.0061 | 0.0086/0.0089 | 0.0004/0.0004 | 0.0004/0.0004 | 0.9353/0.8524 | 0.9692/1.1190 |
| 5 | 0.0706/0.0562 | 0.0762/0.0740 | 0.0187/0.0187 | 0.0186/0.0187 | 1.1010/1.0158 | 1.1198/1.1454 |
| 10 | 0.1898/0.1169 | 0.2081/0.2116 | 0.0255/0.0255 | 0.0255/0.0255 | 1.2237/1.1381 | 1.2246/1.2267 |
| 20 | 0.5654/0.3955 | 0.5918/0.5964 | 0.0263/0.0262 | 0.0263/0.0262 | 1.2628/1.2237 | 1.2596/1.2682 |
| 50 | 1.1542/0.9183 | 1.1928/1.0456 | 0.0247/0.0249 | 0.0235/0.0255 | 0.9570/0.9743 | 0.9214/0.8955 |
| 100 | 1.1434/0.7070 | 1.3003/1.0912 | 0.0203/0.0224 | 0.0248/0.0268 | 1.0584/1.1487 | 0.9152/0.9545 |
| 200 | 2.4806/2.4806 (`n=2`) | 1.8596/1.9991 | 0.0242/0.0242 (`n=2`) | 0.0254/0.0270 | 0.8912/0.8912 (`n=2`) | 0.9150/0.9733 |
| 300 | 缺失 (`n=0`) | 2.2730/2.5450 | 缺失 (`n=0`) | 0.0262/0.0270 | 缺失 (`n=0`) | 0.9673/1.0134 |

第 1–20 步两类轨迹的公开 joint/gripper offset 与七维 action norm 接近；到第 100 步，失败组 joint offset 均值较高（1.3003 对 1.1434），但第 200 步成功组只剩 2 条，不能把该差异解释为稳定的控制因果关系。gripper offset 在各步点的均值差异较小，单凭这组公开状态不能确认 gripper drift 是主要瓶颈。

### Task 差异

| task / outcome | episode | 首次 close 均值/中位数（有值数） | 无 close 的 time-limit | close 总步数均值/中位数 | 最大 joint offset 均值/中位数 | 最大 gripper offset 均值/中位数 |
|---|---:|---:|---:|---:|---:|---:|
| 4 / success | 3 | 74.0/86.0 (`n=3`) | — | 113.0/89.0 | 2.5197/2.6212 | 0.0272/0.0274 |
| 4 / time-limit | 57 | 80.04/81.0 (`n=52`) | 5/57 | 93.32/103.0 | 3.0571/3.1554 | 0.0282/0.0283 |
| 7 / success | 4 | 40.75/38.0 (`n=4`) | — | 123.5/119.0 | 2.3398/2.3299 | 0.0269/0.0268 |
| 7 / time-limit | 56 | 51.07/39.0 (`n=29`) | 27/56 | 80.66/16.5 | 1.7564/1.0858 | 0.0277/0.0278 |

task 7 的失败轨迹中有近半数没有 close；task 4 的同类比例较低。task 4 失败组的 joint offset 较高，而 task 7 失败组的 joint offset 反而低于成功组，说明公开 joint drift 与失败的关系具有任务条件，不能归结为跨任务的单一状态漂移解释。两任务的 gripper offset 范围相近。

### 7 条成功轨迹的命令时序

| task | training seed | 初态 | steps | 首次 close | 首次 release | 末端命令阶段 | 最大 joint offset |
|---:|---:|---:|---:|---:|---:|---|---:|
| 4 | 20261023 | 7 | 128 | 40 | — | transport | 2.6212 |
| 4 | 20261023 | 8 | 259 | 86 | 257 | release | 2.8794 |
| 4 | 20261024 | 17 | 174 | 96 | — | transport | 2.0585 |
| 7 | 20261023 | 3 | 205 | 50 | 165 | release | 2.3620 |
| 7 | 20261024 | 4 | 159 | 38 | 115 | release | 2.2978 |
| 7 | 20261024 | 5 | 159 | 38 | 155 | release | 2.2512 |
| 7 | 20261024 | 19 | 176 | 37 | — | transport | 2.4482 |

成功轨迹既有在记录结束前出现 release 的样本，也有只观察到 transport 的样本；因此“必须在固定步数 close/release”不是由本 artifact 支持的统一规则。

## 解释边界

本项只给出固定 candidate 初态上的轨迹统计，不做显著性检验、checkpoint 选择或因果归因。首个 action 的 norm 和 gripper 值可以描述成功/失败轨迹的输出差异，但没有专家逐步 action，不能称为首步专家偏差。`approach/transport/release` 只依据 Policy 夹爪命令符号，不能说明末端是否真的接近目标、物体是否已抓取或释放。公开 offset 也不是相对于 expert rollout 的误差。

因此当前证据支持的范围是：离线 query MSE 的稳定排序尚未转化为可靠闭环控制；time-limit 轨迹同时包含无 close、close 后未 release、以及出现 release 仍未成功的模式；task 4 与 task 7 的 joint drift/夹爪时序模式不同；训练 seed 间成功集中，但不能据此选择 seed 或 checkpoint。当前不更新 SPT/Knowledge，不注册 Module，也不改变既有训练预算、loss、observation arm、环境 seed 或 checkpoint 规则。

## 下一步

先保持 Policy 和数据边界不变，使用固定 Policy 做短窗口 observation/state replacement 或 action-phase 对齐诊断：分别检查首步输出、首次 close 前后的 RGB/本体输入变化，以及已有 success/failure 状态的公开状态轨迹。任何新训练变量都要在机制区分后单独创建非 formal diagnostic 配置并重新冻结；在此之前不访问 qualification、validation、held-out 或 final evaluation。

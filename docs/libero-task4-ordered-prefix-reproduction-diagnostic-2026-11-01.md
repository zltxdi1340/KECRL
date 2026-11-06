# LIBERO task 4 按原重置顺序复现诊断（2026-11-01）

task 4 两对 RGB 补采窗口的源闭环前缀复现现已修复：恢复原重置顺序后，`56/56` 个窗口精确匹配，四个目标窗口全部通过门槛。用户指定训练保持 **500 updates**；复用现有 baseline/jitter 权重，新增 optimizer updates 为 **0**。本项是复现诊断，不是新的成功率评估或 formal result。

## 运行前证据与冻结协议

旧补采跳过前序状态，直接从 state 9 或 17 开始。四个 task 4 窗口的初始 joint/gripper 数值相同，但完整 reset observation digest 和动作前缀与源闭环不一致。只读核对另一份按 `0..19` 顺序采集的 baseline archive，发现 task 4 `60/60`、task 7 `60/60` 的 reset、action 和 pre-action 公共状态均与源闭环精确一致。该 archive 每集也只有 20 步，说明无需先假定必须重跑 300 步才能复现。

原生 LIBERO `bddl_base_domain.py` 在构造时初始化 `object_property_initializers`，每次 `_load_model → _setup_placement_initializer → _add_placement_initializer` 又向其追加抽屉属性采样器，未清空旧项。`_reset_internal` 随后遍历列表，`OpenCloseSampler.sample` 每项消耗 NumPy 随机数，然后才采样物体/fixture 的放置。task 4 的 BDDL 包含抽屉 `Open` 初态。代码因此提示：即使每次设置相同环境 seed，重置次数仍可能改变观察。运行中只记录该**配置列表的长度**，不读取 `sim.data`、`sim.model`、隐藏目标位姿或其他 private simulator state。

冻结配置：[libero-task4-ordered-prefix-reproduction-diagnostic_v1.json](../configs/libero-task4-ordered-prefix-reproduction-diagnostic_v1.json)。新 runner：[run_libero_task4_ordered_prefix_reproduction_diagnostic.py](../experiments/run_libero_task4_ordered_prefix_reproduction_diagnostic.py)。两者不修改历史配置、adapter 或第三方库。

| 项目 | 冻结值 |
|---|---|
| task / training seeds | task 4；`20261023, 20261024` |
| 两臂 | `source_baseline, public_state_jitter`，均为现有 `500 updates` checkpoint |
| 目标窗口 | seed 20261023 / state 9；seed 20261024 / state 17；共 4 个窗口 |
| adapter 生命周期 | 每 seed × arm 一个新 adapter，按 `0..9` 或 `0..17` 递增执行 |
| 每集 | 5 步 stabilization，20 个 deterministic Policy actions；native horizon 25 |
| 最大新增预算 | 56 episodes、1,120 Policy actions、280 stabilization actions；总显式环境 step 调用 1,400 |
| 训练 | 新增 0 updates，不选择 checkpoint、不改变 loss/观测/预算 |
| 设备 | RTX 4090 `cuda:0` 推理、EGL device 0；CPU torch threads 1 |
| 接纳门槛 | 每个窗口 reset digest、20 个 action、pre-action 公共状态及第 20 步 post-state 全部精确一致，不放宽容差 |

所有环境初态均为已经使用过的 source `candidate_nonregression`，最大索引 17。legacy `.pruned_init` 是单一 pickle 容器，需整体反序列化；runner 立即截取允许的 candidate 条目，再验证数值，未索引、检查数值或实例化 qualification/reserved 条目。文件整体 SHA-256 校验只用于完整性核对。未访问 validation、held-out 或 final evaluation。

只为通过门槛的四个目标窗口写新 NPZ；其余 52 个窗口只保留公开前缀核对和重置序号元数据。原来的失败复现窗口、结果和 checkpoint 均保留。不把 20 步前缀匹配解释成原 episode 最终 success 已重新评估，也不更新 SPT/Knowledge 或注册 Module。

## 结果

| training seed | arm | 按顺序执行并精确匹配的窗口 | 目标 state | 目标时采样器配置项数 |
|---|---|---:|---:|---:|
| 20261023 | baseline | `10/10` | 9 | 11 |
| 20261023 | jitter | `10/10` | 9 | 11 |
| 20261024 | baseline | `18/18` | 17 | 19 |
| 20261024 | jitter | `18/18` | 17 | 19 |

四个目标窗口的 reset observation digest、20 个 action、pre-action joint/gripper 和第 20 步 post-state 均与源闭环精确一致，全部 action/公开状态最大绝对差为 **0**；未放宽容差。两个 baseline 目标 archive 还与此前按 `0..19` 采集的对应 archive 在全部五个数组上逐元素相同，包括两路 RGB。

实际运行使用 **56 episodes、1,120 Policy actions、280 stabilization actions**，全部为已用 candidate 初态；耗时 **127.68s**。设备 RTX 4090 `cuda:0`、EGL device 0，CPU torch threads 1，torch peak allocated memory `13,489,664 bytes`。Python `3.10.22`、PyTorch `2.4.1+cu124`、CUDA `12.4`、NumPy `1.22.4`、MuJoCo `3.2.3`、robosuite `1.4.0`；环境及推理 RNG seed 均为 `20261022`。实际 cuDNN `benchmark=false, deterministic=false` 已记录，不能借用原训练的 flag 声明本次运行。运行前两张 GPU 均有其他作业；本进程使用 `nice -n 10`、单线程 CPU 和小模型推理，没有终止其他作业或启动训练。

独立只读复核确认：4 个新 archive 的文件/数组 hash、160 个采集时 RGB frame digest、两臂目标 reset 配对、config/runner/native code/checkpoint hash 均通过。四个修复窗口与此前 task 7 的十个有效窗口合起来形成 **14 个已核验前缀、7 对固定案例**；映射保存在 independent-audit 的 `verified_7_pair_archive_index`。这不是新增 14 个 episode，也没有把旧失败 archive 改成有效数据。

## 为什么能够修复

偏差来自补采没有保留 task 4 的环境重置历史。原顺序下，adapter 构造后抽屉属性采样器配置有 1 项，第一次 reset 后为 2 项，之后每 reset 增加 1 项；state 9 和 17 分别为 11、19 项。直接跳到目标状态会在第一次 reset 就使用目标初态，因此同一个 seed 和 initial-state index 并不保证相同的随机数消耗历史。原生代码中的属性采样器累积及后续放置采样路径与这次实测一致。

恢复顺序前，旧窗口与修复窗口首帧 joint/gripper 相同，但 agentview RGB MAE 为 `2.1277/2.1869`，eye-in-hand 为 `4.2931/4.0132`（分别为 state 9/17，像素尺度 `0..255`）；前 20 步 action 最大分量差为 `0.059194–0.099773`。保持相同 checkpoint 和 20 步 horizon、恢复顺序后，动作/公开状态差降至 0。因此，修复这次复现不需要增加训练 updates，也不需要先重跑每集 300 步。

本次修复的是**历史复现流程**，没有修改第三方采样器或共享 adapter。以后若要消除原生累积行为，必须另行版本化环境重置契约并重新建立基线，不能把这种环境语义变更静默混入已有结果。

原 300 步闭环未保存逐帧 RGB，故只能对原 reset RGB 摘要、动作与公开状态逐项核对，不能声称直接验证了原 episode 中间每一张 RGB。20 步匹配也没有重新验证最终 success，既有 baseline `7/120`、jitter `6/120` 的结果没有改变；不能据此认定这也是低成功率的原因。

## Provenance 与验证

- 配置 SHA-256：`0d25b289ea4828fbfe033aa8abd47a99caf432da60fd21907a7818d9a0685c33`。
- runner SHA-256：`972f4c5dba9522b23bc1a7d70d71bd4b036e1d5ad54470c9b52fb373100671e6`。
- 结果：`/tmp/libero-task4-ordered-prefix-reproduction-20261101-v1.json`，SHA-256 `d147a29896dcfd5cde0539add84fed8f3e880eb289fde1f7f0adfe71ec730d65`。
- 新 archive 目录：`/tmp/libero-task4-ordered-prefix-reproduction-archive-20261101-v1`；manifest SHA-256 `e891abf7673b6bf97f38c16627c0109597cecef2a9e684356814375d2719d771`。
- 独立复核：`/tmp/libero-task4-ordered-prefix-reproduction-independent-audit-20261101-v1.json`，SHA-256 `f401b24b8e1b37a1bc4563214d451f02fd59e338454ba7f2c6fa8e17876b677a`。
- 日志：`/tmp/libero-task4-ordered-prefix-reproduction-20261101-v1.log`，SHA-256 `805dd6c97d71625c5bb897ddfc76f7e8a743b00f1718895c9315d006c772c305`。
- 结果保存了 Git HEAD/工作区状态、18 个 pinned 输入、四个 checkpoint file/weight hash、原生代码 hash、GPU UUID/driver/占用、起止时间、seed、实际预算及各窗口检查。
- 新专项测试 `15 passed`；连同原始闭环、补采、证据分解和其他相关诊断共 **`100 passed`**。`py_compile` 与 `git diff --check` 通过。未运行需要可选 Crafter 依赖的全量 pytest，本轮未修改共享环境或训练代码。

运行命令：

```bash
nice -n 10 env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n libero-kecrl python \
  experiments/run_libero_task4_ordered_prefix_reproduction_diagnostic.py \
  > /tmp/libero-task4-ordered-prefix-reproduction-20261101-v1.log 2>&1
```

默认输出采用独占创建，重复运行必须显式选择新的 output/archive 路径，不能覆盖这次结果。未访问 qualification、validation、held-out 或 final evaluation，未更新 SPT/Knowledge，未注册 Module。

## 下一步

通过精确复现门槛后，用修复后的 task 4 两对窗口补齐此前逐相机、同观察跨 Policy 的离线分解，task 4/task 7 分开汇报。是否改变训练变量须由这些证据另行决定，训练预算继续保持 500 updates。

后续执行更新：上述离线分解已完成，task 4 两对同观察换 Policy 的 action L2 为 `0.045174–0.045448`，固定 Policy 换完整观察为 `0.019082–0.022118`；task 7 五对重算与旧结果精确一致。没有新增训练或交互。下一步固定三 seed 的现有两臂，对正确 causal 对齐的示教做有专家目标的分量/命令段误差诊断，再选择训练变量；详见 [修复前缀离线分解](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md)。

# LIBERO task 4/7 500-update 证据核对与离线分解（2026-11-01）

本轮完成一次证据核对与离线分解。保持既有 500-update Policy 权重，不训练、不新增 rollout、不读取环境初态或示教。结果支持先保留原 baseline，不能仅因 eye-in-hand 敏感性较大就采用新的视觉增强。

## 做了什么、为什么做

先核验新采集的 14 个 archive 是否复现原 baseline/jitter 闭环前缀，再对固定 Policy 和固定 observation 做四格交叉读出，并分别替换外部相机、眼在手相机、公开 joint/gripper 状态。这样可以分开检查“同观察下 Policy 映射不同”和“固定 Policy 对观察变化敏感”，避免由像素差直接推断动作根因。

新脚本只导入 Policy 实现，不导入环境 adapter、dataset loader、rollout runner 或 optimizer。全部推理在 CPU 单线程运行，elapsed `6.86s`，没有使用 GPU。

## 证据核对

| 检查 | 结果 |
|---|---:|
| 完整 archive 文件与内容结构检查 | `14/14` |
| 实际使用的 500-update checkpoint 文件/权重核验 | `8/8` |
| 两臂采集内初始 observation 数组相同 | `7/7` 对 |
| 与原闭环 reset/action/pre-action 公共状态前缀完全一致 | `10/14` 窗口 |
| 主要分析：task 7 原轨迹前缀已复现 | `5` 对 |
| 单独报告：task 4 未复现源前缀的新窗口 | `2` 对 |

task 4 两个案例仍是 seed `20261023` / state `9` 和 seed `20261024` / state `17`，两臂均存在源前缀不一致。初始 joint/gripper 数值和原轨迹相同，但完整 reset observation digest 不同；随后 action 分量最大绝对差为 `0.099773`。原结果没有保存当时的 RGB 帧，不能确定具体哪路相机或初始化细节造成偏差。adapter 的复用历史、状态顺序和环境 horizon 不同是尚待核验的协议差异，不是已证明的原因。

源前缀检查严格对应：archive action 索引 `0..19` 对原 public trace action steps `1..20`；archive pre-action joint/gripper 索引 `0..19` 对原 trace state steps `0..19`。逐数组完全匹配，不用浮点容差覆盖源前缀失败。缺失 digest、错位、重复 key、内容/hash 不符均拒绝；合法但源前缀不一致的窗口进入单独的 `quarantined_new_windows`，不混入主要分析。

另行记录 CPU 同权重/同输入前向与当时 CUDA action 的数值误差：14 个窗口最大分量绝对差为 `0.000135779`，CPU 重复前向全部完全相同。这个跨设备数值差没有被标为“精确复现”，也没有用于放宽源前缀检查。下列交叉读出各分支都在同一 CPU 上计算。560 个 RGB frame digest 由已保存 archive 离线派生，不冒充采集时记录的逐帧 digest。

## 主要结果：仅 task 7 已复现的 5 对窗口

这 5 对来自 seed `20261023` state `3` 和 seed `20261024` states `4,5,6,18`，按既有成功转移选出，不能代表全部 candidate 初态或新的成功率评价。

**首步观察相同，首动作已经不同。** 两臂首个 pre-action RGB/joint/gripper 数组完全相同；固定 Policy 替换另一臂观察时，首步 action delta 均为 `0`。两个 Policy 的首动作 L2 差均值则为 `0.201346`。因此，这组案例的首步动作分叉不需要后续观察漂移即可出现；这不等于证明首动作是失败原因，也不能把全 Policy 映射差异缩小成 action head 的贡献。

四格读出为 `baseline/jitter Policy × baseline/jitter observation`。以下都是七维 action delta 的逐步 L2 再对前 20 步、5 对案例取均值，不是 action MSE，也不是对专家的误差：

| 比较 | action delta L2 均值 |
|---|---:|
| 相同 baseline observation，替换整个 Policy | `0.100411` |
| 相同 jitter observation，替换整个 Policy | `0.111334` |
| 固定 baseline Policy，替换完整 observation | `0.026819` |
| 固定 jitter Policy，替换完整 observation | `0.037324` |
| 两臂各自 natural observation/action 的输出差 | `0.125647` |

同观察跨 Policy 差异大于固定 Policy 的完整观察替换差异，在上述 5 对中均成立。这是局部计算比较，不是模型权重与观察漂移对成功率的因果贡献率。

### 逐相机/公开状态替换

表内 baseline Policy 以 baseline observation 为参照；jitter Policy 以 jitter observation 为参照，逐项仅替换另一臂对应模态。反向参照也已计算并保存在结果 JSON。

| 只替换的输入 | 固定 baseline Policy | 固定 jitter Policy |
|---|---:|---:|
| agentview 外部相机 | `0.004231` | `0.005604` |
| eye-in-hand 眼在手相机 | `0.024036` | `0.027880` |
| 两路相机一起 | `0.024179` | `0.029938` |
| joint/gripper 公开状态 | `0.012164` | `0.012548` |
| 全部 observation | `0.026819` | `0.037324` |

在 5 对 × 2 个 Policy × 2 种参照方向的 20 个局部比较中，`eye-in-hand > joint/gripper > agentview` 的平均 action delta 排序均成立。这次排序来自分别替换输入的动作变化，不能与此前仅比较两路图像像素 MAE 的结论混为一谈。

同时保存相机间、RGB 与 proprioception 间，以及 Policy/观察之间的有符号 interaction。两种有符号四格分解路径都精确重建 natural 输出差，最大残差为 `0`；各项的 L2 范数不可直接相加，也不能当作贡献百分比。混合观察可能离开自然轨迹分布，仍不能据此确认真实环境因果机制。

## task 4 的边界

task 4 的 2 对新窗口计算结果单独保存在 JSON。它们不能凭源闭环的 success 标签被解释成原 transition 的完整复现，更不能与 task 7 五对混合成总体机制结论。前 20 步一致也不证明短窗口重现了原 episode 的最终 success。

后续修复更新：按新冻结协议恢复原重置顺序后，另采四个 task 4 目标窗口已全部精确复现；原生抽屉属性采样器配置随 reset 累积，直接跳到目标状态遗漏了原环境历史。详见 [task 4 复现诊断](libero-task4-ordered-prefix-reproduction-diagnostic-2026-11-01.md)。本报告和原 JSON 仍对应修复前 archive，不自动将原隔离组重新标为有效；下一步对修复后的 task 4 数据另做离线分解。训练继续保持 500 updates。

后续分析更新：修复数据的离线分解现已完成，结果另存，task 4 和 task 7 分开汇报，且 task 7 四格输出/指标精确复现本报告结果。本报告中的旧 task 4 隔离组仍保留原标签。新结果及下一项有专家目标的误差诊断计划见 [修复前缀离线分解](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md)。

## 本轮决定与下一步

- 保持原 500-update baseline；jitter 的既有闭环为 `6/120`、baseline 为 `7/120`，当前没有把 jitter 升为默认的证据。
- 不直接新增相机增强、gripper weighting 或预算变量。当前已经证明的只是同观察 Policy 差异与输入敏感性，不足以指定一种训练修复。
- **下一步先冻结 task 4 源前缀复现修复协议。** 核对完整初态顺序、adapter 复用/重置历史、环境 horizon、设备和 checkpoint provenance；将本轮严格检查作为输出门槛，单独核算新增交互预算后才补采。当前没有启动这组环境运行，也不扩采全部 120 对。
- task 4 复现问题解决后，再结合当前离线分解决定一个训练变量并冻结诊断。SPT A/B/C、Knowledge Evolution 和 Module qualification 仍需独立正式协议与资格契约。

## 配置、结果与验证

- 配置：[libero-task4-task7-500-update-evidence-decomposition-diagnostic_v1.json](../configs/libero-task4-task7-500-update-evidence-decomposition-diagnostic_v1.json)，SHA-256 `ee7e57efccc23017181eb8cf371024738932d72e95b2502d5b30b1e63540380b`。
- 脚本：[analyze_libero_task4_task7_evidence_decomposition.py](../experiments/analyze_libero_task4_task7_evidence_decomposition.py)，SHA-256 `9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4`。
- 结果：`/tmp/libero-task4-task7-500-update-evidence-decomposition-20261101-v1.json`，SHA-256 `31231b2d71653eae5320773987636b481f0b1f316b76dc7deb57133883eab415`。
- 输入 capture：`f647ce8642b03f93f92c72c68241038c952c6f07684df266c4b7f2423291e2a3`；原 baseline：`03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`；原 jitter：`8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2`。
- 新增专项测试 `14 passed`；相关 LIBERO 诊断回归 `85 passed`。`py_compile` 和 `git diff --check` 通过。没有执行含可选 Crafter 依赖的全量 pytest，本轮未修改共享训练或环境实现。
- 独立核验了输入/checkpoint hash、14 个 archive、560 个派生 frame digest、主/隔离分组、首步观察项为零和四格有符号恒等式。历史数据、checkpoint、失败尝试均未覆盖或删除。

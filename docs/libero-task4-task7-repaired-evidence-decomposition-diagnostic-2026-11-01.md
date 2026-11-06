# LIBERO task 4 修复前缀与 task 7 的离线分解（2026-11-01）

本轮已完成四个修复后的 task 4 archive 与十个有效 task 7 archive 的只读分析。14 个源前缀全部精确匹配，task 7 五对 CPU 分解与旧结果精确一致。task 4 也呈现“同观察跨 Policy 的动作差高于固定 Policy 替换完整观察”，但这还不是哪个动作更正确的证据。训练继续保持 **500 updates**；新增 optimizer updates 和环境 episodes 均为 **0**。

## 冻结配置与方法

配置：[libero-task4-task7-repaired-evidence-decomposition-diagnostic_v1.json](../configs/libero-task4-task7-repaired-evidence-decomposition-diagnostic_v1.json)。脚本：[analyze_libero_task4_task7_repaired_evidence_decomposition.py](../experiments/analyze_libero_task4_task7_repaired_evidence_decomposition.py)。复用 hash 固定的原离线分析数值方法，不修改历史配置、结果或失败 archive。

- task 4：seed `20261023` state `9`、seed `20261024` state `17`，两臂共四个修复窗口。
- task 7：seed `20261023` state `3`、seed `20261024` states `4,5,6,18`，两臂共十个已核验窗口。
- 所有窗口按既有 success transition 选定，属于 outcome-stratified 描述，不能代表全部 candidate 初态或作为 unseen 测试。
- 每个窗口 20 个 pre-action observation；action 对源 trace steps `1..20`，公共状态对 steps `0..19`。源 reset/action/公开状态必须精确匹配。task 4 另核验采集时逐帧摘要、逐步完整观察摘要及第 20 步 post-state。
- 固定 `baseline/jitter Policy × baseline/jitter observation` 四格读出，分别替换 agentview、eye-in-hand、joint/gripper、两路 RGB 及完整观察；两种参照方向均计算。
- 报告首步、前 `1/5/10/20` 步、七维动作的平移/旋转/gripper 分量和有符号 interaction。L2 范数不能直接相加作因果贡献百分比。
- CPU、torch 单线程；task 7 四格输出与数值分解要求同此前 CPU 分析精确一致。CPU 对采集时 CUDA 动作的数值差单列记录，不能用于放宽源前缀门槛。
- task 4/task 7 分开汇总，保留 task × seed 和 task × transition 分层。旧 task 4 失败窗口不纳入新结果。

脚本不导入环境、数据集、rollout runner 或 optimizer，不读取 `.pruned_init`、HDF5、qualification、validation、held-out 或 final evaluation。SPT/Knowledge/Module 边界不变。

## 结果：任务分别汇总

下表为每步七维 action delta 的 L2，先在前 20 步内取均值，再在各任务固定案例内取均值。它不是 action MSE、对专家的误差或成功率。首步指标只取首个 pre-action observation。

| 比较 | task 4：2 对 | task 7：5 对 |
|---|---:|---:|
| 首步观察完全相同，跨 Policy 首动作差 | `0.062507` | `0.201346` |
| 固定 baseline observation，换 Policy | `0.045448` | `0.100411` |
| 固定 jitter observation，换 Policy | `0.045174` | `0.111334` |
| 固定 baseline Policy，换完整 observation | `0.022118` | `0.026819` |
| 固定 jitter Policy，换完整 observation | `0.019082` | `0.037324` |
| 各自 natural observation 下两 Policy 输出差 | `0.053375` | `0.125647` |

task 4 的 2 对 × 2 个参照方向 × 2 个固定 Policy 共 `8/8` 个局部比较、task 7 对应 `20/20` 个比较，都满足同观察换 Policy 的均值动作差较大。首步两臂的两路 RGB 和 joint/gripper 完全相同，所有固定 Policy 输入替换的首步 action delta 为 0。因此，首步输出分叉已经存在于学到的 Policy 映射中，不需要后续观察漂移才能出现；不能把这个全 Policy 差异归为 action head 的独立因果贡献。

结果同时保存均值和中位数。task 4 只有两个案例，二者的中位数等于均值；task 7 同观察换 Policy 的前 20 步均值指标，中位数分别为 `0.091016/0.094563`。没有做显著性检验或成功率归因。

### task 4 两个案例的差异

| seed / state | 原结果分层 | 首步跨 Policy L2 | 同 baseline / jitter observation 换 Policy | 固定 baseline / jitter Policy 换完整 observation |
|---|---|---:|---:|---:|
| 20261023 / 9 | jitter-only success | `0.029216` | `0.019193 / 0.019481` | `0.011710 / 0.011593` |
| 20261024 / 17 | baseline-only success | `0.095798` | `0.071702 / 0.070868` | `0.032526 / 0.026570` |

两个案例的输出差幅度不同，且源成功转移方向相反。不能由“Policy 差异较大”推出 jitter 更好或更差，也不能将这里的源 success 标签当作本次重新评估的结果。前 `1/5/10/20` 步结果和逐步有符号向量均保存在 JSON。

### 逐相机与公开状态替换

baseline Policy 以 baseline observation 为参照，jitter Policy 以 jitter observation 为参照；每次只替换表中指定的另一臂输入。反向参照也计算并保存。

| 只替换的输入 | task 4 baseline Policy | task 4 jitter Policy | task 7 baseline Policy | task 7 jitter Policy |
|---|---:|---:|---:|---:|
| agentview | `0.002642` | `0.001824` | `0.004231` | `0.005604` |
| eye-in-hand | `0.021282` | `0.018569` | `0.024036` | `0.027880` |
| joint/gripper | `0.002733` | `0.002470` | `0.012164` | `0.012548` |
| 两路 RGB | `0.021364` | `0.018448` | `0.024179` | `0.029938` |
| 完整 observation | `0.022118` | `0.019082` | `0.026819` | `0.037324` |

eye-in-hand 在 task 4 的 `8/8`、task 7 的 `20/20` 个案例 × Policy × 参照方向中均为这三类输入里局部敏感性最高的输入。但 task 4 的次序并不统一：state 9 为 `eye-in-hand > proprio > agentview`，state 17 为 `eye-in-hand > agentview > proprio`；task 7 全部为前一排序。不能把 task 7 的完整排序直接推广到 task 4。

这是在两条自然轨迹的观察之间做输入替换；混合观察可能不在自然分布上，动作变化大也不等于原视觉输入错误，不能单凭该结果决定增加相机增强。

### 动作分量

在相同 baseline observation 下换 Policy，前 20 步的分量差均值为：

| 分量 | task 4 | task 7 |
|---|---:|---:|
| translation L2 | `0.044138` | `0.078588` |
| rotation L2 | `0.010112` | `0.009666` |
| gripper action 绝对差 | `0.000326` | `0.041169` |

task 4 此窗口的夹爪输出差很小，task 7 较大，因此也没有统一采用 gripper weighting 的依据。以上是 Policy 间的输出差，尚未与专家目标动作比较；各分量范数不能直接相加当作贡献率。保存的两条有符号四格分解路径精确重建 natural 输出差，最大残差为 0。

## 本轮决定与下一步

继续使用原 500-update baseline，jitter 不升为默认；既有 baseline `7/120`、jitter `6/120` 不因这次只读分析而改变。当前已经分开测量了同观察 Policy 映射差和固定 Policy 的局部观察敏感性，但还不能判断哪一项输出更接近任务所需动作。现在直接选择 phase loss、gripper weighting 或视觉增强，仍缺少动作正确性的证据。

**下一步冻结并执行一次有专家目标的动作误差分解，仍不训练、不新增环境交互：**

1. 固定 task 4/7 × 三个既有训练 seed × baseline/jitter 的全部 12 个 `500 updates` checkpoint，不能只用 outcome-stratified 七对案例判断训练修改。
2. 使用 source `meta_support 0..29` 与 `meta_query 30..49` 的已有示教，分别比较同一 causal pre-action observation 下两个 Policy 对专家动作的误差。按 translation、rotation、gripper，以及专家 close/open 命令段报告，命令段不冒充真实物理阶段。
3. 纯离线初版限定 `t >= 1`：HDF5 `obs[t-1] → expert action[t]`。现有 `_capture_role_dataset` 会为了首帧构造环境，不能在零交互分析中直接调用；缺失的首帧不补造。必须显式冻结排除 `t=0` 的覆盖范围，不把它称为原来包含首帧的完整 query MSE 重算。
4. 先看哪个动作分量/命令段存在跨 seed 一致的目标误差，再决定一个训练变量并冻结 500-update 配对诊断。若没有一致依据，不从输入敏感性大小猜测训练修复。

上述下一项尚未执行，本轮没有读取 HDF5 或示教初态。继续不访问 qualification、validation、held-out 或 final evaluation，不选择 checkpoint、不更新 SPT/Knowledge、不注册 Module。

后续执行更新：上述固定三 seed 两臂的专家目标误差对照现已完成，结果另存；14,490 个 `t>=1` 输入、86,940 行预测、零新增训练/环境交互。两任务 query 平移误差均有三 seed 的 jitter 改善，但夹爪收益不一致；24/24 个 task×role×seed×arm 条件的夹爪误差在 open 切换窗口最高。下一训练候选已记录为该窗口局部 2× gripper MSE、其余设置与 500 updates 不变，尚未执行。结果及完整 provenance 见 [专家目标动作误差诊断](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)。本报告的输入敏感性结果未改写为目标误差或成功率结论。

## Provenance 与验证

- 配置 SHA-256：`f19b6b2ebfac9ac007616cac9027f0c92f3be0a39eb11e6d9f31b9898b2db5d2`。
- 新脚本 SHA-256：`4d16c105c824cd95e0521178a03cf981232a72525ac8492d0ca030bc501ad1de`；原数值分析 core 保持 `9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4`。
- 结果：`/tmp/libero-task4-task7-repaired-evidence-decomposition-20261101-v1.json`，SHA-256 `6d8e54e34921d0e3b606ea58c57787f260d830656fe0f028928ecb6a609912c3`。
- 独立只读复核：`/tmp/libero-task4-task7-repaired-evidence-decomposition-independent-audit-20261101-v1.json`，SHA-256 `41f239c5193044e210702b9d936545af969ff4cd2cca710bd08b4cedc28d6320`。
- 日志：`/tmp/libero-task4-task7-repaired-evidence-decomposition-20261101-v1.log`，SHA-256 `ccdf103ddec8b633a5075f4347528a7c4ed538d9b4d7059d420502ec528eba37`。
- 输入固定 13 个文件 hash；14 个 archive、8 个 checkpoint file/weight hash 均核验，并保存 560 个 RGB frame digest。task 4 的 160 个为采集时记录并重新核对，task 7 的 400 个为离线派生，来源不混写。
- CPU torch 单线程，Python `3.10.22`、PyTorch `2.4.1+cu124`、NumPy `1.22.4`，analysis RNG seed `20261022`，elapsed **`6.59s`**，未使用 GPU。与原 CUDA 采集 action 的最大数值差 `0.000135779` 单列报告，未用于放宽源前缀门槛。
- 新专项测试 **`15 passed`**；相关复现、补采、闭环与分析回归共 **`115 passed`**。`py_compile`、环境检查和 `git diff --check` 通过。未运行依赖可选 Crafter 的全量 pytest，本轮未修改共享训练或环境实现。
- 独立从保存的四格输出重算动作/分量差、均值/中位数、有符号路径恒等式、首步输入替换为零、task 7 旧结果一致性和全部输入 hash。旧 task 4 失败窗口及历史分析输出均保持不变。

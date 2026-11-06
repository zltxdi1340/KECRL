# LIBERO合法专家输入的相机干预—目标动作误差协议（2026-11-02）

**研究设计保持冻结，目标误差诊断现已通过独立绑定执行完成。** 原设计冻结阶段只完成来源、算子、测试和元数据校验；后续7200主前向+48重复的实际结果、成本与独立审计见[执行报告](libero-task4-task7-expert-camera-target-error-diagnostic-2026-11-02.md)。原JSON中的pending和未执行字段是冻结时历史快照，未改写；执行绑定和结果另存。本项仍是非formal diagnostic，不是资格协议。

承接[source缓存输入分解](libero-task4-task7-source-stabilization-input-decomposition-diagnostic-2026-11-02.md)：眼在手响应较大，但无专家目标，不能称动作更错。本项加入原始causal专家目标，区分保空间坐标的外观变化下误差放大，与空间细节损失下的误差变化。空间算子同时损失纹理，不能单独证明几何必要性或闭环根因。

## 既有证据与本次新增问题

- [早期专家状态synthetic设置](../configs/libero_spatial_eye_hand_synthetic_sensitivity_diagnostic_v1.json)为task0的replay/GRU及baseline/增强比较，主要测action delta；不等于当前task4/7三seed的有目标对照。
- [跨任务相机平移敏感性](libero-runtime-native-prefix-cross-task-eye-hand-translation-sensitivity-diagnostic-2026-10-28.md)使用候选前20步和±1/2/4像素平移，只测输出变化；本项不重复其半径或方向搜索。
- [4像素增强负结果](libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic-2026-10-28.md)说明降低敏感性不保证拟合或控制收益，原停止决定保持，不自动重启增强训练。
- [当前专家动作误差](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)提供合法索引、目标、原预测和输入批次摘要，但未执行下面的匹配外观/空间干预及误差方向分解。本项沿用这些已审计接口，不创建新的数据角色。

## 固定模型与合法输入

task4/7、训练seed20261022/23/24，六个现有source_baseline、CNN-only、**500-update终点**；不加载jitter或1000/2000点，不选checkpoint。现行闭环默认**稳定化5步**不变。

仅source `meta_support` demos0..29，全部合法`t>=1`输入：`HDF5 obs[t-1] → 原expert action[t]`。task4为4442行，task7为4188行，共8630行；排除60个t0。未来只按现有角色授权器读取这两个任务的支持组，不读取source query或封存状态。

**离线对照不插入五步稳定化。** 本项没有reset，额外五步会改变输入—目标语义；5步只是沿用的闭环默认。禁止把上一轮0/5缓存、相机混合观察或新稳定化观察配旧action[0]。原action0只用于专家命令切换mask，不评分、不重建首帧。

干预发生在原参考状态的观测通道，原动作仅作为该状态的专家参考。干预图像不被声明为新的自然专家示教，不产生新的真实状态标签或训练数据。原输入、目标和本体数组保持不变；新观察不会送入环境。

## 八个条件与像素量匹配

每个原专家输入同时形成以下条件，两个相机均报告，不挑有利相机或seed：

| 条件 | 操作 | 作用 |
|---|---|---|
| identity_uint8 | 原始uint8观察 | 与原支持集预测精确复现 |
| identity_float | 两RGB按原encoder方式归一化，无其他修改 | 排除输入dtype/归一化路径差异 |
| eye_in_hand_photo_matched | 仅眼在手做弱对比度压缩 | 保留像素坐标的外观控制 |
| eye_in_hand_spatial_matched | 仅眼在手做匹配量的局部低通混合 | 空间细节扰动 |
| agentview_photo_matched | 外部相机的同类外观控制 | 相机参照 |
| agentview_spatial_matched | 外部相机的同类空间控制 | 相机参照 |
| eye_in_hand_spatial_full | 眼在手完全替换为8×8块均值图 | 未匹配强度的次级信息损失probe |
| agentview_spatial_full | 外部相机完全替换为8×8块均值图 | 同类次级probe |

确定性算法，不扫描强度、方向或ROI：

1. 原uint8经`torch.to(float32).div(255)`得到U，与现有encoder归一化一致。随后用float64计算变换，最后转float32给Policy，不重新量化为uint8、不裁剪。
2. 外观基图C是每图、每通道的空间均值广播；`U+alpha*(C-U)`保持坐标和通道均值，均匀减弱对比度。空间基图B是固定非重叠8×8块均值重复回原尺寸，保留粗尺度结构、削弱细节和纹理，不裁剪或平移。
3. 对同一专家帧计算两个相机各自的`d_photo=mean(abs(C-U))`及`d_spatial=mean(abs(B-U))`。共同预算`m=0.1*min(四个d)`只由原像素决定，不看目标或预测。
4. 四个matched条件各用`alpha=m/d`，d为0时alpha为0；所有alpha在[0,0.1]。两个相机、两类matched干预每帧的归一化像素MAE均为m，实际float32误差须<=1e-7。m为0的帧不排除，保留为恒等控制并记录有效扰动数。
5. full条件直接用B，单独报告强度和结果，不能把它与matched条件当等强度比较。

逐帧保留共同m、四个基图距离、alpha、实际MAE及active标记。相同像素MAE并不意味着相同语义破坏；完整块均值也可能造成分布外输入。故本项只提供操作性线索，不把空间损失效应直接叫作几何因果必要性。

## 目标误差与判断规则

主指标为全support、逐步加权的**平移MSE配对变化**，按task×seed×condition分开。七维、旋转和夹爪为次级，均按3/3/1/7维分别归一化；保留每demo均值/中位数及等demo权重汇总，不能事后切换主统计口径。

令`e=clean_prediction-target`，`delta=changed_prediction-clean_prediction`，对动作分量D：

`DeltaMSE_D = 2*mean(e_D*delta_D) + mean(delta_D²)`。

同时报告原MSE、干预MSE、误差变化、方向交叉项和响应能量。相同响应能量可以纠正原误差，也可放大误差；不能仅由action delta较大推出错误。逐步与汇总恒等式残差须<=1e-12，七维仍满足`(3*T+3*R+G)/7`。原float32预测/目标先转float64再计算。

分层沿用all、t1、expert open/close/neutral、open/close切换±5步；它们是专家命令标签，不是物理approach/transport/release。空分层/零分母为null并记录分母，不填0或无穷大。报告相机间的同帧matched误差差与逐行误差方向分解；不做显著性、种群推广或样本独立性声明。

每任务、每相机独立记录两个可同时成立的描述flag：

- **外观误差放大线索**：photo_matched的平移DeltaMSE在三个seed都严格>0。
- **空间细节依赖线索**：spatial_matched减photo_matched的平移DeltaMSE在三个seed都>0，且spatial_full的平移DeltaMSE也在三个seed都>0。

七维按同样规则另报，不能覆盖平移主指标。零、缺失或非有限值不能满足严格正向条件；单seed或命令窗口结果只是局部线索。两个flag并不互斥，未满足也不等于排除视觉机制。任何方向都不自动触发增强、改loss、重训或rollout，更不能声称完整KECRL验证。

## 未来执行预算与门槛

以下是**未执行的固定预算**，不是本轮实际计算量。

| 项目 | 固定值 |
|---|---:|
| 支持示教数／t0排除数 | 60／60 |
| source action数值行（含仅供mask的t0） | 8690 |
| 原专家观察／评分目标行 | 8630／8630 |
| 共享批次，按demo独立batch32 | task4 154；task7 146 |
| checkpoint加载 | 6 |
| 主前向／预测行：300批×3seed×8条件 | 7200／207120 |
| 重复：每task×seed首批32行的全部八条件 | 48／1536 |
| 原批预测对旧archive／uint8-float恒等检查 | 900／900，不另增前向 |
| 新optimizer／backward／环境／初态读取 | 全部0 |

CPU单线程、分析seed20261022，CUDA隐藏、OMP/BLAS/MKL=1。全部source角色授权先于HDF5打开；复用已有`authorize_role/read_causal_batch/command_masks/error_metrics`，不访问init/private属性。原target、索引、mask及300个输入批次摘要必须精确匹配旧support记录；原uint8预测须逐批精确复现，float恒等和八条件重复也须精确一致。权重、输入、目标hash及grad不变。

任何来源、角色、精确复现、MAE匹配、恒等式、有限性、范围或预算不符即停止，保留事件、部分预测和failed结果。1e-7与1e-12只用于新算子的像素/代数舍入检查，不放宽原数据/预测精确门槛，不追加预算或选择其他checkpoint。

## 冻结时的证据与验证

- [设计配置](../configs/libero-task4-task7-expert-camera-target-error-diagnostic_v1.json)，SHA-256 `67654a94e8e41497fa24b0929849f2d9535d9b5ae518bd56c39a863182acfdb4`，UTC冻结时间`2026-11-02T09:19:04.909753+00:00`。
- [纯算子与协议校验入口](../experiments/validate_libero_task4_task7_expert_camera_target_error_protocol.py)，SHA-256 `681a3390bc894e753e945fa0bdab6b79b5dd4ee2a60cc2fccb9e18fb10b7d834`。
- 校验记录`/tmp/libero-expert-camera-target-error-protocol-validation-20261102-v1.json`，SHA-256 `d171d5812925bcff24a2dd8fc65f78185df84b95c8cb52e33fd9088f8230036c`。这是`passed_protocol_validation`，明确`research_run_executed=false`，不是新的MSE结果。
- 16个既有输入pin、六个checkpoint文件hash、两个support archive文件hash及300条批次元数据已核对；本轮没有解码新RGB/目标数组、读取HDF5数值、加载Policy权重张量或创建环境。
- 合成测试验证匹配MAE、坐标/均值保持、块结构、零扰动保留、相同能量导致正负误差变化、分量恒等式及坏角色/预算拒收。新增17项、连同原专家误差相关测试38项通过；CPU回归为666项通过、耗时14.49s，排除可选Crafter依赖的`tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`及`tests/test_smoke.py`，不声称全量pytest。
- 新入口py_compile、CPU隔离下`check_environment.py`及`git diff --check`通过；配置/算子hash在校验后保持不变。最终文件hash、工作区快照、文档链接和验证命令保存在`/tmp/libero-expert-camera-target-error-protocol-freeze-verification-20261102-v1.json`。原结果、checkpoint、用户AGENTS.md/.vscode及无关修改均保留，未暂存或提交。

## 下一步与主实验边界

后续[执行绑定](../configs/libero-task4-task7-expert-camera-target-error-execution-binding_v1.json)与runner现已实现、测试并按原预算完成，科学设计和算子hash保持不变。实际检查、数据/预测行数及输出hash以[执行报告](libero-task4-task7-expert-camera-target-error-diagnostic-2026-11-02.md)为准，原冻结验证记录仍保存。下一步只读已保存的命令/逐demo分层，不追加本轮模型前向。科学设计若确需改变仍须新版本，不覆盖本设计或把次级结果改成主指标。

此项只评估原source支持域中的固定Policy映射，不是query泛化、跨任务迁移、资格成功率或Knowledge机制。qualification、reserved、validation、held-out及final evaluation继续封存，SPT/Knowledge/Module均不更新；Crafter、Continual World和RoboCasa365不纳入本项。

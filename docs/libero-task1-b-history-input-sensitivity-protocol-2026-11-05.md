# task1 B固定历史输入表示敏感性协议（2026-11-05）

用户要求另冻固定权重的历史输入敏感性诊断，检验观测分布变化的影响。承接[专家目标误差](libero-task1-b-fixed-head-expert-target-error-results-2026-11-04.md)：真实历史整体更准，重复当前平移MSE均值+30.74%，但此前闭环仅重复当前在seed24增加4次成功。当前未确定分布偏移是根因；本项先冻结动作头输入表示的受控代理诊断，当前只冻结/检查，不解码真实缓存或研究读出。

## 范围和数据角色

固定task1 B_fomaml、训练seed20261022/23/24、原编码器500及完整support候选读出500端点、同一support scaler。两输入均用同一个候选头，运动读取真实五帧或重复当前；夹爪始终当前。不开baseline头，不重训、不选checkpoint、不改loss或scaler。闭环稳定化保持5步，本项无环境或稳定化动作。

使用已有source meta-query demos30..49缓存，2598个causal pre-action帧，只比较全部2518个t>=5完整历史帧/seed，共7554个模型×专家帧配对。target、同demo[t,t-1,t-2,t-3,t-4]历史和此前两输入预测均hash绑定，不按闭环4条成功、事件、误差或seed选帧。示教编号不是官方初态编号；不打开任何官方初态、HDF5、RGB或PT参数。source-support仅是旧scaler/权重来源，不读取support payload。

本项直接扰动公开观察导出的256维pre-head融合表示，**不重算编码器，不模拟实际RGB/本体传感器扰动**。它检验读出端的局部响应，不验证实际部署分布偏移或视觉/本体根因，不证明历史在环境机制中必要。目标始终是原专家target；未假定合成扰动后的输入对应真实物理状态或仍应执行同一动作，MSE成本仅是相对原target的敏感性。

## 受控扰动与数值空间

每个共同专家帧固定一个256维Rademacher方向u，坐标为±1，RMS=1。用NumPy PCG64测量seed20261105一次生成(2518,256)int8，固定row顺序；三个训练seed共用同一方向表，编码器不同的坐标不解释为同一物理变化。不得按预测/target选择方向或重采样。

方向按目标帧的窗口定义；重叠窗口中的同一历史特征可能得到不同合成delta，因此这些probe不构成时间上一致的观测轨迹，不用于推断真实连续传感器噪声。

幅度epsilon固定0.01、0.05，各自使用±u成对。epsilon是support scaler标准化坐标的RMS单位，非物理单位或实测部署噪声；退化特征的scaler可能按旧合同置1，不能声称每维都是其真实标准差。

先按旧合同构造float32标准化/串接/除sqrt(5)的干净输入，精确转float64；在既有float64读出算术空间加入`epsilon*sign*u/sqrt(5)`。不再量化扰动输入，避免量化差成为强度差；这也意味着本项不是部署float32路径的完整仿真。权重仍为同一归档float32值，矩阵算术/tanh均float64。

| 位置 | 真实历史运动输入 | 重复当前运动输入 | 用途 |
|---|---|---|---|
| current_only | 仅当前块扰动 | 扰动当前后重复五块 | 当前输入依赖；两臂总能量不匹配，仅次级 |
| past_only | 过去四块同向扰动，当前固定 | 输入不变，复用干净输出 | 路由负对照，非鲁棒性胜出 |
| coherent_all | 五块均同向扰动 | 五块均同向扰动 | 主比较，匹配输入扰动能量 |

coherent_all两臂每帧输入delta的L2²均为256*epsilon²；current_only真实历史为其1/5，重复当前为全量；past_only真实历史为其4/5，重复当前为0。记录实际能量，不把不匹配的次级位置直接用于两臂脆弱性排名。夹爪随当前扰动，past_only保持；各条件下两臂夹爪输出必须精确相同。

同一个线性头在coherent_all相同delta下，其logit变化理论上相同；这是软件/代数门槛。若tanh后动作响应或目标误差成本不同，应结合干净工作点和误差方向解释，不称为上游编码器放大或独立历史模块贡献。

## 指标和预定判断

主比较固定epsilon=0.05、coherent_all、全部2518帧、平移3维。每臂先相对自身干净输出计算成对MSE成本：`S_arm=0.5*(MSE_plus+MSE_minus)-MSE_clean`，再报告`S_history-S_repeat`。三seed均>0才记为历史在此代理下目标误差成本较高；不把flag自动称为闭环失败根因。负成本可能出现，不截断为0。

另报epsilon0.01的同定义结果；全部位置、两幅度、正负方向、三seed和五个既有专家分层均保留。主指标按帧等权，不事后换成有利分层或幅度，不新增显著性检验、CI或独立未见样本主张。

同时计算平移、旋转、运动6维、夹爪、七维的动作响应能量`0.5*mean(delta_plus²+delta_minus²)`及其RMS/epsilon、干净/正负MSE。对每一正负方向检查`DeltaMSE=2mean((clean-target)*delta)+mean(delta²)`，再对方向项和能量求成对平均。响应大、误差成本大、干净误差大是不同指标；不相互替代。

分层保持all2518、known_open_event23、known_open_segment232、initial_open974、expert_close1312。事件嵌套在段中，不能相加当独立样本；这些是专家命令标签，不能直接作为Policy在线输入。干净专家分布准确性与旧闭环成功率继续分别报告。

## 精确预算、复用和门槛

每seed：身份current/history各1次，零扰动current/history各1次；每个epsilon×sign组合只新算current一次，以及history当前/过去/整窗口各一次。重复当前在current_only与coherent_all复用同一读出，past_only复用身份；真实历史夹爪也复用对应current输出，无隐藏第三路。

| 项目 | 冻结预算 |
|---|---:|
| 干净身份Linear / 零扰动Linear | 6 / 6批 |
| 扰动Linear | 48批 |
| 总Linear / 总矩阵行 | 60批 / 151080行 |
| 每seed逻辑预测表 | 28（身份/零扰动各两臂＋12条件各两臂） |
| 三seed逻辑预测表 / 行 | 84 / 211512 |
| 目标/feature/候选头/父预测归档解码 | 1 / 3 / 3 / 3 |
| target / feature行解码 | 2598 / 7794 |
| 父两输入预测行解码 | 15108 |
| 新CNN/optimizer/backward/环境/HDF5/RGB/PT参数/初态 | 全0 |

矩阵调用和逻辑输出表是不同预算：复用的输出仍可在逻辑条件表中出现，不算额外Linear或新科学样本。固定seed22→23→24、epsilon0.01→0.05、sign−1→+1、位置current→past→coherent；记录所有attempted/completed和行数。CPU单线程、Python3.10.22、NumPy1.22.4，CUDA_VISIBLE_DEVICES空，OMP/OPENBLAS/MKL线程均1。

身份两输入须与上一轮float64归档动作绝对差<=1e-12且夹爪符号相同；零扰动输入和输出精确元素一致。past_only重复当前全动作及真实历史夹爪不变；每条件两臂夹爪一致。coherent输入delta、输入L2²、相同头logit变化和正负logit对称余差<=1e-10；误差分解余差<=1e-12。任何hash/角色/因果索引/端点/预算/数值门槛失配保存failed，主结论无效，不放宽容差、不补seed/帧或重试预算。全部3seed和精确预算完成才允许主汇总。

## artifact和解释边界

[配置](../configs/libero-task1-b-history-input-sensitivity-diagnostic_v1.json)、[入口](../experiments/run_libero_task1_b_history_input_sensitivity.py)、[测试](../tests/test_libero_task1_b_history_input_sensitivity.py)及本文独立hash绑定，旧已执行代码/协议保持不变。默认命令只核对metadata，不解码真实NPZ或读出；未来--execute另行按预算运行，排他输出到/tmp同名前缀20261105-v1。

共享NPZ保存合法共同帧/target/history/mask及方向表，三seed NPZ保存身份/零扰动/全部扰动动作和logit、实际输入能量。记录代码、配置、源文件、候选参数hash、CPU运行版本与成本；不复制原权重入仓库。冻结与执行状态见[当前记录](libero-task1-b-history-input-sensitivity-diagnostic-2026-11-05.md)。

任一方向结果均只是固定动作头在规定表示探针下的敏感性，不证明真实观测shift、视觉/本体因果、必要记忆或闭环修复。当前保持500端点、稳定化5步、现有loss及旧候选停止决定；不自动训练、更新SPT/Knowledge或注册Module。validation、held-out、qualification、reserved和final evaluation继续封存。

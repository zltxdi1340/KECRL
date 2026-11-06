# task1 B路由读出120 episodes配对闭环结果（2026-11-04）

**全部120条fresh轨迹已完成并审计，运动历史候选未带来闭环收益，停止该配置推进。** current_all7 baseline为3/60（5%），history_motion_current_gripper候选为0/60；三seed分别2→0、0→0、1→0。同seed/state配对为baseline-only3、candidate-only0、共同成功0、共同失败57。

上一轮source-query平移/七维MSE下降11.66%/5.14%的离线结果保留，但没有转化为本配置的闭环成功。不能据此认定历史无用、视觉或动作头是唯一根因，也不能以新的成功率注册Module、接受SPT或更新Knowledge。

## 设置与执行边界

先完成[runtime接口及数值核对](libero-task1-b-routed-runtime-interface-diagnostic-2026-11-04.md)，再冻结[配置](../configs/libero-task1-b-routed-history-closed-loop-diagnostic_v1.json)、[协议](libero-task1-b-routed-history-closed-loop-protocol-2026-11-04.md)、[执行代码](../experiments/run_libero_task1_b_routed_history_closed_loop.py)和[绑定](../configs/libero-task1-b-routed-history-closed-loop-binding_v1.json)，之后才读取candidate初态和构建环境。

固定task1 B_fomaml三训练seed20261022/23/24的500-update编码器，以及[上轮](libero-task1-b-routed-history-readout-diagnostic-2026-11-04.md)六个完整support的500-update读出；没有选三折或中间模型。两臂都是重新执行的新读出控制器，**本轮baseline不是历史原B Policy动作头，也不是旧baseline/jitter**；历史成功率未加入本轮分母或当作同条件对照。

每次reset清空历史，前4个动作两臂共用同seed baseline头，第5步候选才读完整历史：运动6维当前+过去4帧、夹爪1维当前。startup在本轮环境结果之前固定，没有补零/未来帧或混入稳定化观察。所有模型不变，eval/inference模式，无梯度或优化。

仅官方candidate_nonregression初态0..19，每模型按0→19递增，seed→baseline/candidate固定顺序；每模型新建adapter、该模型内复用20次，共6个。环境seed20261022、稳定化5个零动作、Policy horizon300、native horizon305。先验预算120 episodes、最多36000个Policy动作、另600个零动作，完整跑完后才汇总，无成败依赖停跑、重试或替换。

## 完整配对结果

| seed | baseline成功/20 | 候选成功/20 | baseline-only | candidate-only | 共同成功 | 共同失败 |
|---|---:|---:|---:|---:|---:|---:|
| 20261022 | 2 | 0 | 2 | 0 | 0 | 18 |
| 20261023 | 0 | 0 | 0 | 0 | 0 | 20 |
| 20261024 | 1 | 0 | 1 | 0 | 0 | 19 |
| 合计 | 3/60 | 0/60 | 3 | 0 | 0 | 57 |

三个baseline-only配对是seed22/state2（143步成功）、seed22/state10（137步成功）、seed24/state5（109步成功）；对应候选均300步time_limit。状态编号是官方candidate初态，不是示教编号。全部117个失败episode均为time_limit；没有异常终止混入失败分母。

`strict_candidate_improvement_all_three_seeds=false`，三个seed均无候选新增成功。本项未做显著性检验，也不声称严格总体性能差异已统计确立。每臂60条来自相同20个场景跨三训练seed重复，不能视为60个独立未见场景；正式qualification/validation/held-out/final evaluation继续封存。

## 技术门槛、成本与provenance

20个reset公开观察参照、其余100次逐位比较全部通过。相同seed/state前4动作共240次精确比较、前5个pre-action公开观察共300次精确比较通过；前5个256维特征也经独立归档检查逐位相同。两臂从相同reset和共同四步后分开执行，不能将本结果归于已检测的reset或startup失配；这并不排除所有其他实验或机制问题。

实际120 episodes、6 adapters、6次编码器checkpoint加载；35489个Policy动作、同数的编码器/读出前向，另600个零动作，共36089次受控native environment.step。读出primitive Linear调用53249次，候选成熟历史每次包含两个Linear计算。attempted/completed计数逐项相等；没有技术失败、补跑、checkpoint选择或额外训练。

CPU单线程推断，CUDA主动隐藏，确定性/TF32关闭；渲染固定EGL物理GPU1。运行前GPU1为30MiB/0%、无compute作业，资源门槛通过；不做设备回退。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、robosuite1.4.0、MuJoCo3.2.3、LIBERO revision8f1084e3132a39270c3a13ebe37270a43ece2a01。入口总耗时639.919s（约10.67分钟），包含真实控制、仿真、渲染和归档；不与上轮小读出训练9.49s混为相同计算任务。

官方初态使用已有选择性前缀读取器：完整容器checksum和固定pickle头核对后，仅数值解码20行candidate；qualification/reserved解码0，不使用torch.load/pickle完整反序列化初态，不记录原始初态向量或live private state。闭环新增HDF5/RGB数据集读取、optimizer/backward均0；前置接口另有80行合法source-support payload，预算已单独记录，不混为整次会话无示教读取。

120份episode JSON保留所有动作、公开joint/gripper与reset偏移、history步号/route及端点hash；120份NPZ保留35489行公开观察导出的256维pre-action特征及执行动作，不存RGB像素/private simulator state。主结果JSON保留全部episode摘要、文件hash、四格配对与实际计数，events.jsonl记录时间线；所有输入/编码器/读出/配置/代码hash保持。

## 验证与独立审计

闭环及接口相关35项测试、前置接口/训练相关61项测试通过，覆盖完整预算、合法提前成功、reset/前缀拒收、首动作前中止、旧输出保护及角色/hash门槛。排除`tests/*crafter*.py`、`test_spatial_gather_spt.py`和`test_smoke.py`的CPU回归为**1149 passed in 17.14s**；5份新Python文件编译、环境检查与diff检查通过，不声称包含可选Crafter的全量pytest。

独立审计不导入项目实现、Torch或仿真器，从六个读出归档及120个feature/action NPZ逐帧重建标准化、历史路由、共同启动和动作。核对120份JSON、35609行公开状态、35489行动作/特征、全部公开偏移/L2、terminal边界、逐步history、100/240/300次配对以及全部seed四格结果。1184391项检查通过；最大动作重放差6.67e-7，低于冻结1e-5，公开状态指标容差1e-10。审计另有120个算术读出批次，只重放既有动作，没有新科学样本、CNN、环境、HDF5、初态解码或训练。

| 证据 | SHA-256 |
|---|---|
| 科学配置 | `b69b72b5fffba3e239e34eddc1e8e2c6bbcd57148473f6bb982f245518aaa288` |
| 执行绑定 | `0824787f9c6d8bd5e9de98347d21ebc671ea57573248434334653b2901d084ae` |
| /tmp/libero-task1-b-routed-history-closed-loop-20261104-v1.json | `9cd8039dcbe79ec9c847e4e2681e62b47bf5e962472f3a26e00b4c72ad500084` |
| /tmp/libero-task1-b-routed-history-closed-loop-independent-audit-20261104-v1.json | `978cc7e588654693f13ba134f6668f2e760df9a008f6d7100d89615eebee4bab` |
| 独立审计脚本 | `eec8e49be668f262cd4282730730a2c1eda69c5777dff3709f9e75547b207597` |

独立审计结果：`/tmp/libero-task1-b-routed-history-closed-loop-independent-audit-20261104-v1.json`；最终工作区/哈希核对记录：`/tmp/libero-task1-b-routed-runtime-and-loop-final-verification-20261104-v1.json`。artifact留在/tmp，不复制进仓库；AGENTS.md、.vscode及无关旧修改保持，未提交或推送。

## 决定与下一步

停止该路由配置继续训练或扩大闭环预算，不因source-query误差改善推进SPT/SPI正式结果。保持500端点、5步稳定化和现有loss；保留current_all7作为本轮诊断参照，不将其3/60解释为可靠控制或自动替换主线Policy。

下一步先只读全部60对已保存轨迹，三个baseline-only配对单列但不丢弃其余57对：共同前5个观察下检查第5步动作分歧，再比较前20步及完整close/open命令时序和公开drift，区分早期偏离与后续累积。没有真实目标几何/专家runtime动作，不把joint偏移或命令符号直接当任务阶段或根因。所需公开特征已归档，应先另冻只读定义/预算，再分析；暂不增加updates、改loss或添加新训练变量。候选无可靠收益与具体失败机制尚未分离，是下一项要解决的问题。

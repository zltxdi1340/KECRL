# task1 B路由读出runtime接口结果（2026-11-04）

已冻结并完成固定500端点的单步接口检查，全部门槛通过。新模型/读出训练和环境交互为0；接口用时3.663s。此结果说明实现可按合法输入和确定历史执行，不说明闭环任务成功或必要记忆。

## 检查与结果

先冻结[配置](../configs/libero-task1-b-routed-runtime-interface-diagnostic_v1.json)、[协议](libero-task1-b-routed-runtime-interface-protocol-2026-11-04.md)和绑定，再执行。[runtime helper](../experiments/libero_routed_history_runtime.py)强制显式reset与顺序步号，历史最多5帧，跨demo/episode清空，不使用稳定化帧。前4动作两臂共用同seed baseline读出，第5动作起候选使用自身运动历史/夹爪当前路由；该启动合同在闭环之前选定。

缓存检查覆盖三seed、两臂、source-support demos0..29全部4059帧/seed，共24354个流式读出；成熟历史23634行与上一轮固定500预测比较，最大差3.13e-7，全部夹爪符号相同。720次启动输出两臂逐位相同，所有权重/scaler保持。

新增HDF5数值读取仅source demo0前8个sequence10窗口：动作t1..80配stored obs0..79，80行公开RGB/joint/gripper和目标；原batch digest精确匹配，不读action0/初态。三个原编码器各1个batch、80个单步及首步1个重复，总246次前向/483有效行。batch/单步/旧cache特征和原动作最大差2.98e-7；重复首步逐位相同。两臂再用这些单步特征流式读出480次，与旧cache流式动作最大差1.0431e-6，低于1e-5冻结容差。

| seed | 缓存流式最大动作差 | 单步特征相对旧cache最大差 | 单步来源读出相对cache最大动作差 |
|---|---:|---:|---:|
| 20261022 | 2.38e-7 | 1.49e-7 | 3.28e-7 |
| 20261023 | 2.09e-7 | 1.79e-7 | 1.04e-6 |
| 20261024 | 3.13e-7 | 1.19e-7 | 3.73e-7 |

两个读出臂的head/scaler和三个编码器均固定，原CNN gradNone、权重hash保持。全部读出24834次、编码器246次、checkpoint加载3次，与冻结预算一致，无失败或补跑。查询、官方初态/private、环境、优化和backward新增0；正式封存角色保持。

## 验证与证据

接口/路由训练相关61项测试通过；后续闭环准备与共享相关专项35项通过，排除Crafter及smoke依赖组的CPU回归1149项通过。独立审计不导入项目代码，从原cache/权重和新归档重建全部流式动作、成熟历史与启动，以及单步桥接；471463项检查通过，最大差1.0431e-6。审计另有12个算术读出批次，仅复核24834行既有动作，不新增科学样本/编码器/环境/HDF5/训练调用。

| 证据 | SHA-256 |
|---|---|
| 科学配置 | `18f178f15c77585464e93f8c002e8dd764fd919d85da225a50f3f9f44f0669f9` |
| 执行绑定 | `3787ff8bb0b354b2527e63ac69cfd9129b202ac936eeb2134c9427faccdf3743` |
| /tmp/libero-task1-b-routed-runtime-interface-20261104-v1.json | `5a6f52a04a4110a37d219ea24b1c84a0e8e4215db34ce54cedc3ed317392d4f1` |
| /tmp/libero-task1-b-routed-runtime-interface-independent-audit-20261104-v1.json | `0f5112023c7845015718dba49dff0d6e720d8b1544f25a6969cbc6a7fe539501` |
| 独立审计脚本 | `a2765cdd0c8ffdebd5dc9d6681ddecd5256ab5b06cb0c59b8cb0347fb8abd249` |

编码器、执行head和CPU路径相同，单步来源读取约2.01–2.22ms/帧（该接口小样本计时，不包含真实仿真与渲染，不能据此估算可靠闭环速度）。没有以这一小样本进行任务或超参数选择。

下一实验已另冻[120条fresh配对闭环](libero-task1-b-routed-history-closed-loop-protocol-2026-11-04.md)，仅task1 candidate0..19、三seed两臂、稳定化5步、horizon300、固定500端点，最多36000个Policy动作另600个零动作。本接口不提供成功率，闭环实际结果须以单独dated report为准；不接受SPT/Knowledge/Module。

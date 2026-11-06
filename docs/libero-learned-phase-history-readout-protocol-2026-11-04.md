# 固定表示的当前／历史监督读出协议（2026-11-04）

本项承接[切换近邻分解](libero-phase-history-transition-neighbors-diagnostic-2026-11-04.md)，检验可学习读出能否利用合法历史，分离固定等权近邻距离的取舍。用户已要求继续并尽快实验；本协议冻结后执行实际读出训练，不由旧近邻结果直接修改原Policy。

## 角色与样本

[科学配置](../configs/libero-learned-phase-history-readout-diagnostic_v1.json)固定task0/1、三seed20261022/23/24、A/B/C全部18个500-update表示端点。仅现有source-support demos0..29缓存；共同t>=5，历史为当前及过去4帧、同demo、不含未来或动作标签。原Policy不加载、不重训，闭环默认5步和原训练设置保持。

为分开新读出的拟合与评价，完整demo按0..9、10..19、20..29分3折，轮流作为readout_probe，其他20条为readout_fit。两角色都属于meta_support，不是官方validation、held-out或qualification，也不是meta_query。编码器先前已见全部30条：只能解释固定表示条件下新读出的跨demo学习，不能声称未见编码器泛化。3折合并时每个目标帧仅评价一次，全部31/34个open事件保留。

## 配对模型与训练

每个端点×3折×两模式，共108个小读出，**每个500 updates**，合计54000更新；108个并非108个独立seed。当前模式将当前256维表示重复5份；历史模式拼接合法5帧表示。用fit当前帧计算float64均值/总体标准差，std<=1e-6置1，同scaler用于各lag；拼接后除sqrt5并转float32，两个输入均1280维。

两模式均为Linear(1280,7)+tanh，8967参数，权重/bias全0，初始输出与权重完全相同。无隐藏层、门控、事件输入或损失重加权。输入秩和优化几何仍随历史变化，参数量匹配不能独立证明必要记忆。

Adam lr0.0003、betas(0.9,0.999)、eps1e-8、weight_decay0，梯度L2 clip1，batch64，现有均匀七维action MSE。按numpy default_rng(training_seed+100000×fold)生成500×64的fit帧有放回采样表，两模式复用同一张表及hash。任何probe demo及其历史不得进入fit采样或scaler。原始专家动作仅作监督目标，不进入输入。

记录0/100/250/500步曲线，每次对全部共同帧只前向一次，再分fit/probe与既定命令层统计。原始初始化0不当成可执行Policy；zero输出单列且对非零专家符号计错。不早停、不加预算、不挑中间checkpoint，主比较永远使用500。

## 指标和预设判断

保存平移、旋转、夹爪、七维MSE，符号错误率和exact-zero数；分层为全部帧、open事件当步、完整已知open命令段、初始open、expert close。全段包含当步，不伪造物理release阶段。每端点合并3折的out-of-fit预测，六个task×feature_arm分别保留全部三seed。

描述性联合cue要求每个seed均满足：历史的open当步符号错误率严格下降、完整已知open段错误率不增加、整体七维MSE不增加、整体平移MSE不增加。缺失为null，零差不算严格改善。无论cue如何，不自动触发Policy训练、闭环、SPT接受或Module资格检查。

## 固定实际预算

| 项目 | 数量 |
|---|---:|
| 读出模型／每模型更新 | 108／500 |
| optimizer updates／backward／训练前向 | 各54000 |
| 训练样本行（有放回） | 3456000 |
| 指标前向／样本行 | 432／1450224 |
| 合并评价行／模式 | 60426 |
| 端点×open事件配对 | 585 |
| 新完整Policy前向、原Policy更新、环境调用 | 0 |
| 新HDF5、RGB、初态、private state、原checkpoint payload读取 | 0 |

运行前记录GPU占用；实际仅CPU单线程、CUDA_VISIBLE_DEVICES空，Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、确定性开启。不占GPU、不创建环境；source-query、validation、held-out、qualification、reserved及final evaluation保持封存。

## 绑定、归档和停止

[runner](../experiments/run_libero_learned_phase_history_readout.py)、[测试](../tests/test_libero_learned_phase_history_readout.py)和本文通过binding锁定后才进行真实训练。先核对11个输入pin、父结果专用完成状态/审计/执行代码，再读取18份表示NPZ。默认命令只核对元数据，--execute才训练。

新结果为`/tmp/libero-learned-phase-history-readout-20261104-v1.json`，108份读出归档和逐模型事件位于同名前缀artifact目录。保存全部曲线预测和权重、scaler、fit/probe/source行、配对批次、每步loss/裁剪前范数、final Adam缓冲、实际超参数及每个参数500步计数。原checkpoint和旧artifact只读。

输出或失败路径存在则拒绝覆盖。角色/hash/完整性/初始配对/非有限值/实际预算不符立即停，保存完成读出、当前读出权重、已执行更新和失败；不放宽规则或换seed。数值验证使用合成小预算；独立审计复算全部保存预测/指标与配对、角色和实际步数，不额外训练模型。独立NumPy预测绝对容差1e-5、指标1e-10，不能事后调整。完成后写dated结果并同步README/roadmap/后续计划。

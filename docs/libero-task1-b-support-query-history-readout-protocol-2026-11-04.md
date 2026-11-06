# task1 B完整support／query历史读出协议（2026-11-04）

承接[三折监督读出](libero-learned-phase-history-readout-diagnostic-2026-11-04.md)唯一通过支持集联合条件的task1 B线索。用户已要求继续本步；本项先冻结并训练完整source-support两臂，再检查source meta-query固定500端点，不直接进行闭环。

## 角色与训练

[配置](../configs/libero-task1-b-support-query-history-readout-diagnostic_v1.json)固定原task1 B_fomaml三seed20261022/23/24的500-update编码器。support为demos0..29、query为30..49；全部为source SPI内部既有示教角色。query以前用于诊断，不能称未见或未触碰测试集。validation、held-out、qualification、reserved及final evaluation继续封存；不读取初态或private simulator state。

本次新训练6个读出：每seed×current_repeat5/causal_history5，各500 updates，总3000。**不挑三折中的任何模型**，从相同全0权重/bias重新训练；两臂Linear(1280,7)+tanh、8967参数，输入与scaler、七维均匀MSE及Adam设置沿用父诊断。完整support共同t>=5有3939帧，用它们的当前表示计算mean/std，应用于所有lag；std<=1e-6置1，拼接除sqrt5。

两臂共用numpy default_rng(training_seed)产生的500×64有放回support采样表。Adam lr0.0003、betas(0.9,0.999)、eps1e-8、weight_decay0，clip1。记录support的0/100/250/500曲线、全部权重/预测及实际更新；query仅评价500，不读取中间点选择端点、预算或seed。原Policy权重冻结，闭环默认5步和原loss不改。

## query门槛与CPU数值对齐

六个support读出及归档必须都完成500 updates，实际总optimizer/backward为3000，随后才允许创建query数据集。任何缺失、错seed/模式或更新数不符立即拒绝。source-query从不进入scaler、采样或梯度。

执行前两GPU均有其他作业，因此本新协议明确固定CPU单线程、CUDA_VISIBLE_DEVICES为空，不抢占GPU、不做自动设备回退。旧support表示在GPU上采集，使用CPU query表示前必须完成数值桥接：同一组原始support首8个窗口，输入摘要精确匹配父记录；三个冻结编码器各前向一次，checkpoint文件/tensor hash不变，公开输入不变、gradNone。与旧缓存比较80行256维表示和原动作，绝对差均<=1e-5；两臂新500读出在其中76个合法历史目标帧上也须与旧缓存输入预测相差<=1e-5。桥接失败保留产物，不放宽容差或读取query。

三个桥接都通过后才实际创建meta_query数据集，沿用CausalT1LiberoSequenceDataset：t>=1的动作配stored obs[t-1]，sequence10、stride10、batch8、无hidden state，不读action0或init_state。历史只取同demo的当前及过去4帧；主目标共同t>=5。query原始t>=1帧数2598、窗口269来自已归档历史metadata，在新query数值访问前冻结；若实际不同则停止，不改预算。

每个编码器query主前向34批，首批另重复一次，动作和表示重复须逐位相同。原始输入摘要在三个seed间一致，捕获后再核对输入/target不变、权重不变和gradNone。使用既有action-head输入hook，不增添其他部分神经前向。旧完整2598帧query的原Policy连续分量MSE作为CPU/runtime数值核对，绝对差<=1e-5；gripper sign accuracy差只描述，不声称CPU/GPU符号逐位等价。

## 预算

| 项目 | 固定数量 |
|---|---:|
| 新读出／每读出更新 | 6／500 |
| optimizer updates／backward／训练前向 | 各3000 |
| 训练有效样本行 | 192000 |
| support指标前向／样本行 | 24／94536 |
| query最终读出前向／样本行 | 6／15108 |
| 原编码器query主批次／有效行／含padding槽 | 102／7794／8070 |
| 原编码器query重复批次／有效行 | 3／240 |
| support数值桥接批次／有效行 | 3／240 |
| 原checkpoint加载 | 3 |
| 新support公开观察每模态／动作行 | 80／80 |
| 新query公开观察每模态／动作行 | 2598／2598 |
| 原Policy更新、环境调用、初态/private/action0读取 | 0 |

共108次冻结原Policy完整前向，仅用于桥接/特征；不是108次Policy训练。query共享一次加载，张量缓存<=512MiB。query共同目标2518帧/seed/mode；当前与历史同目标、同scaler。CPU桥接中numpy小读出算术单列，不作为额外神经前向或新研究样本。

## 指标与停止决定

沿用平移/旋转/夹爪/七维MSE、符号错误率和exact-zero数；分全部、已知open当步、完整open命令段、初始open、expert close。阶段来自专家命令而非物理真值。初始open不伪造t0事件；若query事件在t<5则单列排除数，不补历史或事后放宽范围。

主query cue要求三个seed各自满足：历史open当步符号错误率严格降低，完整已知open段符号错误率不增加，整体七维MSE不增加，整体平移MSE不增加。缺失为null，持平不算事件严格改善。support同指标保留作描述，不能替代query主条件。CPU/GPU桥接不是机制干预，不把设备微差或读出收益认作必要记忆。

无论结果如何，本次都不自动追加闭环、搜索历史长度/权重、更新活动SPT或Knowledge、注册Module。只有完成审计并另冻candidate闭环及历史启动规则后，才可能检验实际控制；本项t>=5专家输入不建立runtime初始4帧的行为。

## 绑定与provenance

[runner](../experiments/run_libero_task1_b_support_query_history_readout.py)、[测试](../tests/test_libero_task1_b_support_query_history_readout.py)及本文绑定后才执行。默认只做元数据核对；--execute启动训练和后续条件query读取。26个输入pin、旧bound代码、三个特征/checkpoint的hash、source角色和文件SHA均核对，完整HDF5 SHA与角色payload访问分开记录。

结果为`/tmp/libero-task1-b-support-query-history-readout-20261104-v1.json`及同名前缀artifact目录；保存六个support读出、三个桥接、query共享目标/ID、三个query表示与六个新读出预测、canonical窗口、batch摘要、逐步loss/norm/Adam缓冲、事件和实际计数。旧结果、权重及失败记录不改写、不复制checkpoint进仓库。

任何角色、状态/hash、非有限值、数值桥接、重复或预算失配立即停，保存完成产物、当前读出及部分query表示，不覆盖失败或重训补样。独立审计仅读取归档，不新增训练、原Policy前向或HDF5数值访问；独立读出预测容差1e-5、指标1e-10。完成后更新dated结果、README/roadmap和后续计划，明确仍为diagnostic。

# 阶段／历史信息可辨性诊断协议（2026-11-04）

本项承接[完整open段复核](libero-source-internal-overnight-error-review-and-full-open-results-2026-11-04.md)。新问题是：当前固定Policy表示和过去5帧表示，在相同的跨示教近邻规则下，对专家命令及两类expert-open上下文的区分有什么差别？本协议不把已有阶段差异当作记忆缺失或闭环根因。

## 范围与预算

[科学配置](../configs/libero-source-internal-phase-history-observability-diagnostic_v1.json)固定task0/1、三seed20261022/23/24、A_original/B_fomaml/C_pooled_bc全部18个500端点；继续闭环默认稳定化5步和现有loss。仅source-support demos0..29，t>=1 causal输入。query、qualification、validation、held-out、reserved、final evaluation、action0、初态和private simulator state保持关闭。

| 项目 | 固定预算 |
|---|---:|
| 完整Policy主前向／重复 | 819／18 |
| 有效模型预测及表示行：主／重复 | 62586／1440 |
| 含padding计算槽：主／重复 | 65340／1440 |
| 新共享源观察每模态／动作行 | 6954／6954 |
| 固定checkpoint加载 | 18 |
| 主表示近邻fold读出／样本行 | 2160／179280 |
| 既有公开夹爪状态次级fold／样本行 | 240／19920 |
| 新Policy训练、backward、环境调用 | 0 |

重复为每模型原首批，原batch8/sequence10/stride10不变。非修改钩子截取action head输入256维和logits；logits的tanh严格重建输出。旧62586行预测、819个batch摘要、18次重复的特征和动作、输入/权重不变与gradNone均须通过。没有真实帧warmup、额外partial Policy链或pool前后向；logit重建是检查算术。

既有公开gripper2只作次级对照，先读已缓存数组、每task只算一次，不当作9个独立seed。GPU读出前原角色/file SHA和idle门槛保持：physicalGPU1（cuda:0），<=512MiB、util<=5%、无compute进程；不占GPU0或CPU回退。Python3.10.22、Torch2.4.1+cu124、CUDA12.4、NumPy1.22.4、h5py3.11.0，单线程、确定性开启、TF32关闭，沿用原确定性pool。

## 标签、历史与参照隔离

所有模式共用t>=5，历史为当前h_t及h_(t-1)..h_(t-4)，每个h只来自该动作之前的完整原观察；同demo、不补padding、不取未来或action0。跨原batch边界仍按源帧ID取合法过去表示。当前模式使用相同目标帧和参照帧，不能以较多早期样本获得优势。

两个标签任务分别为：专家夹爪命令sign（-1/+1，neutral单列排除）；仅expert-open帧的上下文（初始可见open前缀=-1，已知close/neutral→open命令段=+1）。后者不是物理approach/release标签，初始前缀起点未知，不伪造t1切换。标签只用于参照库和评分；专家当前/过去动作、事件、phase标签都不进入特征。

每demo整条从近邻库和标准化统计排除，所有历史也必须来自各自demo。用其余demo共同合法当前帧的float64均值/总体标准差，std<=1e-6用1；同一个scaler应用到所有历史lag和两标签，避免按标签另选scaler。**Policy此前已训练全部support**：留一只隔离近邻和scaler，本项不是未见示教泛化，也没有重新训练30个编码器。

固定k=5、query chunk64；直接float64平方差按64维块累加，避免重复夹爪状态的Gram抵消误差。当前距离为单帧平方欧氏距离，历史距离为五个对应lag距离的均值。平局稳定保留源行顺序，投票平局取最近的并列类。重复五份当前信息的维度对照按数学等价复用当前距离/预测，不增加独立样本或额外模型读出。它限制“只是坐标数变多”的解释，但不证明历史内容必要或模型因果作用。

## 指标与判断

主空间为256维pre-head表示，次级为2维公开夹爪；不同空间不比较绝对距离或作OOD阈值。按task/arm/seed和共同样本分别报告command、phase balanced accuracy；额外报告全部31/34个已知open事件当步的command error，初始open、完整known-open段、expert-close以及逐demo分母。单类/空分层的balanced accuracy为null，错误率另报，不补0。

同task/arm全部三个seed同时满足以下条件时才记描述性联合history cue：phase BA严格增加、command BA严格增加、事件当步command error不增加。任何缺失则null，零变化不是严格增加。次级gripper不能救回主失败。原Policy符号误差只作为固定输入上的另一种读出参照，不将5NN和原线性head的差写成action head独立因果贡献。

无论cue如何，不自动训练、延长500预算、选择checkpoint、接受SPT、更新Knowledge或注册Module。近邻失败不证明真实欠观测，近邻改善也不证明必须有记忆；度量/表示、已见support训练和示教上下文都限制解释。旧task4/7单帧5NN和停止决定保持，其结果不与当前task0/1合并。

## 执行和保留

[核心算子](../experiments/libero_phase_history_neighbors.py)、[执行入口](../experiments/run_libero_source_internal_phase_history_observability.py)、[专项测试](../tests/test_libero_source_internal_phase_history_observability.py)及本文由[执行绑定](../configs/libero-source-internal-phase-history-observability-binding_v1.json)锁定后才读取新源观察/checkpoint并执行前向。无损保留旧artifact，新结果和特征/近邻NPZ、fold scaler、事件另存 `/tmp/libero-source-internal-phase-history-observability-20261104-v1.json`及同名前缀artifact目录。

任何角色、hash、shape/dtype、因果ID、预测/重复、钩子、权重/梯度或预算不符立即停止，保存失败和partial表示，不放宽门槛或重挑模型。执行前合成测试的夹具遗漏与异常消息匹配错误另存两份preflight失败记录，均发生在新研究前向0时；实现和科学规则未因它们改变。实际结果与独立复核写另一份dated报告。

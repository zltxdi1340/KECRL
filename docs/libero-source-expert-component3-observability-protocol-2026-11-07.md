# 第3平移动作分量正负方向可辨性对照：科学协议冻结（2026-11-07）

**科学方案已冻结，本轮未执行表示采集、近邻读出、训练或环境。** 用户要求承接[错号/幅值分解](libero-source-expert-component3-error-types-results-2026-11-07.md)，检查现有观察信息是否容易区分专家第3分量的正负目标。冻结配置为[component3 observability v1](../configs/libero-source-expert-component3-observability-diagnostic_v1.json)，验证器为[纯元数据协议](../experiments/libero_source_expert_component3_observability_protocol.py)；配置SHA与本轮核对记录保存在`/tmp/libero-source-expert-component3-observability-freeze-20261107-v1.json`。

本项是来源训练拟合diagnostic，不是formal结果、独立未见示教泛化、SPT更新或Module qualification。G未通过/H未执行的决定保持。validation、held-out、qualification、reserved、final payload与source meta_query均不访问；不更新Knowledge或注册Module。现有500 updates、未来闭环5步稳定化和均匀七维masked action MSE保持。

## 问题、假设及现有证据

当前30个固定端点在source专家末段同时存在错号、同号不足、过大与零目标误差。task7的75个material正目标末帧来自17条示教，A/B分别有60/51帧在三seed上共同错号，不能仅凭总体投影偏小就统一放大动作。

主问题是**固定Policy的当前pre-head表示是否支持跨示教参考库读出正负方向，以及合法过去表示是否增加同一读出规则下的分辨力**。它检查的是给定编码器/距离/读出下的信息可读性，不是原始RGB信息的理论上限。低读出准确率不能证明环境欠观测，高准确率或历史改善不能证明head/记忆是闭环根因。

当前canonical专家缓存有目标、预测和帧ID，没有30端点匹配的完整pre-head表示；原G离线归档仅有汇总指标，旧18模型的feature端点weight hash与本批30模型无匹配。清单结论限于已声明的产物，不声称全机无缓存。实际执行需要新的有限源观察前向；本轮只读取源码、已声明JSON和已有计数CSV，无NumPy数组解码、checkpoint/HDF5数值读取或CNN调用。

既有[历史可辨性](libero-source-internal-phase-history-observability-diagnostic-2026-11-04.md)针对早期task0/1的夹爪命令和条件open标签；[task1路由历史候选](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)离线改善但闭环0/60。它们不覆盖本次第3平移分量标签，不能当作当前方向已经可辨或历史必然有效的证据。

## 模型、数据、标签与输入

- 固定task0/1/3/4/7 × seed20261106/07/08 × A_original/B_fomaml全部30个现有500端点，逐一绑定checkpoint路径、文件SHA和weight hash；不挑seed/checkpoint。
- 只使用source meta_support demos0..29，保持`obs[t−1] → action[t]`、t≥1、CNN-only、sequence10/stride10/batch32、float32、无增强。原四输入为agentview、eye-in-hand RGB及公开joint7/gripper2，不读取物体/末端/接触或private simulator state。
- 标签为`sign(action[t,2])`，仅`abs(action[t,2])≥0.025`进入正负方向读出，边界两侧对称且包含0.025；零/近零帧保存排除计数，不更改阈值或改为其它分量。动作正负不能直接称世界上下。
- 非修改hook截取原action head前256维`h_t`，同时归档原logits及完整七维动作；`torch.tanh(logits)`须逐位重建原输出。新全部111888预测行须与上一读出逐位一致，不能只核对MSE或第3分量。
- 主空间为256维固定表示；次级为同一次源观察读取获得的公开`joint7+gripper2`共9维，每task仅计算一次，不复制为六个独立模型。两空间不比较绝对距离或OOD阈值，次级不能救回主对照的阴性。

两输入模式为当前`h_t`与当前加过去4帧`[h_t,h_(t−1),...,h_(t−4)]`；公开9维采用对应时刻的同样窗口。每个h均来自相应动作之前的观察，最早为`h_1←obs[0]`。共同目标帧及参考帧均限制t≥5；历史只来自同demo，不补零、padding、未来帧、专家过去动作、方向/阶段标签、demo ID或时间步作为特征。过去帧即使目标为零也保留其观察特征。

重复当前5次的维度对照在历史距离取五lag均值后与当前数学等价，直接alias当前结果，不额外采集或计作第三份独立证据。末20仅为示教时间分层，不自动解释成放置/释放或目标达成前的必要动作。

## 示教隔离、距离与两种固定读出

每模型/空间固定30折，整条query demo从参考库和scaler排除。scaler使用其它29条demo的**全部t≥5当前帧**，含中性/近零目标；float64总体均值/std，std≤1e−6置1。同一scaler用于所有lag、输入模式与读出规则，不按标签、尾窗或结果重新拟合。参考库只取其它demo中t≥5且material的目标行；当前/历史对照的query和参考ID完全相同。

编码器已训练全部30条support示教。留一仅隔离本次近邻和scaler，不是重新训练编码器，也不能称未见示教迁移。参考帧的每个历史窗口同样只用该参考demo的合法过去。

直接float64平方欧氏距离，按64维块累加，query chunk64，不用Gram差以避免重复状态的抵消；history距离为五个对应lag距离的均值。源行顺序固定demo升序、action step升序；等距邻居稳定保留此顺序。五lag距离供两规则及两模式共享，不能重复计为额外CNN调用。

| 读出 | 固定规则及目的 |
|---|---|
| 主：每类5近邻 | 分别取负/正类最近5个参考，比较各自平均平方距离；正类均值严格小才预测正，否则负。两类平均距离完全相等固定判负，保存tie计数。控制每类邻居数量，减少直接多数投票的类别比例影响；仍不完全消除参考密度/覆盖差异。 |
| 参照：普通5NN | 全参考库最近5帧，至少3票正才判正；用于识别主结果是否依赖读出规则，不以其局部改善代替主结果。 |

任一fold任一类不足5参考即停止，不降低k或删fold。已有计数表在扣除每demo前4帧的最坏情况下，所有fold各类下界仍≥906，故该前提在当前数据定义下可由元数据保证。每类的实际query/reference数在首次CNN前由合法缓存标签一次确定并归档，不能据其改变预算、阈值或读出。空query fold仍保存元数据，不补算假样本。

## 指标与判断边界

主报告每task/arm/seed、全部共同帧/末20/补集及逐demo的混淆计数、BA、正/负召回、错误数与分母。先在一个固定Policy内汇总各fold混淆计数再计算BA，不平均或删去单类fold；正负单类/空分层BA为null，仍保留可定义的召回/错误率。三seed均值不得删缺项。

在相同目标帧上对照原Policy第3分量sign；原输出为0时按错误计，不能丢掉。固定原Policy错号帧是次级outcome-stratified分层，不是独立测试。保存全部邻居ID、距离、类别、标签/预测、fold scaler与排除计数。

不另设任意的“容易”准确率门槛，不做p值或正式方法接受。连续分辨力须同时看两类召回和BA，不能用多数类高accuracy掩盖正类失败。两个描述性cue按task/arm分别保留全部三个seed，使用相同类别分母下的整数混淆计数比较，避免浮点微差制造严格改善：

1. 当前主读出相对原Policy：全部共同帧与末20 BA均严格提高，且末20两类召回均不下降，三个seed全部满足才记cue。
2. 主history5相对主current：同一联合条件全部满足才记history cue。缺类/缺项为null，零改善不当严格增加。

cue仅表明该固定读出下的可读信息差别，不能唯一归因为action head或缺历史；主阴性不能被普通5NN或公开9维结果救回。无论结果如何，均不自动启动训练、改loss、选checkpoint、解封评价或更新SPT/Knowledge/Module。本批缓存缺原生目标达成时刻，不能据方向预测确认哪些动作在成功前必需或已解释闭环失败。

## 冻结预算：未来执行额度，本轮均未消耗

精确CNN/几何帧数从已冻结action长度推导。material t≥5精确数未知；用已审计的逐demo计数推导上下界，严禁把上界当实际样本数。每demo前4帧均在补集内；若该demo的material尾窗/补集计数为T/C，则共同帧数在`T+max(0,C−4)`与`T+min(C,L−25)`之间。对应正负类分别计算任意留一fold参考下界。

| task | 原t≥1行/模型 | 共同t≥5行/模型 | material共同帧下界–上界 | 每模型CNN批次 | 末20 material负/正 |
|---|---:|---:|---:|---:|---:|
| 0 | 2895 | 2775 | 2501–2616 | 10 | 269/300 |
| 1 | 4059 | 3939 | 3279–3399 | 14 | 274/285 |
| 3 | 3064 | 2944 | 2568–2688 | 10 | 260/271 |
| 4 | 4442 | 4322 | 3930–4050 | 15 | 334/170 |
| 7 | 4188 | 4068 | 3529–3649 | 14 | 402/75 |

| 项目 | 精确值或明确上界 |
|---|---:|
| 固定checkpoint读取/Policy对象 | 各30 |
| 主CNN / 每模型原首批重复 | 378 / 30，共408批 |
| 主有效预测/表示行；含padding槽 | 111888；115980 |
| 重复有效行；含padding槽 | 9210；9600 |
| 唯一HDF5 / 显式File打开 / source窗口 | 5 / 10 / 1933 |
| 每模态共享源观察/动作行 | 各18648 |
| 未排中性共同帧：共享 / 模型 | 18048 / 108288 |
| material共同帧：共享 / 模型 | ≤16402 / ≤98412 |
| 主模型fold×模式×规则记录 / 次级公共空间记录 | 3600 / 600 |
| 主 / 次级评分输出行 | ≤393648 / ≤65608 |
| 主 / 次级lag距离对 | ≤1603811460 / ≤267301910 |
| 独立核对所选邻居距离 / lag距离 | ≤3444420 / ≤10333260 |
| 独立完整参考库首query哨兵fold | ≤1050，两模式/两规则共用五lag重算 |
| 新训练更新/backward/环境/初态/private/action0/query/sealed | 全部0 |

以上读出记录数包含两模式和两规则的确定性重用，不是独立实验重复。pair上界为每task各demo material上界U_i所给的`(sum U_i)^2−sum U_i²`；主空间乘6模型×5lag，次级乘5lag。全部计算和坐标差上界均写入配置，不通过额外试跑或计时pilot定预算。

未来分三段执行：

1. GPU采集：读旧label/prediction参考35个数组文件；写30模型feature/logit/prediction NPZ、5共享label/ID/公开9维NPZ、30完整首批repeat NPZ，共65个新数组文件。不存原RGB、private或新初态。
2. 释放模型/RGB后CPU近邻：只加载30模型+5共享文件，共35次，写30主probe+5次级probe共35个NPZ，fold scaler/计数另存JSON。普通/平衡、current/history共享距离，不额外采集。
3. 独立审计：加载全部100个新数组文件及35旧参考文件，共135次；重建帧/历史/标签、全部原动作parity、重复、scaler、邻居距离/隔离、计数/指标及每fold首query的全库搜索。不调用Policy/环境/HDF5/checkpoint。

三段总数组文件加载额度205次，历史/新文件清单必须绑定；不把每文件多次打开漏计。精确CNN预算必须attempted/completed一致；label相关评分与距离记录预算是上界，必须记录实际次数和实际material分母。任何技术失败停止并保存partial，不重试或换seed/设备。

## 复现、设备与执行门控

沿用已验证的deterministic pool和原CNN-only配置；不增加真实warm-up或pool前后向探针。所有30份权重在数据payload前核验，只用weights-only加载。每主批输入摘要与原批次对齐，完整旧预测逐位相等；G500汇总abs1e−10/rel0、首批完整padding槽的feature/logit/prediction重复精确、输入/权重不变与gradNone均须通过。audit距离abs1e−8、指标abs1e−10。

未来固定物理GPU1（UUID GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff），逻辑cuda:0，CUDA_VISIBLE_DEVICES=1，确定性开启、TF32关闭，Python3.10.22/Torch2.4.1+cu124/NumPy1.22.4/h5py3.11.0/CUDA12.4，CUBLAS_WORKSPACE_CONFIG=:4096:8，OMP/OPENBLAS/MKL单线程。任何其它compute进程即等待，即使显存<512MiB；同时要求显存≤512MiB、利用率≤5%。等待≤3600秒、每30秒检查，不分配模型/读取payload，不占GPU0或回退CPU采集。

采集活动≤600秒、CPU读出≤2400秒、独立审计≤1200秒；GPU等待另计，外部总体watchdog8100秒（含300秒启动余量）。观察缓存≤2GiB、CPU近邻working≤1GiB。不因资源不足改预算，失败即停止。本轮仅CPU元数据核对，不检查/占用GPU。

**执行入口、方向专用算子、访问guard和独立审计尚未实现/绑定。** 旧`frame_plan/probe`绑定的是夹爪/phase标签和旧18端点，不能直接换标签就运行。允许复用其距离、稳定排序、vote与capture hook逻辑；新实现须按本协议单独验证标签/同demo历史、全demo隔离、类平衡读出、完整预算与角色guard。

元数据验证器在没有完整执行绑定时明确拒绝可执行检查。未来绑定必须包含runner、direction_neighbors、audit_runner、payload_guard、runtime_tests、metadata_protocol、protocol_tests、protocol八种角色的指定路径与SHA，以及本科学配置SHA和完整预算；还须检查全部上下文引用。未绑定前不得消费CNN/HDF5/checkpoint或数组payload预算。

## 本轮验证与下一步

本轮仅冻结JSON/文档、提供元数据预算/绑定门控验证及合成测试。测试覆盖精确CNN与标签相关上界区别、t≥5与尾窗边界、参考类别下界、30端点完整性、role/标签/计算规则漂移、计数重复/缺失/跨seed不一致、禁止伪造执行绑定与无数值runtime导入。实际测试数量、编译/环境检查和1353个已有仓库文件保护核对见freeze收据；README/roadmap/500计划同步科学冻结状态，不写新实验结果。

下一步实现并验证专用执行入口/方向算子/访问guard/独立审计，绑定后再按408批采集及上述CPU读出预算运行。任何科学口径或预算变更均须新版本，不修改本次冻结定义；继续500/5步/现有loss与G停止、封存角色。

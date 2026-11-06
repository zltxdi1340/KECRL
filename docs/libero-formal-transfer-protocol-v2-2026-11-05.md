# LIBERO正式验证方案冻结 v2：独立G/H训练seed（2026-11-05）

**当前有效版本为v2；科学设计已冻结，尚未执行。** v1复用了准入与最终训练seed，存在条件筛选影响最终seed推断的设计风险，已在读取任何封存payload前停止采用。v1文件和metadata核验原样保留，修正记录为`/tmp/libero-formal-transfer-v1-cohort-design-review-20261105-v1.json`。本版使用独立G/H seed及来源生成器，绝不将v1当正式执行依据。 本文件预注册一次固定预算、同族SPT/SPI迁移对照。冻结本身不读取validation/held-out/qualification的payload，不训练、不构造环境、不解码初态。正式运行还必须实现并通过与本设计一致的执行入口、选择性初态读取器和独立审计，完成代码/数据/全部checkpoint绑定。不得把本文件或metadata预检通过写成正式结果。

依据：[理论9.1–9.5](../PROJECT_BIBLE.md)、[既定数据角色](libero-spatial-transfer-split-2026-10-21.md)、[四任务source收尾结果](libero-four-task-abc-completion-2026-11-05.md)。source内部结果尚未证明FOMAML稳定优于普通预训练，因此本方案允许验证失败后停止，不以必须得到正结果为完成条件。

## 1. 主假设和解释范围

主问题：在source经验固定、目标SPI各适应500 updates时，B的闭环控制是否比C和A稳定改善？主指标是**500端点闭环成功率**，不是离线动作MSE或单seed最高成功率。

确认性比较同时包含B−C与B−A，固定在held-out tasks5/8/9上检验。每seed先对三个固定任务的成功率差等权宏平均，再比较20个seed。要求两比较均满足：

1. 跨seed严格胜出概率的单侧精确二项检验，经两比较Holm校正后p≤0.05；
2. 观测到的seed宏平均成功率差≥0.05，即至少5个百分点；
3. 任一held-out任务的跨seed平均差不低于−0.05。

检验的统计对象是**对这三个固定任务、固定初态集合，随机训练重复下严格胜出的概率是否超过一半**，不是总体平均增益的显著性检验。5个百分点是预先设定的实际效应门槛，只对观测平均差作判定。三任务不是从无限任务总体随机抽样，结论限于该benchmark范围；不据此宣称任意未来SPI或其他环境有效。固定500下性能提高可支持有限预算迁移收益，不能独立证明达到同等性能所需更新更少、在线RL样本效率、Knowledge Evolution或Module qualification。

## 2. 任务、数据和初态角色

| 角色 | tasks | 离线更新数据 | 离线独立读出 | 本方案在线初态 |
|---|---|---|---|---|
| source | 0,1,3,4,7 | meta_support0..29；B/C共享更新另用meta_query30..49 | 来源query属于训练反馈，不计独立评价 | candidate_nonregression0..19，仅A/B |
| SPT validation | 2,6 | adaptation_support0..29 | selection_query30..49，仅日志 | candidate_acceptance10..29，A/B/C |
| held-out | 5,8,9 | 验证通过且执行绑定后才读adaptation_support0..29 | final闭环全完成后才读sealed_query30..49 | final_evaluation20..49，A/B/C |

示教编号与official初态编号是不同命名空间，禁止按相同整数配对。source覆盖全部四类几何；最终held-out的movable/table/fixture几何都已在source出现。validation只覆盖table，且task2的center关系未在source出现，这一代表性限制保留，不据验证结果修改划分。

source qualification20..39/reserved40..49、validation checkpoint_selection0..9/reserved30..49、held-out qualification0..19在本研究中保持不读。本方案没有checkpoint选择，用不到validation的0..9。最终评价20..49绝不改名作qualification；后续也不能将final结果重复用作Module资格证据。

## 3. A/B/C和配对

- A：原始未训练上下文生成器，无来源反馈更新；目标SPI仍照常适应500次。
- B：每个source SPI先在support批次上做一步SGD，再用适应后query的一阶梯度，通过原始上下文生成器Jacobian更新隔离候选；不求内循环二阶导数。
- C：相同生成器容量、上下文接口和source批次，平均未适应Policy的support/query梯度做普通pooled BC，无适应后元目标。

每阶段三臂每seed从配对生成器开始；B/C在全部五个source任务上共享窗口ID与顺序。每阶段每seed只训练一组source生成器，供该阶段目标任务复用，不再做留一source探针。G/H的生成器权重、seed和优化器状态全部独立；H在G通过后从新初始化重新做相同的来源更新，禁止复用G候选或筛选后的seed子集。目标任务三臂support窗口与顺序相同，任务之间不共享适应后Policy参数。B/C数据梯度、VJP与生成器更新次数匹配；B多了临时Policy SGD，计算成本单列，不声称全部FLOP或参数写入相等。

共40个全新RNG标识：G固定20261106..20261125，H固定20261126..20261145，两组各20、互不重叠且全部保留。最终统计样本数仍为20，不能将G/H合成40个最终重复；这些整数是RNG标识，不表示训练日期。禁止复用诊断中按效果筛选的模型或seed。生成器初始化的Python/NumPy/Torch/CUDA使用seed；来源torch采样流为seed+1000003，任务按0/1/3/4/7、support再query顺序共享给B/C；目标support采样流为seed+task_id×10000+777，三臂相同。每个在线adapter重设环境seed20261022；PYTHONHASHSEED=0、num_workers=0。正式实现绑定前必须通过这些子流的合成配对测试，不能更改科学设置。

## 4. 算法、端点和loss

保持现有CNN-only双RGB、7维joint、2维gripper和7维连续动作；128像素输入、uint8/255、tanh动作范围[−1,1]。采用确定性等价池化，关闭TF32。

- 每seed的B/C各50次生成器Adam，每次五个source SPI；B每SPI一步SGD，lr0.001；生成器lr0.0003。
- 每个实例适应500次Adam，lr0.0003、betas(0.9,0.999)、eps1e−8、weight_decay0、clip_norm1。
- batch8、sequence10、stride10；仅causal `obs[t−1]→action[t]`、t≥1，无t0重建。
- 均匀七维masked MSE，不加jitter、相机增强、历史头、phase loss、夹爪重权或新学习率计划。
- 固定保存0/50/150/300/500五点，所有闭环只用500。support/query曲线和分量误差是次要证据，不选择端点、不改预算、不救回失败主指标。

## 5. 分阶段执行和接受门槛

**G阶段：来源更新、非退化检查和validation。** 20组source生成器训练完成并审计后，A/B在五个source SPI上各适应500，共200个Policy；A/B/C在两个validation SPI上各适应500，共120个Policy。每个validation任务全部60个support fit完成后，才创建其query dataset。来源query不作为非退化的独立离线分数。

来源0..19的A/B闭环共4000条，validation10..29的三臂闭环共2400条。完整结果和审计齐全后，一次性判定以下条件：

- validation的B跨任务、seed等权平均成功率≥20%；
- validation的B−A、B−C观测平均差均≥5个百分点；两个比较各至少16/20个seed的任务宏平均差严格为正，平局不算胜出；
- 五个source任务分别满足B−A跨seed平均差≥−5个百分点，且每个单seed差≥−10个百分点；
- 所有角色、hash、配对reset、动作接口、预算和数值检查通过。

这些是预注册的候选接受/最终评价准入规则，阈值不是从validation中搜索出来的；非退化门槛只描述当前固定source bank上的观测结果，不声称总体无遗忘。任一条件失败，结束本版本，不读取held-out适应数据、query、qualification或final初态，不追加seed、预算或附近超参数。记录“验证未通过，最终主检验未开展”，不能写成held-out负结果。

通过时，将方法配置准入决定、G阶段A/B/C生成器和全部source/validation结果hash冻结，才允许准备H阶段。G的准入针对训练方法配置，不将某组通过筛选的权重或seed送入最终推断。H在独立seed上重新训练来源生成器，保持算法、数据和预算不变，最终20个seed不受G筛选。G的source非退化是方法校准证据，不代表每个H新权重已通过非退化/资格检查。活动SPT不自动切换；最终数据不能反过来选seed/版本。方法准入、最终科学假设判定和Module qualification是三个不同决定。

**H阶段：锁定的held-out最终比较。** 先在H的20个独立seed上重新做source更新（B/C各50次生成器Adam/seed、B每SPI一步SGD），冻结新生成器；不载入G权重。全部三任务×三臂×20seed共180个Policy各适应500（9万更新），全部端点和训练审计完成后才执行最终闭环。每模型30个final初态20..49，共5400条。全部5400条闭环完成并保存后，才读取sealed_query做固定五点离线日志。全部seed/臂保留，不因qualification未开展或某臂成功率低而删减。

## 6. 总训练和交互预算

| 项目 | G阶段 | 条件通过后的H阶段 | 全部阶段上限 |
|---|---:|---:|---:|
| 适应Policy数 | 320 | 180 | 500 |
| 每Policy适应updates | 500 | 500 | 500 |
| 适应Adam更新 | 160000 | 90000 | 250000 |
| 生成器Adam更新 | 2000 | 2000 | 4000 |
| B临时SGD更新 | 5000 | 5000 | 10000 |
| optimizer更新总数 | 167000 | 97000 | 264000 |
| 闭环episodes | 6400 | 5400 | 11800 |
| 最大Policy动作 | 1920000 | 1620000 | 3540000 |
| 固定稳定化零动作 | 32000 | 27000 | 59000 |
| 最大native step | 1952000 | 1647000 | 3599000 |

来源数据梯度40000次、VJP20000次单列。最多2500份Policy曲线checkpoint、120份生成器端点；共4000个固定离线dataset sweep、最多120000个离线评价前向batch（eval batch32，首尾padding口径固定）。先以授权轨迹长度元数据计算实际批数，超过上限即在模型读出前技术中止，不动态加预算或减少日志点。Policy构造次数、所有前向/反向、有效预测行、训练和渲染耗时也须据实记录，不以optimizer计数替代。

最多500个adapter、230个初态数值行、230个reset reference、11570次精确比较。仅数值解码source100行、validation40行、held-out final90行，数组在同阶段seed/臂之间复用，不重复读取封存角色。全容器SHA读取只作字节校验，与数值解码分开记账。

以近期每360条约30.5分钟的运行成本估算，全部11800条约需16.7小时仿真，另加训练/审计；这是排期估计，不是结果或吞吐保证。G/H每阶段活动wall time上限16小时，超时属于资源中止，不按成绩延长；GPU排队不计研究调用但单列等待时间。G失败后不支出H预算，不替换为另一项实验。

## 7. 统计计算与功效限制

记`t,s,a`的final成功数为`x[t,s,a]`，每格分母30。逐seed差为`d[s]=mean_t((x[t,s,B]−x[t,s,reference])/30)`。任务与seed等权，所有差值用整数/Fraction精确计算，平局不靠浮点epsilon判断。推断单位是20个训练seed，绝不把重复初态的episode当成20×3×30个独立统计单位。

每比较令`K=#(d[s]>0)`，平局计非胜出，始终n=20；检验`H0:Pr(d>0)≤0.5`，单侧p为`sum(comb(20,j),j=K..20)/2^20`。这不是删平局后的传统sign test，也不是平均效应检验。G的准入事件与H独立训练seed分离，不用通过筛选的同一批模型做最终随机重复推断。独立训练RNG重复、共同任务/数据条件下稳定的seed结果分布是推断假设；不声称固定任务bank中的场景是独立随机任务总体样本。

B−C与B−A构成唯一确认性检验族，以Holm控制familywise α0.05；次要分量和逐task分析仅描述，不选择显著项。另为两比较分别报告严格seed胜出概率的单侧97.5% Clopper–Pearson下界区间`[L,1]`，Bonferroni保证该概率区间族至少95%覆盖；不是任务成功率或平均差的置信区间。报告所有seed差、观测mean/median/sample SD、逐task成功数和配对四格。

离散检验核算：在n=10、首个Holm门槛0.025时至少9胜；若真实seed胜出概率0.8，单项通过概率约0.37581。在n=20时至少15胜，单项约0.80421。因此固定20个新seed。该计算只针对胜出方向的单项检验；两比较联合、5个百分点实际效应和逐task门槛会降低总体通过率，不是对5个百分点效应作80%功效保证。三个历史seed即使全胜，单侧最小p=0.125，不能提供本方案所需确认性结论。

统计方法依据：[SciPy二项检验定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)、[R官方Holm说明](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)。对少量RL训练重复需考虑不确定性的动机见[Agarwal等，NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)；本方案的胜出概率统计与具体阈值由本研究预先定义，不冒充该论文推荐的唯一方法。

## 8. 接受、停止和负结果

- 技术检查失败：中止，保留partial、尝试/完成计数和实际费用；不得将不可判定episode悄悄计失败或剔除分母。技术失败没有有效主检验结果。
- G接受门槛失败：保持原始活动SPT，留出数据继续封存，本配置结束；不能把验证失败说成最终检验失败。
- G通过：方法准入决定先冻结，H在独立seed重新训练source后做所有最终比较；不复用G权重，不以最终结果选checkpoint、预算或版本。
- H完整但两比较任一校正检验、5个百分点门槛或逐task非退化门槛失败：主假设不获支持，原始数据全部报告；不代表证明方法普遍无效，也不追加数据直到显著。
- H通过：只支持上述固定预算/任务范围内的预注册比较结论。活动SPT、Knowledge和Module均无自动写入权限。

完整H前无基于效果的早停。无效数据、非有限输出、hash/role/预算/reset错位或资源不足立即技术中止。重试/更改协议必须另建版本并保留历史读取账本；已经打开的final集合不能重新宣称未见。

## 9. Module qualification边界

本方案qualification预算为0，held-out0..19和source20..39保持封存，Module注册关闭。当前SPI实现仍标注`qualification_defined=False`，benchmark success不满足完整资格契约，不能更改这一事实。

后续资格契约的最低操作门槛预先写明：每个固定Policy至少20次有效资格episode、经验目标成功率≥0.80；启动/保持条件、目标可检测性、资源/契约、适用范围及执行预算都必须可判定且通过。任一硬契约违反判失败，unknown/不足保留未决；高成功率不能抵消硬违反。该经验门槛只针对明确固定bank范围，不给连续初态总体的置信保证。实际启用还需独立冻结完整检测器、scope、Policy选择规则、样本数/置信方法与执行manifest；本方案既不支出资格预算，也不能据此注册Module。

## 10. 设备、provenance和执行缺口

物理GPU1、进程cuda:0/EGL1，GPU0不占用；启动前无计算进程、显存≤512MiB、利用率≤5%。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、h5py3.11.0、robosuite1.4.0、MuJoCo3.2.3；LIBERO revision和数据revision由原manifest固定。单线程、CUBLAS `:4096:8`、确定性、TF32关闭、tensor cache≤8GiB。

每个产物记录协议/代码/输入hash、seed与RNG子流、真实optimizer参数和更新次数、runtime/device/git/版本、checkpoint的文件及tensor hash、访问角色/行号、环境seed/初态全局与局部索引、动作/零动作/native步数、终止原因、stage耗时、峰值内存和失败provenance。只记录公开action、joint/gripper和reset观测摘要，不记录private live state或初态数值。

尚需完成的执行工作是：支持任意合法连续初态行区间的选择性读取；注入局部索引时保留全局provenance；强制G决定与hash绑定后才能读H；目标训练/query gate和数值桥接；独立预算、轨迹、统计重算审计。metadata预检不冒充这些工作已实现。冻结配置、统计决定函数和本协议hash先固定，执行代码绑定完成前不打开任何封存payload。

## 11. 文件和下一步

[科学配置](../configs/libero-formal-transfer-protocol_v2.json)、[冻结绑定](../configs/libero-formal-transfer-freeze-binding_v2.json)、[纯metadata/统计入口](../experiments/libero_formal_transfer_protocol_v2.py)、[测试](../tests/test_libero_formal_transfer_protocol_v2.py)。冻结验证证据另见[核验记录](libero-formal-transfer-freeze-v2-verification-2026-11-05.md)。

下一步实现并独立核对正式执行入口，绑定所有可执行依赖，先做不接触真实封存数据的合成角色/行区间/统计测试和metadata预检。执行绑定通过后方可按既定阶段启动G；本次冻结没有启动G或H。

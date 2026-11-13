# task7固定当前表示的连续动作可纠正性协议（2026-11-08）

**本轮冻结科学方案，尚未执行。** [上一轮误差对齐](libero-task7-policy-error-focus-alignment-results-2026-11-08.md)已确认原Policy正目标末窗有51–75/75帧错号，多数反向输出已有幅度；共同评分集负焦点误差较参照高，但末窗5/6负焦点无错号。近邻标签变化不能直接指导加历史或改loss。因此本次检验：**保持表示与原Policy不变，当前表示是否可读出能修正第3分量的连续残差；相对于仅作常数偏置修正，是否还有额外收益且不恶化负方向。**

这是source-support内部、给定已暴露表示的辅助读出诊断。不是新Policy适应、编码器未见泛化、SPT/SPI正式迁移、Knowledge Evolution或Module qualification；专家动作也不是唯一成功控制。无论cue真假均不自动部署读出、追加闭环或恢复G/H。原Policy维持500端点、未来稳定化5步、现有loss，G失败/H未运行及正式评价封存保持。

## 固定端点、缓存与数据角色

[新科学配置](../configs/libero-task7-current-residual-correctability-diagnostic_v1.json)与[stdlib元数据门控](../experiments/libero_task7_current_residual_correctability_protocol.py)固定task7、seed20261106/07/08、A_original/B_fomaml六个现有500端点。A/B是已有特征来源，不能与本次原输出/偏置/仿射三种读出混称，也不是早期baseline/jitter。原CNN-only当前256维是已经融合当前输入的表示，不把它称作纯视觉或历史表示。

未来主执行、独立审计各读取同7份现有NPZ：shared中的targets、demo_indices、action_steps、demo_lengths、tail20五字段；六份features.npz中的features、predictions两字段。逐项固定文件SHA、checkpoint文件/weight身份、capture/G500/legacy来源。不打开checkpoint、不解码logits/public proprio/RGB/HDF5，不读旧NN标签CSV、距离、其它读出缓存或新观察。

source meta_support demos0..29、因果obs[t−1]→action[t]、t≥1共有4188共享行。整demo三折沿用既有固定表示诊断的0..9/10..19/20..29轮作readout_probe，另20条readout_fit；按demo/step重建source_row。

| fold | probe demos | fit t≥1行 | probe t≥1行 | probe t≥5行 |
|---|---|---:|---:|---:|
| 0 | 0..9 | 2782 | 1406 | 1366 |
| 1 | 10..19 | 2817 | 1371 | 1331 |
| 2 | 20..29 | 2777 | 1411 | 1371 |

每帧只作为新读出probe一次，合并三折后每模型/读出/快照完整4188行；三折不是独立seed。两个子角色都属于support，不是meta_query或官方validation/held-out。编码器和原Policy此前已用全部30条示教训练，冻结缓存解码也物理包含probe行，不能声称probe字节未读或表示未见。scaler统计、梯度及采样只允许fit行；训练阶段仅标准化fit行并按任务释放临时张量，全部36头500步完成后才对全缓存标准化/快照读出并评分probe，避免同时缓存18份标准化大张量。

source meta_query、validation、held-out、qualification、reserved、final evaluation payload全部封存。既有已消费的14次动作误差分析额度关闭；本次未来读取预算单独计费。

## 三种读出与唯一新增机制

设已保存原七维动作为p0、当前表示为h，fit-only标准化后为z。本次只改辅助读出的action index2，不能将其解释为物理世界z位移。

| 读出 | 第3分量残差δ | 可训练参数 | 作用 |
|---|---|---:|---|
| original | 0 | 0 | 固定原Policy输出参照，无新前向/训练 |
| bias_only | b | 1 | 检查常数偏置修正是否足够 |
| current_affine | Wz+b | 257 | 检查当前表示带来的条件修正是否超过常数偏置 |

两种辅助头都W/b全0起步，初始化全七维预测数值精确等于p0；不使用随机残差、另选初始化或旧读出权重。输出p2=clip(p0[2]+δ,−1,1)，其它六维逐位复制p0；在fit和评分中使用同一clip。梯度仅进入新W/b，不载入/更新原Policy或生成器。其它六维目标误差仍计入均匀七维MSE，不能暗中把loss改为单轴或方向加权；它们对辅助参数是常数项，第3分量梯度仍带1/7因子。

偏置与仿射的有效参数量和优化自由度不同。相同数据、预算和零初始输出不消除这一差别，不能把结果解释成action head独立因果贡献。current_affine若额外有效，仅支持这个有界读出可利用给定当前表示；失败也不能证明表示没有信息或必须加历史。

每端点/折使用fit当前特征的float64均值、population std，std≤1e−6的scale置1，其余取原std；变换后转float32，两个头共用元数据。无标签归一化、phase/历史/专家动作历史/NN标签输入，不改gripper、采样权重或方向权重。所有4188行中只以fit索引进行监督训练。

## 配对优化、预算和评分门控

每辅助头固定500次Adam，batch64，有放回采样。lr0.0003、betas(0.9,0.999)、eps1e−8、weight_decay0、clip norm1；amsgrad/maximize/foreach/fused/capturable/differentiable均False。实际参数组、参数量、500次loss、裁剪前后梯度norm、Adam步数和最终m/v必须保存并核对，不能只抄配置。裁剪后norm的浮点技术余量固定1e−6，不用于科学退化护栏。

PCG64(seed+100000×fold)从升序fit source_rows采样500×64绝对行号。三seed×三折共9张唯一采样表，同seed/折的A/B和两个辅助头使用同表；18个端点×折配对任务、36个辅助头，合计**18000辅助optimizer更新和backward**。这不是原Policy追加18000次适应；原Policy新增更新为0，本轮方案冻结实际辅助更新也为0。

训练时仅保存0/100/250/500权重快照、逐步fit loss和norm，不做probe评分或全缓存快照读出。全部36头精确到500且fit-only核对通过后，再回溯四个固定快照，按头/快照作144次全4188行读出，记录fit/probe曲线。始终以500为唯一主端点，不按曲线选点、早停、调lr/预算、换fold/seed或重采样。中途异常按技术停止规则保留partial，无重试。

| 未来计算 | 精确额度 |
|---|---:|
| 辅助头 / 各头updates | 36 / 500 |
| optimizer / backward / 训练前向 | 各18000 |
| 训练帧访问 | 1152000 |
| 唯一采样表 / RNG行抽取 | 9 / 288000 |
| fit-only scaler / fit行访问 | 18 / 50256 |
| scaler fit特征坐标 | 12865536 |
| 主训练fit标准化行 / 坐标 | 50256 / 12865536 |
| 全缓存标准化行 / 坐标 | 75384 / 19298304 |
| 主阶段两次标准化合计行 / 坐标 | 125640 / 32163840 |
| 主 / 审计快照读出批次 | 144 / 144 |
| 每阶段快照读出行 / 七维预测坐标 | 603072 / 4221504 |
| OOF三读出×四快照逻辑评分行 | 301536 |
| final500 OOF逐帧输出行 | 75384 |
| 原CNN/Policy前向、训练、环境、近邻搜索 | 0 |

审计不重新训练，不把审计核对说成独立复现18000步优化。它独立验证fit/probe角色、scaler、精确采样表、实际optimizer设置/步数、loss/norm有限性/裁剪边界、最终状态和保存权重，重算全部快照预测、曲线、OOF、类别、参照、决策及不变分量。

## 指标、固定参照与解释

t≥5共同集合4068行，material集合3598行（负2273/正1325）；末20步material负402/正75，neutral470个共同帧保留在总指标中。四个时间层为all_source(t≥1)、all_common(t≥5)、tail20每demo最后20个动作、complement共同集合去末20。tail20不是物理接近/释放阶段；all_source与共同集合分母不混用。

各时间层均报告all、negative(y2≤−0.025)、positive(y2≥0.025)、neutral(|y2|<0.025)。主动作指标为第3分量MSE和material_sign_error_rate：material目标上严格反号或预测恰为0计错，neutral组分母0/错误率null；不得用epsilon重标。另保存六种严格错号/零/同号不足/过大/精确相等/零目标非零输出类别和SSE贡献、MAE/bias/目标能量比、幅值分布，以及七轴、平移、其它平移、旋转和夹爪指标；clip前残差、超界裁剪与边界输出另报。

每个更新点固定三差：bias−original、current−original、current−bias，同一OOF帧/目标；0步必须无差。全部30个demo保留，空组计数/SSE0、率/MSE null；报告逐demo差、非空demo等权均值及方向数。主判据固定用合并OOF帧权重，demo均值为构成敏感性描述，反向结果明确报告，不改权重替换主判据。全部三seed逐一报告和完整均值，按A/B分开，fold不作为独立重复，不做p值/显著性或合并环境成功率。

## 500端点的描述性cue及停止规则

每模型对每个固定比较，须同时满足：

1. 正目标all_common及tail20的MSE、material错号率均严格低于参照；正目标complement两项均不增加。
2. 负目标all_common、tail20、complement的MSE、material错号率均不增加。
3. all_common全部帧的七维MSE不增加，其余六分量在全部帧/快照逐位相同。
4. 所需值非空/有限，final500 material共同帧的候选及学习参照预测符号对NumPy重建稳定，见下述数值门。

阈值用直接float64差<0或≤0，不设“改善足够大”、容忍退化百分比、显著性或事后门槛。每个A/B来源分别报告三seed全部同时满足的bias_vs_original cue；current的额外cue须三个seed同时通过current_vs_original及current_vs_bias。任何seed/护栏失败，该cue为false，不能用均值补救、挑fold、挑seed或延长训练。

cue只决定是否保留下一机制线索，不是正式接受/Policy替换/新闭环许可。bias有效而current无额外cue仅保留常数校准线索；current额外cue支持表示上的条件残差可读出性；正改善但负退化保留取舍并停止本候选；两者均失败结束这项500预算对照，不能认定历史必要、原head有错或表示不充分。

## 新的文件/字段预算与独立审计

每阶段源NPZ为7次/17字段：共享5＋6×(features,predictions)。主写36份新头归档，独立审计每份只加载一次，19字段：fit_rows、probe_rows、fit_demo_indices、probe_demo_indices、sample_rows、scaler_mean、scaler_scale、updates、weights、bias、predictions、step_loss、gradient_norm_before_clip、gradient_norm_after_clip、optimizer_steps、final_adam_m_weight、final_adam_v_weight、final_adam_m_bias、final_adam_v_bias。bias_only归档的weight/对应Adam数组均为零占位，不能冒充256个被训练参数。

两阶段源文件共14次/34字段，另审计新归档36次/684字段；**总NPZ加载50次、字段解码718次**，输入标签CSV遍历0。新归档只写/tmp，不入仓库；旧缓存/结果/原绑定不修改。每头predictions为4×4188×7的float32，weights为4×256、bias为4；fit/probe/绝对采样ID及fit-only统计随归档固定。训练头/档案全36保留，不只存cue通过者。

七CSV精确行数：curves4608、oof-groups1152、oof-demo-groups34560、contrasts1152、demo-contrasts34560、oof-frames75384、legacy-parity18，总151434行；OOF逐帧表仅final500，其它快照保存在头归档/汇总。curves=6端点×3折×2头×4更新×2角色×4时间层×4方向，其它分组保留original基线；seed JSON汇总384行、模型比较18组、arm联合cue4组。全源legacy复现每阶段216标量（六模型×all/tail20/complement×12），计数/夹爪sign严格一致。

CPU Python3.10.22、NumPy1.22.4、Torch2.4.1+cu124但CUDA隐藏，PYTHONHASHSEED0、BLAS/OMP单线程，Torch确定性。主/审计各≤180秒，总watchdog420秒；当前地址空间VmHWM/VmRSS≤512MiB，CSV≤256MiB、新头NPZ≤128MiB，流式表输出，技术重试0。配置、源码七角色与依赖、父证据、端点/数据SHA、软件/设备、种子、实际预算、起止UTC/PID/成本及异常都须记录；环境seed/初态不适用。

独立审计另进程，NumPy数值实现不导入主实现或Torch。按保存float32头预测计算严格类别、指标和cue，float64指标abs1e−10/rel0、ID/计数/标签及其它六维逐位核对；NumPy从权重/scaler/p0重建第3分量允许abs1e−5/rel0。记录重建符号分歧及abs(保存输出)<1e−5计数，不改零/错号定义。若final500 material共同OOF帧的候选或学习参照存在符号分歧，对应cue数值门为false；不放宽容差或自动声称稳定符号改善。初始输出数值相同、全部其它六维位级不变单独验证。

SHA只作字节核对，与数值加载分别记账；源文件阶段前后、新头/CSV审计前后不变。预算、字段/ID/角色/scaler/采样、父证据、optimizer、有限性、复现或其它审计校验失败即停止，保留partial，无覆盖/修旧缓存/技术重试。主pending-audit结果不回写，audit/completion另存。

## 本轮状态与下一步

本轮只读取源码/当前文档、固定父JSON元数据及七源文件字节SHA，创建协议/科学配置、stdlib门控与合成测试；真实缓存解码、辅助/原Policy训练、前向和环境调用均0，无这项对照的新结果。科学配置固定本协议与门控/测试源码SHA，验证和保护证据保存于/tmp/libero-task7-current-residual-correctability-freeze-20261108-v1.json；README/roadmap/500计划同步。

下一步实现辅助训练入口、7源文件/36新头字段guard、fit/probe阶段门控及独立审计，完成合成测试并绑定七角色代码/依赖后，再按本额度执行。未绑定必须拒绝真实读取/辅助拟合。本轮不创建执行绑定或启动训练，正式评价数据继续封存。

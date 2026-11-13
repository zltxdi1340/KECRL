# task7原动作误差保护与纠正保留配对诊断方案（2026-11-08）

**本轮只冻结新假设、对照、预算和停止规则；没有启动训练。** [上一轮目标匹配](libero-task7-target-magnitude-matching-results-2026-11-08.md)的精确共同支持域很小，没有三seed共同焦点格，不能判定loss根因。同时，“仍符号正确”的参照组也可能有较大幅值误差，不能直接把较小原输出作为加权依据。本方案检验另一项可拒绝假设：**给辅助读出的新增专家误差加惩罚，能否减少对原本较准动作的损害，同时保留基线已经实现的正方向纠正。** 成功与失败都只解释为这个固定预算辅助目标的诊断，不是原Policy、SPT/SPI正式结果、Knowledge Evolution或Module qualification。

[科学配置](../configs/libero-task7-residual-harm-protection-diagnostic_v1.json)和[stdlib元数据门控](../experiments/libero_task7_residual_harm_protection_protocol.py)固定task7、seed20261106/07/08、A_original/B_fomaml六个原Policy500端点。A/B是特征来源，下面λ=0/1是每个来源内部的损失对照，不能混称或当六个独立seed。原Policy与CNN不更新，未来稳定化5步、原Policy现有loss、G失败/H未运行和全部source-query/正式评价封存保持；四个父joint cue=false不回写。

## 唯一训练变量与对照

两臂都用已保存current256融合表示，fit-only标准化后为z，辅助残差δ=Wz+b，均257个参数、W/b全0起步。第3分量p2=clip(p0[2]+δ,−1,1)，其它六维逐位复制原输出p0；不加新视觉/历史/动作/phase输入，不把index2解释为世界z或夹爪。原输出p0为固定参照，无新原Policy前向。

| 模式 | 新训练 | 目标 | 可训练参数 |
|---|---|---|---:|
| original | 无 | 固定保存p0 | 0 |
| uniform_affine | 500 updates | 现有均匀七维MSE，λ=0 | 257 |
| harm_protected_affine | 500 updates | 同一MSE＋新增误差惩罚，λ=1 | 257 |

对fit批次中每帧定义e0=(p0[2]−y[2])²、e=(p2−y[2])²，B=64：

L0 = Σ_i Σ_j (p_ij−y_ij)² / (7B)

P = Σ_i ReLU(e_i−e0_i) / (7B)

Lλ = L0 + λP，**只比较预定λ=0与λ=1**。

P保留1/7梯度尺度，没有改成单轴MSE；其它六维常数误差仍在L0中。专家误差改善或相等时P=0；变差时加强该帧第3分量梯度。规定ReLU在0处子梯度为0；不使用未经验证的clamp_min零点梯度。λ=0直接使用历史L0张量和backward路径，不以“加0×penalty”替代；两臂初始预测相同、初始P及附加梯度为0，首步数值相同须在执行绑定前用真实Torch合成梯度测试验证。记录λ=0的惩罚遥测时使用detach/no_grad，不改变其训练图。

这不是限制|δ|≤|p0|：必要的越零纠正仍可发生。连续误差保护不保证符号纠正，所以保留独立符号和同帧纠正护栏。没有方向、phase、gripper、目标尺度归一化、采样或输出幅值权重，也不根据匹配覆盖率选择训练样本。y和e0仅在fit损失及全部fit完成后的合法评分中使用；推理仍只需p0/z/W/b，不向控制器提供专家目标、原动作好坏标签或oracle。

## 数据角色、配对与500端点

仍仅source meta_support demos0..29、因果obs[t−1]→action[t]、t≥1共4188行。整demo三折分别以0..9、10..19、20..29为readout_probe，另20条readout_fit；fit/probe行数为2782/1406、2817/1371、2777/1411。编码器已见全部30示教，读取缓存也物理包含probe行，不能声称probe字节未读或未见编码器泛化。

沿用历史fit-only float64均值/population std、std≤1e−6置scale1、标准化转float32；同端点/折两臂共用scaler。PCG64(seed+100000×fold)从升序fit source_rows有放回抽500×64，同seed/折的A/B与λ两臂共用9张采样表。全部fit t≥1包括neutral进入均匀采样和损失，“原本较准”mask不参与训练抽样或梯度。

实际Adam lr0.0003、betas(0.9,0.999)、eps1e−8、weight_decay0、clip norm1；amsgrad/maximize/foreach/fused/capturable/differentiable全False。只新建36个辅助头，各500次forward/backward/step，共18000更新、1152000训练帧访问；原Policy/生成器更新0。两臂零初始输出、参数数目、优化器、批次、数据角色、dtype/设备和clip都一致，只有λ改变。

训练时只保存0/100/250/500权重与fit遥测。全部36头精确到500、fit-only检查完成后，才允许probe变换、专家好坏分组和快照评分；没有提前probe读出、早停或中间选点。四个快照只作诊断曲线，500是唯一主端点。

## “原本较准”的固定定义与计分

material仍为abs(y2)≥0.025。仅在最终评分中按原p0/y固定：

- original_good：material且e0<y2²，即第3分量比零动作更接近专家，严格不等式。
- original_bad：material中其余帧。
- original_wrong：material上p0严格反号或恰0，是original_bad子集。
- all_material：good与bad的互斥完备并集。

这是相对动作误差定义，不是绝对高精度、任务成功或小输出阈值；不使用候选输出或上一轮少量匹配帧定义分组。original_good与original_wrong互斥，三模式成员完全相同；保存每模型3598共同material帧的membership。neutral在总MSE和训练中保留，不进入这些material分组。

时间层和方向沿用all_source(t≥1)、all_common(t≥5)、末20及共同补集，all/negative/positive/neutral；保护和保留主判据仅all_common、tail20、complement三个t≥5层。末20不是物理阶段。分组主要量为h=max(e−e0,0)的均值/和、harmed_n(e>e0)、MSE及同帧新增/解除损害；报告剩余损害，不将组级下降称为每帧完全无损。七轴、严格六种类别、全局MSE/错号及其它六维不变仍完整报告。

主判据固定为每模型拼接三折OOF后的帧加权统计；全部30条示教含空组/null完整保留，示教等权仅描述反例，不切换主权重。每来源保留三seed原值、均值/中位数和改善/持平/恶化数，fold与重复帧不视为独立seed；不进行p值、置信区间、正式显著性或因果推断。

## 对照有效性、联合线索与停止规则

**对照有效性先于科学结论。** 新鲜λ=0必须逐位复现历史18个current_affine头的14字段：updates、weights、bias、sample_rows、scaler_mean/scale、step_loss、梯度norm前/后、optimizer_steps、最终Adam四个m/v数组。比较全部四个权重快照和500步流，不比较新旧NPZ文件字节（归档字段/标识不同）。历史头只作审计参照，不能初始化任一新头。复现不通过属于技术停止，不能把它解释为新loss效果。

每个task×seed×特征来源的候选联合线索须同时满足以下固定规则，直接使用保存float32预测上的float64统计，没有科学epsilon或容忍退化百分比：

1. **相对原输出的纠正与护栏**：正目标all_common/末20的MSE和material错号率均严格更低；正补集及负目标三个层的两项均不增加；all_common全帧七维MSE不增加。
2. **相对新鲜λ=0的非回归**：三个层、两个material方向的MSE/错号率均不增加，all_common全帧七维MSE不增加。
3. **保留同一批纠正**：正方向各层，λ=0已将原p0错号纠正的帧，λ=1丢失数必须0，且纠正总数不减少；不能用新增纠正抵消丢失旧纠正。基线纠正集空时保留率null、标记无此项证据，但规则1仍要求common/末20真实改善。负方向同帧保留也报告，不用其替换正方向主规则。
4. **保护固定original_good**：负方向common和末20要求分组n>0。若λ=0在该组h均值>0，λ=1必须严格降低h；若基线h=0则保持0。每模型这两个预定层至少一个有基线h>0，否则标记“无可检验损害”、cue=false。
5. 所有非空original_good组、三个层及两方向，λ=1的h均值、harmed_n和MSE均不高于λ=0，且**相对λ=0新增被损坏的原good帧数=0**。正方向或负补集原good组确实为空时，输出null并记not_applicable，不当保护证据；负common/末20为空则主cue=false。成员/计数不一致、缺字段或正n下非有限值不能借空组规则跳过。
6. 全部其它六分量逐位不变；历史λ=0复现通过；最终500的严格符号及original_good上的“e>e0”判定在独立重建中无分歧。否则数值门false，不改符号或放宽阈值。

每个A/B来源分别要求三个seed全部通过；共6个模型候选记录、2个来源联合cue。任一seed/护栏失败或支持域不足，该cue为false，按500终点结束本候选。不得挑seed/fold、调λ、改good阈值/权重、选100/250点或加预算。即使cue=true也只是保留下一机制线索，不自动替换Policy、启动闭环、恢复G/H、更新SPT/Knowledge或注册Module。

技术与科学停止分开：来源SHA、角色、字段、optimizer、shape/有限性、步数、预算、loss遥测/历史对照或审计错误，停止保留partial、真实重试0；来源有效但联合规则不满足，是这项固定λ=1假设的诊断结束，不反推表示缺信息、历史必需或loss已是根因。

## 新读取与计算预算

本项LIBERO数据预检只读10份父JSON元数据及相关源码/报告，**不读取或哈希下面任何LIBERO研究NPZ或CSV**。开工仓库完整性快照另含72份旧Crafter结果CSV和1份smoke CSV的文件级摘要，未作数值分析；它们不参与本实验，最终检查不再次哈希这些CSV。以下零读取数字只指本项LIBERO来源/历史头，不能扩写成整个工作区没有CSV字节哈希。未来另绑定代码才开放：

| 未来数值读取 | NPZ次数 | 成员解码 |
|---|---:|---:|
| 主：shared5字段＋6份features/predictions各2字段 | 7 | 17 |
| 审计：同7源 | 7 | 17 |
| 审计：36个新头×27字段 | 36 | 972 |
| 审计：18个历史仿射头×14参照字段 | 18 | 252 |
| 合计 | **68** | **1258** |

输入研究CSV遍历0，不读取历史bias_only头或未列字段。25份源/历史NPZ的字节SHA在未来fit前检查可用性并前后核对，字节SHA与数值解码分开计费；原Policy checkpoint只保留JSON身份，不打开或哈希。原来的50次缓存及只读CSV额度已关闭，本次未来预算单独记录。

新头沿用19字段，并加8项：step_base_loss、step_penalty_loss（未乘λ，含1/7）、step_active_harm_n、step_original_component3_mse、step_candidate_component3_mse、step_weights_before(500×256)、step_bias_before(500)、step_train_prediction2(500×64)。每步沿用同一训练前向输出，只记录fit批次，不额外训练前向。保存loss/norm/500步Adam和固定快照，全部36头不选择性归档，产物留/tmp。

| 未来计算 | 预算 |
|---|---:|
| 头 / 每头updates / 总optimizer与backward | 36 / 500 / 各18000 |
| 训练帧访问 / 唯一采样表 / RNG抽取 | 1152000 / 9 / 288000 |
| fit-only scaler / fit行访问 / 特征坐标 | 18 / 50256 / 12865536 |
| 主fit标准化＋全缓存标准化行 / 坐标 | 125640 / 32163840 |
| 主、审计固定快照读出 | 各144批、各603072行 |
| 审计逐步线性/损失重建 | 18000批、1152000行、294912000特征坐标 |
| 审计线性重建合计 | 18144批、1755072行 |
| 原Policy/CNN/生成器/环境/近邻/封存payload | 0 |

14个主CSV固定191994行：curves4608、oof-groups1152、oof-demo-groups34560、contrasts1152、demo-contrasts34560、oof-frames75384、legacy-parity18、protection-groups432、protection-demo-groups12960、protection-contrasts144、protection-demo-contrasts4320、original-group-membership21588、correction-retention36、correction-retention-demos1080。OOF逐帧和保护/保留表只记500端点，原始曲线保留四快照；所有30 demo和空组保留。另384份全seed曲线JSON、6候选flags、2来源cue和18历史头逐字段复现记录。

CPU Python3.10.22/NumPy1.22.4/Torch2.4.1+cu124、CUDA隐藏、单线程、Torch确定性，沿用已有Adam eager数值路径并固定本地依赖SHA。主/审计活动各≤180秒，总watchdog≤420秒，VmHWM/VmRSS≤512MiB，CSV≤256MiB、新头NPZ≤128MiB、禁止覆盖/真实重试0。未来训练前仍检查GPU占用并记录CPU设备/软件、配置/代码/端点/文件SHA、seed、实际attempted/completed更新/前向/行/解码、UTC/PID/时间/RSS；无环境seed或初态索引。

## 独立审计和执行前门控

审计另进程、NumPy实现，不导入主训练数值实现或Torch，不重新训练18000步。独立重建fit/probe、scaler、9张精确采样表，验证实际Adam参数/500步/状态/有限loss/norm及角色日志；用每步保存的更新前W/b、fit索引重建线性输出，与保存训练prediction2比较abs1e−5。再从保存float32训练预测和原p0/y重算base/penalty/total、活跃损坏计数（float32逐元素，计数严格）、loss abs1e−6；不将其说成独立重演全部优化器或梯度轨迹。

144批固定快照从权重/scaler/p0独立重建，保存float32预测的float64统计abs1e−10/rel0，预测重建abs1e−5；ID/角色/历史14字段及其它六维位级严格。最终common material符号或original_good损坏判定若有重建分歧，相关cue数值门false。主pending文件不回写，审计/完成另存，最终cue以独立审计/完成记录为准。

实现入口、27/14字段guard、fit/probe阶段门控、独立审计和合成测试后先绑定七角色及传递依赖。真实执行前须验证ReLU零点梯度、1/7尺度、改善不罚/恶化加罚、两臂零初始化和首步相同、λ=0历史算子复现、原good分组/空集规则、同帧纠正丢失不能靠总数补救及所有预算。未绑定必须拒绝研究归档字节哈希、数值读取或辅助训练。

## 本轮状态与下一步

新增172项合成规则测试、相关七个协议文件合计530项测试全部通过，两个新Python文件编译通过，check_environment确认Python3.10.22/Torch2.4.1+cu124；CUDA按本轮CPU设置主动隐藏。只跑元数据/合成回归，没有重跑全量pytest或真实Torch训练核，不能把这些通过当作数值执行已验证。日志为/tmp/libero-task7-residual-harm-protection-new-tests-20261108-v1.log、focused-tests及environment同前缀日志；执行前仍须单独完成上述Torch算子与真实历史复现检查。

本轮仅创建科学配置、方案、stdlib门控及合成规则测试；没有新训练结果、没有读取/哈希本项LIBERO来源数组、CSV或历史头，没有新辅助/原Policy更新或原CNN/环境调用；仓库快照中的73份其它历史CSV摘要按上述单列。实际验证和完整性证据保存于/tmp/libero-task7-residual-harm-protection-freeze-20261108-v1.json，README/roadmap/500计划同步。下一步实现入口、成员/角色guard和独立审计，合成验证并绑定后才可按本额度运行；本轮不创建执行绑定或启动训练。

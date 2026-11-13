# task7保护损失后的修正几何与同帧转移只读方案（2026-11-08）

**本轮另冻只读分析，未读取/哈希研究CSV，未执行新机制分析或训练。** [上一轮结果](libero-task7-residual-harm-protection-results-2026-11-08.md)已证明固定λ=1联合假设未通过：负good平均H六模型降低，但各模型丢失10–25条正纠正且新增86–326条原good受损。这里不恢复候选，而是明确哪些动作变化贡献了收益、哪些带来了代价。

[科学配置](../configs/libero-task7-protection-residual-decomposition-readonly_v1.json)、[stdlib元数据门控](../experiments/libero_task7_protection_residual_decomposition_protocol.py)及合成数学/预算测试固定范围。task7、seed20261106/07/08、A_original/B_fomaml六个原Policy500端点及λ=0/1辅助500端点保持。A/B是特征来源，下面B、C分别指λ=0、λ=1的保存动作，p0为保存原输出；不可混称。仍只有source meta_support demos0..29，编码器已见全部30示教；不是未见泛化、完整SPI/SPT结果或Module qualification。

## 问题与已知数学边界

每帧第3动作分量记专家y、原p0、基线B、候选C，e0=(p0−y)²，eB=(B−y)²，eC=(C−y)²，hB=max(eB−e0,0)，hC同理。只讨论归一化动作index2，不把它解释为世界z、夹爪或物理阶段。

若C在p0与B的闭区间内，它是两端点的凸组合，平方误差凸性保证eC≤max(e0,eB)，所以**hC≤hB**。尤其B未损坏而C损坏的帧必定在该区间外。这也适用于每帧采用不同缩小比例的情况，不只适用于全局常数比例。上一轮已报告新增损害，因此“候选全部只是向原输出缩小”已被已有摘要与该数学约束排除；这不是本轮新逐帧结果。

同时，缩小仍可能丢失纠正：例如y=0.25、p0=−0.25、B=0.125、C=−0.0625，C在区间内且eC<e0，hB=hC=0，但B正确而C错号。平均H减少不能保证符号纠正保留。新分析据此区分两类代价，不预先认定“整体保守化”、偏置或视觉/历史是根因。

## 输入和严格配对

唯一数值CSV为本轮`/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/oof-frames.csv`：75384行=6模型×3模式×4188个t≥1源行，已有SHA为`6c8e67ea6cabc17278442eca14969991e33dcd66d7166d0899c114a7d0be5010`，本轮仅从已审计JSON登记该摘要，不重新哈希CSV。

未来每阶段只一遍，投影17列：task、seed、arm、mode、update、source_row、probe_fold、demo、action_step、target_direction、tail20、component3_category、raw_residual、clip、target_2、prediction_2、error_2。完整35列表头固定；其它18列会随CSV词法切分出现，但不得数值转换，不声称这些列的字节未读。父6份JSON只用于provenance与固定1548项摘要parity，不作为新指标或分组选择依据。

先核对全部25128个三mode源行组（包括t1..4），同task/seed/arm/source_row/update500，目标、demo、step、fold、direction、tail完全一致，无重复/缺失。原mode raw=0/clip=false；预测和目标为精确保存float32，float64重算error2/六类，按float64 raw+p0→clip→float32重建输出/clip标记。各模型共享专家目标及source布局。分析仅t≥5的4068帧/模型，输出24408条配对记录，含neutral；all_common、末20、共同补集固定，末20不是物理阶段。

禁止其它来源CSV（包括curves/protection等其它13表、旧实验CSV）、NPZ/NPY/头/checkpoint/HDF5、RGB/特征/本体/私有状态及它们的字节SHA；不作新前向、线性头读出、回归拟合、counterfactual Policy或环境调用。source-query、validation、held-out、qualification、reserved、final封存。

## 原始修正与实际修正分开

raw为保存的裁剪前残差r；effective为保存float32动作与原输出之差d=B−p0或C−p0，运算转float64。对两种表示分别作互斥完备七分类：

| 分类 | 基线→候选 |
| --- | --- |
| zero_stays | 0→0 |
| zero_to_nonzero | 0→非0 |
| to_zero | 非0→0 |
| same_sign_smaller | 同号、幅值严格减小 |
| same_sign_equal | 同号、幅值相等 |
| same_sign_larger | 同号、幅值严格增大 |
| sign_reversal | 两个非0修正反向，不因幅值较小而算缩小 |

严格“向p0缩小”只包括to_zero/same_sign_smaller；闭区间还包含不变情况。全部判定无epsilon。raw与effective各自报告，不从raw缩小直接推出实际动作缩小：例如p0=0.875、rB=0.5、rC=0.25，两输出都裁剪为1，raw缩小但effective不变。

报告全部帧及各时间/方向层的计数、比例、幅值与变化的均值/中位数/线性q25/q75、零基线覆盖率、clip与符号margin。比值dC/dB仅在dB≠0且结果有限时报告；零分母/溢出为null并分开计数，不填0，不按比值挑样本或加权。报告基线幅值分布，不能用近零分母下的均值比值定义总体保守化。无需拟合全局α或搜索阈值。

## 固定人群、四格与精确分账

original_good仍为abs(y)≥0.025且e0<y²，表示比零动作更准；original_wrong为material上原输出反号或0，两组互斥。成员只由p0/y定义，不按候选结果或上一轮焦点筛输入。

1. **全部帧的movement面板**：三时间层、all/negative/positive/neutral，raw/effective七类及总行，保留共同完整分母。
2. **原good的harm面板**：两material方向，按B/C是否e>e0分为未损坏保持、新增、解除、仍损坏四格；与七种effective运动交叉。报告n、e/h/SSE/MAE、符号margin/错误、Δe和Δh，区分“仍损坏但程度减轻”和帧数减少。
3. **原wrong的correction面板**：两方向，按是否符号严格正确分为两者均未纠正、候选独有、丢失、同帧保留四格；与七种effective运动交叉，明确丢失纠正是否来自向原输出缩小或区间外变化。

每个单元既报告单元内均值，也报告sum(Δh)/整个原good人群n等对总体均值变化的贡献；分子可正可负，不把贡献除以很小的净变化生成“因果百分比”。四转移×七类的计数、SSE和Δh贡献须加和为完整人群；all/total行仅作守恒参照，不能重复加总。逐帧严格核对闭区间⇒hC≤hB，新增损害⇒区间外，违反数必须0，否则技术停止。

所有30示教与空单元保留（空单元内均值/率null，计数/和0；完整人群n>0时空单元贡献0/n=0，人群n=0时贡献null）；主分析是每模型帧加权，示教等权仅作带覆盖率的描述，不切换主权重。标量求和/均值固定math.fsum，线性分位数先排序，再按相邻两端点权重插值，避免大比值下不同归约顺序的伪差。两来源分别保留三个seed原值、完整均值/中位数和方向数，缺一个seed支持则完整均值null；不把fold、重复帧或时间层当独立seed，不作p值、置信区间或因果推断。按结果分组的差异可能由定义机械产生，不能反推训练机制或输入信息缺失。

## 预算、验证和停止

未来主分析/独立审计各1遍75384行，共2遍150768词法行；完整35列词法切分共5276880列值，17列投影共2563056列值。每阶段25128组源triplet、24408共同帧、48816次raw/effective分类；父摘要parity1548字段/阶段。仅一份当前CSV，其余研究CSV/数组/checkpoint哈希和读取、新训练/前向/环境/近邻/封存调用全部0。

| 主输出CSV | 固定行数 |
| --- | ---: |
| paired-frames | 24408 |
| movement-summary / movement-demo-summary | 1152 / 34560 |
| harm-decomposition / harm-demo-decomposition | 1440 / 43200 |
| correction-decomposition / correction-demo-decomposition | 1440 / 43200 |
| invariants | 18 |
| 合计 | 149418 |

movement含7类＋总行，harm/correction含4四格＋总行并交叉7类＋总行；逐demo表完整乘30。另1344份三seed摘要（movement384、harm480、correction480），不输出新的候选接受cue。

CPU Python3.10.22/NumPy1.22.4、CUDA隐藏、单线程，主/审计活动各≤180秒，总watchdog≤420秒，VmHWM/VmRSS≤512MiB，CSV≤256MiB，真实重试0、禁止覆盖。先实现入口、CSV字段投影guard和独立审计，合成测试与七角色/传递代码哈希绑定后，才可读取或哈希真实CSV。审计另进程且不导入主数值实现，独立核对三mode配对、float32/裁剪、七类、四格、贡献守恒、凸性、1548项父摘要parity和所有产物。统计abs容差1e−10/rel0仅用于复算比较，ID/float32 bits/类别/符号/几何不变量均严格，不放宽科学判定。

任何provenance/SHA、表头/投影/角色/缺失/重复、数值/裁剪/分类、预算/分账/凸性或独立审计错误，停止并保留partial，不重试或补读其它来源。即使大部分修正缩小，也完整报告放大/反向/激活和反例，不凭单均值认定根因，不恢复父6模型或A/B联合cue。原Policy500、辅助500端点、未来稳定化5步/原loss及既定λ0/1保持；G失败/H未运行、旧4与新2 cue=false、SPT/Knowledge/Module不变。

## 本轮交付与下一步

新增112项合成数学/元数据测试、相关八个协议文件合计642项测试通过；两个新Python文件编译、环境检查通过，CUDA为本轮CPU设置主动隐藏。没有重跑全量pytest或真实CSV数值分析；上述通过验证的是方案、数学边界和未绑定拒绝，不是新增逐帧机制结果。日志在/tmp/libero-task7-protection-residual-decomposition-{new-tests,focused-tests,environment}-20261108-v*.log。

本轮仅冻结配置、方案、stdlib门控及合成数学/协议测试；真实CSV字节SHA、数值投影、训练、前向和环境调用均0。实际验证与完整性记录见`/tmp/libero-task7-protection-residual-decomposition-freeze-20261108-v1.json`，进度同步README/roadmap/500计划。下一步实现入口、字段guard和独立审计，完成代码绑定后按唯一新额度执行；本轮没有新的逐帧机制结论。

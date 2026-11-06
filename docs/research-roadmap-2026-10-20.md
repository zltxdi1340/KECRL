# KECRL 研究路线冻结（2026-10-20）

几何/接触记录核对完成（2026-11-06）：[字段、证据与下一步](libero-formal-G-geometry-record-availability-audit-2026-11-06.md)。G逐步归档只有动作及joint/gripper，没有末端、物体、接触或RGB帧；历史专家ee_pos及task4/7图像不能拼接成当前36对的几何真值，因此尚不能检验close与实际接近是否协调。本次0原轨迹重读/训练/前向/环境/私有状态，115项重点测试、1648项CPU回归通过（不含可选Crafter收集项）。保持500/5步/现有loss、G阴性停止及封存角色；下一步先另行设计来源candidate的有限几何观测与时钟对齐，尚未冻结或采集，不追加validation或训练。

36对较晚close反例分段完成（2026-11-06）：[只读结果与事件邻域](libero-formal-G-close-runs-late36-diagnostic-2026-11-06.md)。31对输家首段close仍持续至赢家首次close，4对为已结束的单步脉冲，1对为已结束的两步段；首次多步段口径下赢家仍较晚33对、较早2对、同步1对。30条赢家首段因成功终止右删失，不能用观察长度决定保持时长。72条旧轨迹、254段、858行邻域，84项相关测试及15904项独立表级核对通过；原始第二遍复核无完成记录，未计通过、未重试，详见报告。0新训练/Policy前向/环境，G未通过及held-out/qualification封存保持。下一步先明确close与接近物体的几何协调假设和合法观测需求，现有joint/gripper不能替代目标/接触证据，不立即改loss或重训。

validation B/C 800对只读轨迹诊断完成（2026-11-06）：[完整结果与反例](libero-formal-G-validation-BC-trace-diagnostic-2026-11-06.md)。1600条公开轨迹校验通过；相同reset下800对首步平移动作已不同，162对成败不同中155对前20步夹爪符号始终相同。赢家较早close较常见，但36对较晚、9对同一步；83对open因提前终止不可判序，264/326条成功轨迹终止前没有close后的open命令。93对赢家前20步joint路径更短、69对更长，不能直接统一提前close或抑制运动。39项新增、109项相关测试与66010项独立检查通过；0新训练/Policy前向/环境重放，G未通过及held-out/qualification封存保持。后续36对close段与事件邻域已完成，见上方新报告；原G结果保持。

G任务/seed只读诊断完成（2026-11-06）：[分析与全部seed表](libero-formal-G-task-seed-differences-2026-11-06.md)。保留全部20个seed，task2的B−C为7胜/3平/10负、task6为9胜/9平/2负；只有4个seed一任务输一任务赢，3个两任务都赢，均值抵消不是统一根因。17个source护栏违反分布在12个seed；验证表现存在组间差异，也保留违反组5个B−A正向及未违反组1个负向反例。固定专家MSE与闭环方向混合，不据此改loss或加预算。31项新增、80项相关测试及21318项独立核对通过；0新预测/训练/环境/初态，G未通过与H/qualification封存保持。后续800对公开轨迹诊断已完成，见上方新报告，原G结果保持。

G阶段完整复核、准入未通过（2026-11-06）：[最终结果与停止依据](libero-formal-G-results-2026-11-06.md)。320个Policy各500次适应、6400条闭环齐全，原始及本次独立训练/完整审计均通过。validation任务2/6的A/B/C等权成功率为21.125%/20.375%/20.375%；B−A为−0.75个百分点、B−C为0，严格胜出11/20和10/20，均未达预设+5个百分点及16/20门槛。source另有17/100个task×seed配对退化超过10个百分点。累计活动约8小时3分37秒，167000次优化更新及固定预算完成；原审计字段故障与修复记录保留，0重复source训练。按冻结规则结束当前配置，H主检验未开展，held-out/qualification/reserved继续封存；不追加训练、不接受活动SPT/Knowledge/Module。后续只读任务/seed解释已完成，见上方新诊断；原准入和停止决定保持。

以下为启动前冻结快照，实际执行进度以上方记录为准。

正式验证方案v2已冻结（2026-11-05）：[当前有效协议](libero-formal-transfer-protocol-v2-2026-11-05.md)修正v1复用准入/最终训练seed的条件筛选风险；v1未执行且原样保留。G/H各20个独立seed及全新来源生成器，保持每Policy500和5步稳定化；先source非退化+validation2/6，准入通过才做held-out5/8/9。H以20个训练重复作单侧精确胜出概率检验+Holmα0.05，另要求B−A/B−C观测平均增益≥5个百分点和逐task退化护栏。全部阶段上限500适应Policy、250000适应+4000生成器+10000临时SGD=264000更新，6400+5400条闭环。80项相关、1459项CPU回归及32组独立数学核对通过；仅metadata预检，0研究训练/环境/封存payload。执行入口尚待实现/绑定；qualification预算0，不自动写入SPT/Knowledge/Module。


四任务A/B/C收尾完成（2026-11-05）：[完整结果](libero-four-task-abc-completion-2026-11-05.md)已补齐task4/7的18个500-update Policy及360条fresh闭环，两份独立审计通过；task0=8/12/12; task1=0/11/1; task4=0/1/1; task7=0/0/0（A/B/C，各60条）。维持5步稳定化/现有loss，task0/1几何覆盖与task4/7几何外推分层，不合并总成功率或挑seed。48项训练相关、133项闭环相关、1351项CPU回归及12项汇总测试通过；已交付36条seed端点、24行四格及360行曲线CSV。本周source内部验证收尾，正式validation/held-out/qualification仍封存；下一步冻结正式方法/预算/统计与停止规则，不自动接受SPT/Knowledge/Module或继续零散调参。



本路线将 SPT/SPI、持续任务流、Knowledge Evolution 和最终联合验证拆成逐级实验目标。各环境只承担它能够支持的主张，不把不同环境的成功率拼接为一个总分。

最新只读横向整理（2026-11-05）：[四个source task现有成绩](libero-task0-task1-task4-task7-results-comparison-2026-11-05.md)已核对task0/1 A/B/C与task4/7 BC/jitter两条实验线，统一展示500端点、三seed、candidate0..19及5步稳定化下各自成绩，同时保留来源更新与执行栈差异。task0/1为8/12/12及0/11/1，task4/7 BC/jitter为3/3及4/3（各60条）；不计算混合总分或难度排名，不把缺项填0。6个原始结果hash及逐episode聚合校验通过，0新训练/环境。正式四task方法对比需另冻补齐设计并交代task4唯一开放容器几何，原观测诊断与全部封存角色保持原状态。

最新表示探针结果（2026-11-05）：[同候选头历史输入敏感性](libero-task1-b-history-input-sensitivity-results-2026-11-05.md)按60批151080行完成；主条件三seed历史MSE成本/动作能量均略低，两个幅度同向，未支持读出端这一类表示扰动下历史更脆弱。固定线性头logit增量一致，事件/初始open局部方向混合，不能当真实RGB/本体或部署shift结论。4221928项独立检查和63项相关专项通过，0新训练/CNN/环境。下一步建议转原始公开观测层另冻单分量小窗口和明确CNN预算；500/5步/loss及旧停止、正式封存角色保持。

前置敏感性冻结（2026-11-05，执行结果见上）：[固定权重历史输入表示诊断](libero-task1-b-history-input-sensitivity-diagnostic-2026-11-05.md)锁定三seed、同候选500头、全部source-query2518个完整历史帧；0.01/0.05幅度、固定正负成对方向、当前/过去/全窗口对照，主比较全窗口0.05的相对干净平移误差成本。身份、零扰动、过去路由和同线性头logit门槛共同冻结，60批151080行预算；58项相关测试与metadata通过，研究调用0。表示探针不替代真实观测shift检验，下一步执行并独立审计后再决定训练修改，500/5步/loss及正式封存边界保持。

最新专家读出结果（2026-11-04）：[同头真实历史/重复当前目标误差](libero-task1-b-fixed-head-expert-target-error-results-2026-11-04.md)按6批15108行完成，独立重建7554配对、15个分层×模型和60个demo×模型指标。真实历史整体平移/旋转/七维均更准；重复当前平移均值+30.74%、七维+12.47%，但已知open段平移/七维保留反向收益。497380项独立检查、85项相关测试通过，0新训练/CNN/环境，未发现需改冻结代码的失配。专家MSE与此前闭环4次seed24收益不可互相替代；500/5步/现有loss和旧停止保持，下一机制建议检验历史对观测分布变化的敏感性，封存角色保持。

前置目标误差协议（2026-11-04，执行结果见上）：[同一候选500头真实历史/重复当前](libero-task1-b-fixed-head-expert-target-error-diagnostic-2026-11-04.md)已冻结source task1三seed、全部2518个t>=5合法query专家帧/seed，主判断三seed平移MSE是否均严格降低，并记录旋转/七维/逐demo取舍及误差方向—能量分解。预算当前3批7554行+历史3批7554行，无CNN/训练/环境；77项相关测试和metadata预检通过，研究读出及缓存数值解码0。下一步按冻结入口完成读出及独立审计，不把输出分叉当准确性；500/5步/现有loss及正式封存角色保持。

前序受控闭环结果（2026-11-04）：[同一候选权重双输入120条结果](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)已完成，真实历史0/60、重复当前4/60，新增成功仅seed24；三seed改善条件未通过。60/60对第5步运动动作分叉而夹爪相同，53对后续出现符号差异；输入可改变部分控制结果，但不能概括稳定收益或失败唯一根因。35470 Policy动作+600零动作、625.70s，1147101项独立检查及77项相关专项通过，0新训练。下一步建议另冻同头两输入的专家目标误差对照；500/5步/现有loss、旧停止决定与全部正式封存角色保持。

前置冻结快照（2026-11-04，执行结果见上）：[同权重双输入120条配对闭环](libero-task1-b-fixed-head-input-closed-loop-diagnostic-2026-11-04.md)已固定task1 B三seed500端点、真实历史/重复当前运动路由、共同前4步和5步稳定化，代码/配置绑定与metadata预检通过。两臂共120条fresh、最多36000 Policy动作+600零动作，0训练；目前研究前向/环境/初态数值解码均0。65项相关专项及1212项CPU回归通过，下一步按冻结入口执行并独立审计，不把第5步代数差预写成闭环根因，旧停止决定和封存角色保持。

最新机制读出（2026-11-04）：[第5步固定候选头交叉读出](libero-task1-b-step5-crossed-readout-diagnostic-2026-11-04.md)完成全部60个已用source场景，3批60行新增H(C)、6批120行已有格重放。内容项L2在50/60大于权重项，seed23仅10/20；seed22权重与内容相抵，不能将范数相加或投影称因果占比。夹爪内容严格0，数值桥接及280781项独立检查、66项相关测试通过，0新CNN/训练/环境。当前500/5步/loss和候选停止保持；下一建议另冻同权重双输入fresh配对闭环，检验计算差异是否改变控制，不据此直接接受候选或打开正式角色，后续已独立冻结（见上），尚未执行。

最新只读分解（2026-11-04）：[全部60对路由闭环轨迹](libero-task1-b-routed-paired-traces-diagnostic-2026-11-04.md)保留57共同失败和3个baseline-only，完整120份JSON与17489同钟帧。两臂第5步共同观察下开始运动映射分歧，前20步夹爪符号相同；三例close候选晚2/晚166/早1，低joint距离也有丢失成功，不能统一调close/release或惩罚位移。57项相关测试及558511项独立复核通过，0研究前向/训练/环境/NPZ读取；继续500/5步/现有loss和路由停止决定。下一项建议先另冻第5步固定候选运动头的历史内容/已学权重代数参照，尚未读取特征或执行新读出，不据此开放正式评价或接受SPT/Knowledge/Module。

最新控制核对（2026-11-04）：[固定路由读出120条配对闭环](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)全部完成，baseline3/60、候选0/60，三seed2→0/0→0/1→0；该配置的离线平移/七维收益未转为控制收益，停止候选推进。前置[runtime接口](libero-task1-b-routed-runtime-interface-diagnostic-2026-11-04.md)通过，全部100次reset/240次共享启动动作/300次前缀观察相同；无技术失败或权重变化。639.92s、35489个Policy动作+600个零动作，1149项CPU回归及1184391项独立检查通过。下一步只读已有全部60对，单列3个baseline-only，定位第5步之后的动作与close/drift分歧，不据此断言记忆根因或改loss/预算；500/5步、正式封存角色及SPT/Knowledge/Module边界保持。

最新配对训练（2026-11-04）：[运动6维历史／夹爪当前读出](libero-task1-b-routed-history-readout-diagnostic-2026-11-04.md)按新冻结协议完成task1 B三seed、整demo三折和全support共24×500=12000更新，12个baseline逐位复现。支持OOF和已暴露source-query均通过新路由条件：query平移/七维均值下降11.66%/5.14%，事件错误仍28/69、完整open139/696，夹爪切换未改善。178项相关测试、4052292项独立检查通过；审计标签错误另存修订，不改训练。CPU9.49s，原CNN/Policy更新/环境0。前置[固定头交叉读出](libero-task1-b-crossed-history-readout-diagnostic-2026-11-04.md)只支持内容/权重的代数机制线索，不证明物理因果。下一步冻结runtime历史接口及启动4步规则后核对，再另冻candidate闭环；500/5步/现有loss、旧完整历史停止决定与封存角色保持，不更新SPT/Knowledge或Module。

最新只读机制分解（2026-11-04）：[task1 B query margin/时序](libero-task1-b-query-event-margin-timing-diagnostic-2026-11-04.md)保留全部23事件×3seed及完整命令段，6新增/4纠正、35都对/24都错。事件夹爪MSE降低主要来自仍错误的输出幅度改善；新增错误有lag1/3/5恢复和命令边界未恢复，不能概括为近0一帧噪声。58项相关测试及59399项独立检查通过，0新增模型/训练/源数据/feature调用。不推进历史候选闭环，500/5步/现有loss保持；下一项应先冻结固定权重下的缓存表示受控读出，分开历史内容与已学权重变化，不以代数分解宣称物理因果。正式角色与SPT/Knowledge/Module边界保持。

最新source内部检查（2026-11-04）：[task1 B完整support/query读出](libero-task1-b-support-query-history-readout-diagnostic-2026-11-04.md)实际完成6×500新读出更新和CPU query特征采集，原编码器冻结，26.13s、98项相关测试及独立审计通过。query切换错误三seed11→9、7→9、10→12，虽然七维/平移及完整open段都改善，事件主条件失败，停止该配置进入闭环，不挑seed或放宽门槛。query数据在所有fit与数值桥接后读取，属于已有诊断暴露的source meta-query，不是正式未见评价。下一项只读既有23事件的连续输出/符号与时序矛盾，不追加训练；原500/5步/loss、SPT/Knowledge/Module边界及封存角色保持。

最新实际训练（2026-11-04）：[固定表示监督读出](libero-learned-phase-history-readout-diagnostic-2026-11-04.md)以source-support内部整demo三折分开新读出fit/probe，原编码器已暴露全部support。108个读出各500 updates、共54000更新，25.65s完成；整体七维/平移在18/18端点改善，只有task1 B三seed事件改善且其他主指标不增加（1/6组），事件23/23/21→22/20/19，task0平均事件错误反而增加。91项专项/训练回归和45029项独立检查通过，0完整Policy前向/原Policy更新/环境。保持原500/5步/loss和全部封存角色，不把条件读出学习当必要记忆或闭环修复。下一项仅保留task1 B线索，完整support拟合/source-query须另冻，不选最好折或直接更新活动SPT。

最新分解（2026-11-04）：[已保存切换近邻](libero-phase-history-transition-neighbors-diagnostic-2026-11-04.md)完成全部585个模型×事件，93个变差、76个改善，121/195个arm×事件的三seed类别混合。整体BA的非切换帧贡献可覆盖切换损失；变差组新增close邻居多位于自身open前1..4步，改善组也有相反交换。26440个逐lag距离和10175个交换配对通过独立核对，仅解释固定距离规则，不识别真实Policy的必要记忆。v1状态门槛错误在数组读取前停止并保留，另存v2按同定义完成；72项相关测试与165968项独立检查通过，0新前向/训练/环境/全库搜索。保持500/5步/现有loss，后续训练须先冻同结构同预算的固定表示读出对照，不直接加平滑、改loss或接受SPT；封存角色保持。

最新机制诊断（2026-11-04）：[阶段／历史可辨性结果](libero-source-internal-phase-history-observability-diagnostic-2026-11-04.md)已按819主前向+18重复完成全部18个固定500端点读出。加入当前及过去4帧后总体命令近邻BA在18/18提高，条件phase BA在17/18提高；只有task1 B/C满足三seed联合条件（2/6组），task0三臂平均切换错误反而增加。条件phase标签仅区分expert-open上下文，不代表识别何时open；编码器已暴露于全部support，不能称未见泛化或记忆根因。56项相关测试及199200条近邻行、2400次完整库搜索独立复核通过，0训练/环境调用。下一步只读本轮事件近邻和原Policy margin解释取舍；保持500/5步/现有loss、角色封存，不直接触发训练或接受SPT。

夜间修正与复核（2026-11-04）：[完整open段错误复核](libero-source-internal-overnight-error-review-and-full-open-results-2026-11-04.md)已执行65个既定事件的全段跟踪，1755模型×事件段及540初始前缀保留，0新增研究调用。未发现需重训的实现数值失配；原局部15处连接细分为13延长、1桥接、1示教末端未解，另2处孤立。task0/1完整段错误650/3555、1326/4131，整段未首次正确20/51；初始open与切换open、首次响应与持续保持、未恢复和重入分开。259项专项/回归、150436项独立检查通过，500/5步/现有loss不变，不采用回退输入。后续机制转向阶段/历史可辨性，未证实记忆根因、不自动训练或接受SPT，封存角色保持。

前序只读定位（2026-11-04）：[新增错号margin与时序](libero-source-internal-gripper-sign-margin-temporal-diagnostic-2026-11-04.md)完成原5508个模型×帧和1170个事件窗复核，0新研究调用。主17处margin范围0.000803–0.031598，lag3/4/5分别1/4/12；可见窗内单帧新增，15处连接既有错close段、2处孤立。次级五处均为主条件子集；原错仍错457/963帧保持，margin范围重叠，不能阈值分型或当物理原因。12处新增右截断，后续全部65事件完整段跟踪及范围更正现已完成，见上方；继续500/5步/现有loss。78项专项、574940项独立标量核对通过，封存角色保持。

前序受控读出（2026-11-04）：[固定Policy局部gripper对照](libero-source-internal-gripper-state-local-control-diagnostic-2026-11-04.md)完成2457主前向+54重复，18个500端点62586行原预测精确复现，全部未处理/padding/sham及重复通过。事件锚点和lag1各18模型的局部平移/夹爪MSE均增加、符号错误率无模型下降；主条件17个模型×帧由正确变错、0纠正，六组主cue全部false。混合观察的结果不支持该输入回退修复，不归为物理反馈、视觉或head单因。90项专项、605232项独立标量核对通过，41.77s、0新训练/环境；保持500/5步/现有loss。其后margin与时序只读核对已完成，见上方，不打开封存角色。

前序设计冻结（2026-11-03）：[输入敏感性复核与局部gripper受控读出](libero-source-internal-input-sensitivity-review-and-gripper-state-control-protocol-2026-11-03.md)已完成历史hash/判断核对，旧视觉敏感性与gripper幅值收益不足以支持改训练。新设计固定18个500端点、task0/1全部149/157个open后expert-open帧，仅替换两维公开gripper为事件pre-action或lag1原始过去状态，所有原正确/错误标签保留。预算2457主前向+54重复，48项专项与独立样本/预算审计通过；冻结时实际新研究前向、HDF5/RGB/checkpoint、训练/环境调用0。该执行前快照保留，后续执行已完成，见上方结果；继续500/5步/现有loss，已停止分支和封存角色保持。

前序均值参照核对（2026-11-03）：[非切换帧固定参照](libero-source-internal-nontransition-translation-reference-diagnostic-2026-11-03.md)完成全部18模型、两种参照、384组原同帧对照，0新增前向/训练/源数据调用。open后B调整均值+0.047595/+0.081137，同专家符号参照+0.049245/+0.064848；两task三臂均保留正均值，不由简单加性均值差消除。其他阶段反号保留，未排除所有整体拟合混杂或定位因果。40项专项、482110项独立检查通过；保持500/5步及现有loss。该阶段建议的输入审计及单分量设计冻结已完成，见上方，新读出尚未执行。

前序只读定位（2026-11-03）：[事件前后/错号方向/同帧平移](libero-source-internal-event-phase-direction-translation-diagnostic-2026-11-03.md)复用原NPZ，0新模型/训练/环境/HDF5等源数据调用。open后expert open预测close与较高平移误差在两task三臂均有同帧跨seed正均值，B为+0.052950/+0.070214；事件当步和close之后存在反例，不当统一因果机制。原窗口全部由t>=2事件覆盖，unresolved0；42项专项和191092项独立检查通过。保持500、5步和封存角色，不改loss或接受SPT；该阶段建议的非切换帧参照核对已完成，见上方最新进度。

前序逐帧诊断（2026-11-03）：[固定18个500端点的专家动作读出](libero-source-internal-fixed500-expert-frame-target-error-diagnostic-2026-11-03.md)按预先冻结819+18批次完成，0训练/环境交互。三臂两task的open窗口四项主误差均在三seed高于补集，close窗口夹爪误差也集中；早期专家目标全open、预测符号全正确，仍无runtime首动作目标证据。B的open符号错误占比平均58.28%/54.09%，两种错号方向依seed反转，不能统一重加权。61项专项、294770项独立复核通过，保留action0未归档的标签复核范围说明。保持500/5步和全部封存角色，该阶段建议的事件前后只读分解已完成，见上方最新进度，未新增前向或改loss。

前序只读分解（2026-11-03）：[source专家输入0/500点](libero-source-internal-initialization-endpoint-target-error-diagnostic-2026-11-03.md)已冻结并完成，复用历史36组同support缓存指标，新增前向/训练/数据payload调用0。B-A初始平移/夹爪优势在适应后平均缩小，B-C初始夹爪六组均更差、最终仅各task的2/3 seed更低；支持集误差与闭环结果存在反向案例。代数分解不识别独立因果贡献，不据此改loss、延预算或接受SPT。59项相关测试、2930项独立检查通过，保持500与5步、封存角色。后续逐帧预算与结果见上方最新进度。

前序只读分析（2026-11-03）：[source内部六组配对轨迹](libero-source-internal-six-group-paired-trace-diagnostic-2026-11-03.md)完成360条唯一episode的全部正反分层，主240对、次级360对均保留。差异首步已出现，但没有统一close时序/持续长度或公开位移规则：task0 seed22 B-only相对A主要见A缺少close，task1 seed24的11个B-only则逐对比A/C更晚close；同命令时序也有相反结果。32项相关测试、813项CPU回归与独立133166项数值检查通过，0新模型/训练/交互/数据payload调用。该阶段不改loss/增强或接受SPT，保持500端点；后续0/500目标误差分解见上，不开放封存任务或资格初态。

最新闭环结果（2026-11-03）：[source内部A/B/C闭环](libero-source-internal-spt-spi-closed-loop-diagnostic-2026-11-03.md)完成全部360 episodes并独立复核，原配置和绑定不变。task0 A/B/C成功8/12/12（各60条），task1为0/11/1；B合计收益集中在个别seed，两个task的三seed严格改善规则均未满足。99457个Policy动作+1800个零动作、320次reset精确比较、18份权重及360份轨迹均通过，耗时1851.77s、792项CPU回归通过。没有新训练、封存数据访问或候选接受。下一步只读六个task×seed组全部正反配对轨迹解释seed敏感性，继续500端点和5步；不由诊断合计成功数推进Module qualification或完整KECRL主张。

冻结阶段记录（2026-11-03）：[source内部SPT/SPI配对闭环协议](libero-source-internal-spt-spi-closed-loop-protocol-2026-11-03.md)在执行前冻结18个500端点和360 episodes，5步稳定化、环境seed20261022。初态前20行的选择性读取与40个reset参照/320次精确比较不放宽。冻结阶段74项专项和781项CPU回归通过、实际交互0；该快照不改写，后续执行结果见上。validation、held-out、qualification保持封存。

最新主线更新（2026-11-03）：按用户同意的推进计划，[夹爪动作映射核对](libero-gripper-action-mapping-audit-2026-11-03.md)收束幅值机制诊断，转入[source内部SPT/SPI可行性对照](libero-source-internal-spt-spi-diagnostic-2026-11-03.md)。task0/1轮流排除于当前fold的source生成器更新，A原始/B上下文FOMAML/C匹配数据梯度与容量的普通预训练，三seed、18个探针Policy各500 updates；source/探针query角色严格分开。B-A曲线面积两探针各3/3 seed降低，端点方向与B-C比较仍不稳定，未接受候选版本。v1 CUDA确定性中止及v2资源拒绝单列，数学等价池化v2在GPU1完成原预算，108份权重/180组指标复核与739项CPU回归通过。新增环境交互0；下一步先冻同18个端点的source candidate闭环对照，不因离线指标开放validation、held-out、qualification或Module注册。

最新只读结果（2026-11-03）：[task7缓存分量取舍定位](libero-task7-expert-camera-component-tradeoff-diagnostic-2026-11-03.md)完成三个seed和四个弱干预的命令/窗口/示教分解。共同局部模式是expert close且远离切换窗口时连续夹爪读出更偏离+1；平移仅seed24的全局均值改善，不能称普遍优化冲突。重叠窗口分区和有符号全局贡献核对通过，保留全部30条及不利参照；新模型/训练/交互/HDF5调用0，690项CPU回归通过。保持500 updates和闭环5步，下一项先核对现有公开夹爪动作映射及既有接口审计，不把离线幅值误差直接当物理闭合不足或触发训练。

最新执行结果（2026-11-02）：[专家相机—目标误差诊断](libero-task4-task7-expert-camera-target-error-diagnostic-2026-11-02.md)完成7200主批次+48重复、208656行预测并独立复核。三seed弱干预在平移MSE上方向不一致，两个相机的主flag均未通过；完整空间信息损失probe均有退化，但受纹理/分布外与强度限制，不证明几何必要性。局部七维flag与分量取舍单列，不升级主结果。原科学配置/算子hash不变，0训练/环境，677项CPU回归通过。保持baseline500、闭环5步；下一项只读已有分层查task7平移与夹爪误差取舍，不追加模型前向、访问query/资格或自动恢复增强训练。

前序设计快照（2026-11-02）：[合法专家输入的相机—目标误差对照](libero-task4-task7-expert-camera-target-error-protocol-2026-11-02.md)已冻结研究设计、算子和判断规则，尚未运行，执行绑定待完成。保留六个baseline500、闭环默认5步；只评估source-support的8630个causal`t>=1`输入，不对离线输入额外稳定化。八条件将matched外观/空间扰动与目标误差方向、响应能量分开；不由输入响应大小证明几何必要性或重启增强。声明未来7200主前向+48重复、0新训练/环境，本轮只有元数据与合成算子验证。下一步完成runner与绑定，不提前访问query/qualification或主实验。

前序固定Policy实验（2026-11-02）：[source缓存输入分解](libero-task4-task7-source-stabilization-input-decomposition-diagnostic-2026-11-02.md)已完成并独立复核。固定六个baseline500，在60对既有source0/5首观察上读出八组合；三seed、两参照均为RGB响应大于本体、眼在手大于外部相机，没有夹爪符号变化。这定位当前source输入替换的响应关系，不是目标误差、六组闭环根因或增强训练依据。48主前向+12重复、1800行预测、0新训练/环境/HDF5读取，649项CPU回归通过；保持5步与500 updates。后续目标误差设计已冻结，见上方最新协议；不重新搜索增强强度或提前进入主实验。

前序只读结果（2026-11-02）：[120对稳定化轨迹机制对齐](libero-task4-task7-stabilization-trace-alignment-diagnostic-2026-11-02.md)已完成，六组相反结果逐对检查。首动作、close早晚及持续长度没有统一成功方向；12对首步夹爪符号不同者中11对仍共同失败。时间轴区分Policy序号与共同物理时钟，结束后的状态/事件不补造。保留5步、500 updates及两臂各7/120的源结论；新增研究训练/交互为0，独立复算与631项CPU回归通过。其后source缓存输入分解已完成，见上方最新实验；不把输入敏感性或六组结果选择当闭环因果证据。

前序闭环结果（2026-11-02）：[当前栈下stabilization0/5成功率对照](libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic-2026-11-02.md)完成240条fresh episode并独立复核，0/5均为7/120。task4零步2/60、五步3/60；task7零步5/60、五步4/60，均无三seed一致零步改善。保留现行5步和500 updates，其后120对只读分析已完成，见上方最新报告，不把输入距离差当控制收益。实际70,188 Policy动作、600固定零动作、0训练，614项CPU回归通过；两次历史长轨迹复现failed保持，累计技术开销独立记录，不混入主对照分母或升级为qualification。

前序历史复现设计（2026-11-02）：用户提出直接比较稳定化两设置的闭环成功率，优先运行[stabilization0/5 candidate配对对照](libero-task4-task7-stabilization-closed-loop-diagnostic-2026-11-02.md)，RGB/本体分解暂后移。固定六个baseline500，240 episodes/最多72,000 Policy动作、600固定零动作，native horizon与初态顺序匹配，先复现五步臂全部120条再运行零步臂。预运行测试与已有帧CUDA精确比对通过，新增训练0；结果与实际预算另记，不提前宣布某一设置更好或进入 qualification。

前序source输入结果（2026-11-02）：[source 稳定化 0/5 配对诊断](libero-task4-task7-source-stabilization-control-diagnostic-2026-11-02.md)已完成并独立复核。两任务三 seed 在共同尺度下的首步距离比值 five/zero 均 <1，分别为 task 4 `0.0529–0.4340`、task 7 `0.0513–0.0892`，超线比例也均下降。稳定化是初始观察差异的一个可复现来源，尚未证明动作正确性或闭环收益。实际 120 source reset、300 固定零动作、0 Policy 控制/训练；原 Path 类型错误、完整采集与零环境离线续算的 provenance 分开保留。55 项相关测试、566 项 CPU 回归通过；下一项用本轮 source 公共缓存分离 RGB 与本体响应，不把五步观察直接配旧 action[0] 训练。

前序夜间状态（2026-11-02）：[夜间诊断](libero-overnight-diagnostic-progress-2026-11-02.md)完成并独立复核。保留六个 500-update baseline；projection 区分下降、错误 tanh 强饱和和夹爪输入替换均未达到各自完整条件，BCE 条件训练及闭环分支停止。专家观察上 open 命令偏迟，初始覆盖差异则依赖参照时相，尚不能唯一解释低成功率。下一项先准备固定 Policy 的 source 稳定化匹配对照，不提前进入 SPT A/B/C、qualification 或其他环境正式实验。新增训练/环境交互为 0，CPU 回归 532 项通过；详细预算、失败恢复与 provenance 见晨间报告。

## 当前顺序

该轮冻结时的设置见[source 稳定化匹配对照](libero-task4-task7-source-stabilization-control-diagnostic-2026-11-02.md)：保持原 500-update Policy，source 支持集 0/5 固定零动作对照，两臂 native horizon/初态/顺序/参照数量匹配。现已完成，原 60 个零步公开观察及六模型读出精确复现；最新结果和实际成本以该执行报告为准，不由配置冻结推称正式结果。

| 阶段 | 环境 | 主要问题 | 当前证据边界 |
|---|---|---|---|
| 1 | LIBERO | 同一 Skill Family 的 SPI 迁移、support/query 适应和 SPT 更新 | 主要是操作技能和知识迁移；标准设置偏 lifelong imitation learning |
| 2 | Continual World | 任务按顺序到达后的 forward transfer、forgetting 和历史 Module 保持 | 主要是持续强化学习；CW10/CW20 不等于同族元学习 |
| 3 | Crafter | 长期记忆、探索、Capability 关系和因果 Knowledge Evolution | 不把当前 wood gather 结果升级为自然 Module 或完整 KECRL |
| 4 | RoboCasa365 | 原子 Module、复合任务、任务流和 Knowledge–Skill 联合收益 | 先做冻结子集，再扩展到更大任务规模；不直接把 benchmark 分数称为 Knowledge Evolution |

## 阶段 1：LIBERO

先选择一个共享动作结构的任务族，例如 `Pick`、`Place` 或 `Open`，而不是把整个 130 任务集合直接作为一个 Skill Family。

每个实验应包含：

1. 训练 SPI 和独立 held-out SPI；
2. A：原始 SPT；
3. B：来源 SPI 经验更新后的 SPT；
4. C：相同来源数据和预算的普通策略预训练；
5. 相同观测、动作接口、适应预算、评价初始状态和随机种子；
6. 独立 query、Module qualification 和迁移效率评价。

B 优于 A 才支持来源经验对共享 SPT 有帮助；B 优于 C 才支持收益来自 SPT 更新，而不是普通预训练。LIBERO 原生实验偏 imitation learning，因此若使用 RL 或 FOMAML，必须单独冻结算法和数据协议。

## 阶段 2：Continual World

Continual World 用来补充阶段式任务流证据。使用 CW10 做快速校准，CW20 做正式任务流候选；每个阶段只训练当前任务，并在阶段边界重测全部历史任务。

至少报告：

- forward transfer；
- backward transfer；
- forgetting；
- 历史 Module 保持率；
- 新任务达到资格门槛所需的交互步数；
- 任务顺序改变后的性能变化。

Continual World 的默认任务并不都属于同一个 Skill Family。因此它用于持续性、保持和遗忘，不作为 LIBERO 同族元学习结果的替代品。

## 阶段 3：Crafter

Crafter 继续承担 Knowledge Evolution 侧实验。当前 wood gather 诊断和历史 RGB/PPO 结果已封存，后续重点应转向：

- 可检测 Capability 的定义和验证；
- 资源、工具和任务前置条件的机制干预；
- 机制关系的证据累积、修正和置信度变化；
- Knowledge 对后续任务规划和任务可达性的影响。

Policy 成功率、Module 性能和梯度信息不直接写入 Knowledge Bank。Crafter 结果只支持 Knowledge Evolution 范围内的主张，除非另有独立的 SPI 和 Module qualification 协议。

## 阶段 4：RoboCasa365

RoboCasa365 是联合验证环境，不是早期调参环境。先冻结一个小任务子集：

- 2–3 个原子 Skill Family；
- 每个族包含训练实例和 held-out 实例；
- 2–3 阶段复合任务；
- 少量 4–5 阶段复合任务；
- 固定机器人 embodiment、观测、动作和数据来源。

在这个子集上比较四个臂：

1. 无 Knowledge Evolution、无 Skill Evolution；
2. 仅 Knowledge Evolution；
3. 仅 Skill Evolution；
4. Knowledge–Skill 双演化。

评价同时包含 Knowledge 结构正确性、SPI 适应效率、Module qualification、复合任务成功率、任务成本和遗忘。只有子集协议稳定后，才扩展到更多厨房环境、移动操作和更长组合任务。

## 统一边界

- 不把 LIBERO、Continual World、Crafter 和 RoboCasa365 的成功率直接合并。
- 不使用最终评价集选择 checkpoint、调参或修改任务顺序。
- 不把标准 benchmark 的多任务分数直接称为 SPT 更新或 Knowledge Evolution。
- 不要求同一个 Policy 后端跨 Crafter 和机器人环境复用；跨环境共享的是理论接口和评价边界。
- 每个环境都记录配置、seed、设备、预算、实际交互步数和数据来源。

## 当前状态

LIBERO 当前工作项（2026-11-01）：[固定权重梯度诊断](libero-task4-task7-gradient-interference-diagnostic-2026-11-01.md)已完成。六个 baseline 的共享整体局部余弦中位数均为正，不满足三 seed 一致整体冲突的主指标；局部负向窗口和 task 7 投影层负向结构另列，不能从整体统计推出无干扰。同权重虚拟裁剪窗口数虽增加，仍未重建实际 Adam 轨迹。完整 v2 的 2,670 个主窗口、890 个 source 输入摘要与六份梯度 archive 已独立复核；134 项相关测试通过。下一项按分支规则准备 H2 跨示教条件可分性设计，继续使用既有 500-update baseline、零新增训练/环境交互，不恢复局部候选或进入 qualification/SPT/Knowledge/Module 实验。v1 数值失败与两精度审计作为独立 provenance 保留。

截至本路线冻结日：

- Crafter 仍是诊断/Knowledge Evolution 候选环境；没有自然 wood Module 通过完整资格检查。
- LIBERO 已完成 Python 3.10 联合环境、完整 `libero_spatial` 数据审计、10 个任务的 KECRL adapter smoke、候选 Skill Family 和数据隔离审计、CUDA 训练路径、候选 BC Policy/SPT/SPI pipeline、单 source SPI 固定预算学习与闭环归因。HDF5 时序审计确认 `obs[t]` 在 `action[t]` 后记录。随后完成 same-index/causal pre-action 对照、hidden-state follow-up、逐步动作/状态偏差、单步反事实、受控状态分布偏移、RGB/本体感知和 agentview/eye-in-hand 相机分解；三 seed 相机分解均确认 eye-in-hand 局部敏感性大于 agentview。单变量 eye-in-hand 平移增强 pilot 将 support/query 闭环从 baseline `0/9, 0/9` 变为增强臂 `3/9, 3/9`，但 query MSE 均值略升，新增成功集中在一个 seed。固定三 seed 的 499 个 expert-state 合成平移敏感性诊断显示增强臂在三个半径的四方向聚合中均降低 action、translation、rotation 和 gripper 敏感性。最新动作误差—状态偏移对齐显示：增强臂前 20 步 expert-state action MAE 更低，但两臂 joint-state 首次越界都在第 2–3 步，后段 drift 和 gripper-state 越界继续放大；因此相机敏感性不是唯一瓶颈，动作 head/数据覆盖与闭环反馈需要分开诊断。随后完成 source-SPI 条件覆盖、relation×geometry×phase 分层误差、专家重放一致性、HDF5—当前重放状态对齐、回放语义冻结、当前 runtime 轨迹和输入对齐审计；四条示教都在首个动作后偏离 HDF5 状态，包括成功控制示教，而检查的 controller 配置和动作边界没有差异，当前 runtime 重复轨迹确定，两种 causal 输入视图从第一个动作后开始分叉；随后固定重建 task 3/7 Policy 做输入替换敏感性，四条示教共 456 步的 shared-hidden action MAE 为 0.02185，独立 recurrent 分支为 0.03544，且差异均从 step 1 开始。输入语义分叉会改变动作输出，但不能解释全部失败；随后对 5 个 source-SPI 任务的 150 条 meta-support 示教做了完整 replay，只有 17 条（11.33%）能覆盖 HDF5 action 序列末端，平均前缀覆盖率为 87.35%。因此已冻结 HDF5 causal 作为完整离线 BC 协议，当前 runtime 仅用于闭环/对齐诊断。阶段/夹爪损失 targeted diagnostic 的两任务加权 query action MAE 为 baseline 0.11593、phase-balanced 0.11324、6x gripper-weighted 0.13180；阶段均衡有小幅候选收益，但单独加大夹爪损失没有收益。随后容量匹配的 CNN-only/CNN+GRU 对照得到 0.10707/0.11324，只有 task 7 transport gripper 有局部 GRU 优势，因此不主张稳定 memory benefit。CNN-only + phase-balanced 扩展至 5 个 source task、2 个 seed 后，25,238 个 query step 的加权 action MAE 为 0.12295；其配对 uniform-loss 对照为 0.11708。phase-balanced 将 release action MAE 从 0.18487 降至 0.16041，但 transport 从 0.12186 升至 0.13935，两个 seed 的总体差值方向一致，因此不作为通用 loss 默认值，基础离线策略采用 CNN-only + uniform action MSE。当前 runtime 的闭环、action/proprioception trace 和 RGB 快照诊断均已完成，下一项是阶段对齐诊断。validation、held-out、qualification 未访问，SPT/Knowledge 未更新，Module 未注册。详细结果见 [`docs/libero-environment-feasibility-2026-10-21.md`](libero-environment-feasibility-2026-10-21.md)、[`docs/libero-spatial-skill-family-audit-2026-10-21.md`](libero-spatial-skill-family-audit-2026-10-21.md)、[`docs/libero-demonstration-phase-structure-audit-2026-10-21.md`](libero-demonstration-phase-structure-audit-2026-10-21.md)、[`docs/libero-spatial-transfer-split-2026-10-21.md`](libero-spatial-transfer-split-2026-10-21.md)、[`docs/libero-cuda-training-path-smoke-2026-10-21.md`](libero-cuda-training-path-smoke-2026-10-21.md)、[`docs/libero-bc-policy-pipeline-check-2026-10-21.md`](libero-bc-policy-pipeline-check-2026-10-21.md)、[`docs/libero-single-spi-bc-learnability-diagnostic-2026-10-21.md`](libero-single-spi-bc-learnability-diagnostic-2026-10-21.md)、[`docs/libero-bc-closed-loop-attribution-2026-10-21.md`](libero-bc-closed-loop-attribution-2026-10-21.md)、[`docs/libero-bc-causal-alignment-diagnostic-2026-10-22.md`](libero-bc-causal-alignment-diagnostic-2026-10-22.md)、[`docs/libero-bc-hidden-state-diagnostic-2026-10-22.md`](libero-bc-hidden-state-diagnostic-2026-10-22.md)、[`docs/libero-stratified-error-diagnostic-2026-10-25.md`](libero-stratified-error-diagnostic-2026-10-25.md)、[`docs/libero-expert-replay-consistency-audit-2026-10-25.md`](libero-expert-replay-consistency-audit-2026-10-25.md)、[`docs/libero-replay-state-alignment-audit-2026-10-25.md`](libero-replay-state-alignment-audit-2026-10-25.md)、[`docs/libero-replay-semantics-audit-2026-10-25.md`](libero-replay-semantics-audit-2026-10-25.md) 和 [`docs/libero-current-runtime-replay-audit-2026-10-25.md`](libero-current-runtime-replay-audit-2026-10-25.md) 和 [`docs/libero-replay-input-alignment-audit-2026-10-25.md`](libero-replay-input-alignment-audit-2026-10-25.md)。
- Continual World 尚未在本仓库中启动正式实验。
- RoboCasa365 尚未进入正式训练，需先完成环境安装、接口适配和小规模协议预检。
- 本路线是研究设计冻结候选，不是正式实验结果。

### LIBERO 当前诊断项

最新进展：500-update 前缀对照已在三个训练 seed 上完成，candidate 初态与评估环境 seed 固定。前三个初态上 HDF5 共 `5/45` 成功，runtime-native `8/45`；两臂共同失败 36 对，runtime-native 仅净增 3 次且加权离线 MSE 略差。随后对两个新增训练 seed 覆盖完整 `candidate_nonregression` 初态 `0..19`，HDF5 为 `26/200`、runtime-native 为 `37/200`；完整覆盖的 200 对中有 158 对共同失败，成功主要集中在 task 0，仍不足以确认观测来源优势。固定分层轨迹诊断随后重建 8 个对应模型，采集 18 条 RGB/action/proprioception 轨迹并全部通过模型 SHA、终止步数和初态顺序核验；task 7 的 HDF5 臂在两个抽样状态均没有 close 命令，而 source prefix 的 close 命令占比为 71.8%，其他任务显示状态依赖的夹爪时序差异。support-vs-free-run 分解进一步显示两臂 support gripper sign accuracy 几乎相同（HDF5 `0.9646`、runtime-native `0.9638`），但 HDF5 两个 candidate 状态前 80 步都没有 close，支持“早期 candidate 观察/状态漂移”作为下一诊断方向，不能归因成缺少 close 标签或一般 observation-source 优势。详细协议、逐 task 统计和 SHA 见 [runtime-native 前缀训练 pilot](libero-runtime-native-prefix-training-pilot-2026-10-27.md)、[失败轨迹诊断](libero-runtime-native-prefix-failure-trace-diagnostic-2026-10-27.md) 与 [task 7 夹爪分解](libero-runtime-native-prefix-task7-gripper-output-diagnostic-2026-10-27.md)。下一步比较 candidate 初始 observation、support 窗口和第一步/free-run action divergence。

后续更新（2026-10-27）：上述初始 observation 检查已完成。task 7、training seed `20261022` 的两个 candidate 状态中，两臂初始 proprioception 相同，但首步动作 L2 差为 `0.211` 和 `0.261`；两臂前 20 步均保持 open，因而首步差异表现为运动命令差异，不能直接解释为夹爪触发差异。该诊断支持“首步运动分叉先于后续 close 时序差异”的假设，但未区分视觉编码、动作 head 和状态反馈。下一项使用同一批固定模型和自由运行轨迹做交叉输入评估：分别把两条轨迹在相同时间点的观察输入两臂模型，测量固定观察下的模型输出差异，以及固定模型下的观察变化效应；它是机制诊断，不是 teacher-forcing 训练实验或因果干预。完整结果见 [task 7 初始观察诊断](libero-runtime-native-prefix-task7-initial-observation-diagnostic-2026-10-27.md)。

后续更新（2026-10-28）：task 7 交叉输入评估已完成。两臂固定模型、固定 candidate state `0/8` 的既有 20 步 observation 轨迹，四种模型/观察组合均通过逐步回放核验。第 1 步 observation 完全相同，跨模型动作差为 `0.2105/0.2609`；到第 20 步，跨模型动作差仍为 `0.098--0.109`，同一模型替换另一臂 observation 的动作差为 `0.022--0.028`。因此目前证据更支持早期 Policy 映射差异是首要直接因素，观察状态漂移是随后累积的次要因素；该排序尚未区分视觉 encoder、fusion、action head、控制器和数据覆盖。结果仍仅来自 task 7、一个 training seed、两个 candidate 状态，不是 SPI qualification 或 observation-source 优势。详细结果见 [task 7 交叉输入诊断](libero-runtime-native-prefix-task7-cross-input-diagnostic-2026-10-28.md)。下一步扩展到 task 1/3/4/7 的成功/失败分层，并保持同一四格输入分解。

边界修订（2026-10-28）：现有 task 1/3/4/7 failure trace 只保存逐步 action 和 joint/gripper 状态，以及稀疏 RGB 快照，没有保存每一步的两路 Policy RGB observation。因此不能在不重跑环境的情况下扩展同一四格输入分解；直接使用快照近似会破坏时间对齐。下一步先实现跨 task/seed 的短轨迹采集协议，固定模型哈希、adapter 生命周期、候选状态和前 20 步完整 observation，再进行分层交叉输入诊断。当前 task 7 结果不外推到其他任务。

后续更新（2026-10-28）：跨任务短轨迹采集和四格交叉输入评估已完成。task 1/3/4/7 的 9 个预先固定状态、8 个 500-update Policy 均通过完整 `0..19` 候选重放与旧 trace action 核验；每臂前 20 步的两路 RGB、joint 和 gripper observation 已保存为 72 个数组。第 20 步固定 HDF5 observation 的跨模型动作差在 task 1/4/7 的选中状态通常大于固定模型的 observation 替换差，但 task 3 两个状态均由 observation 差异占主导。该结果不支持跨任务统一的“模型映射优先”结论；下一步固定 task 3 的已采集 observation，拆分 RGB 与 joint/gripper 输入贡献。详细结果见 [跨任务观察诊断](libero-runtime-native-prefix-cross-task-observation-diagnostic-2026-10-28.md)。

后续更新（2026-10-28）：task 3 RGB/本体模态拆分已完成。固定两个 Policy、两个 candidate 状态和前 20 步 observation，RGB replacement action MAE 平均 `0.04095`，joint/gripper replacement `0.01621`，联合 `0.04084`，交互项绝对 MAE `0.00330`。这说明 task 3 局部动作敏感性主要落在 RGB 输入，但仍未区分视觉 encoder、相机内容、fusion、action head 与状态覆盖；下一步拆分 agentview/eye-in-hand 两路相机。详细结果见 [task 3 模态诊断](libero-runtime-native-prefix-task3-modality-diagnostic-2026-10-28.md)。

后续更新（2026-10-28）：task 3 两路相机拆分已完成。固定两个 Policy、两个 candidate 状态和前 20 步 observation，agentview replacement action MAE 平均 `0.00351`，eye-in-hand `0.04006`，双相机联合 `0.04095`，交互项绝对 MAE `0.00141`；eye-in-hand 约为 agentview 的 `11.4` 倍，四个 Policy×状态组合排序一致。该结果仍是局部输入敏感性证据，不能直接归因于相机或 CNN；下一步扩展到 task 1/3/4/7 的 9 个固定状态。详细结果见 [task 3 相机诊断](libero-runtime-native-prefix-task3-camera-diagnostic-2026-10-28.md)。

后续更新（2026-10-28）：跨任务两路相机复核已完成。复用 task 1/3/4/7 的 9 个固定状态、8 个 500-update CNN-only Policy 和前 20 步 observation archive，四个任务的第 20 步 task-level agentview/eye-in-hand replacement action MAE 分别为 task 1 `0.000113/0.003958`、task 3 `0.003513/0.040062`、task 4 `0.000392/0.005266`、task 7 `0.001014/0.006391`；双相机项均接近 eye-in-hand 项，interaction 较小。eye-in-hand 主导排序在这组固定分层上复现，但效应量和任务状态动力学不同，不能据此断言相机或 CNN 是根因，也不改变 HDF5 causal 离线训练边界。下一步保持 CNN-only + uniform action MSE 不变，做单变量 eye-in-hand 几何覆盖/输入增强诊断。详细结果见 [跨任务相机诊断](libero-runtime-native-prefix-cross-task-camera-diagnostic-2026-10-28.md)。

后续更新（2026-10-28）：固定 Policy 的 eye-in-hand 平移敏感性诊断已完成。在相同 9 个状态、8 个 Policy 和前 20 步 observation 上，只对各自 observation arm 的 eye-in-hand RGB 做反射填充的 `±1/±2/±4` 像素平移。四任务、两臂和四方向平均的第 20 步 action MAE 为半径 1/2/4 的 `0.010919/0.021758/0.025939`；task 7 半径 4 为 `0.043019`，task 4 为 `0.011722`。结果支持输入几何扰动会改变动作读出，但效应依赖任务和方向，不能写成相机因果或闭环收益。下一步在同一预算和数据边界下训练 paired eye-in-hand 平移增强臂。详细结果见 [跨任务 eye-in-hand 平移敏感性诊断](libero-runtime-native-prefix-cross-task-eye-hand-translation-sensitivity-diagnostic-2026-10-28.md)。

后续更新（2026-10-28）：跨任务 eye-in-hand 平移增强对照已完成。task 1/3/4/7 分别用既有一个 seed、历史 HDF5/runtime-native 两个 observation arm、相同初始化与 batch 顺序、每臂 500 updates；增强只作用于训练图像，evaluation 使用原始图像。增强臂在四个任务的未增强离线 MSE 全部变差，按有效动作步汇总从 `0.034627` 升至 `0.037607`；固定状态敏感性明显降低，但 candidate 非资格闭环成功从 baseline `6/18` 降到 `3/18`（配对结果：baseline-only 4、augmentation-only 1）。因此停止扩大当前 4 像素随机平移方案，不把敏感性下降等同于控制改善。下一项保持 CNN-only + uniform action MSE，不改 Policy，按任务阶段分析早期平移 action error 与已有 RGB/joint/gripper trace 的关系，不使用 simulator 私有目标坐标。详细结果见 [跨任务 eye-in-hand 增强诊断](libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic-2026-10-28.md)。

后续更新（2026-10-28）：阶段命令诊断已完成。source-support 前缀按 HDF5 gripper action 符号统计，candidate 则按 Policy 实际 gripper 命令统计，明确不把 candidate schedule 称为 expert action error 或真实 phase。task 7 HDF5 两条候选轨迹 600 步均无 close，而 source task 7 的 close 占 65.4%；task 4 HDF5 两个失败状态在第 83/101 步附近进入 release，runtime 三条轨迹则保持 transport；第 20 步 task 4 eye-in-hand RGB drift 最大（HDF5/runtime `47.788/49.143`），task 7 较小（`21.117/22.462`）但仍存在 close 缺失。因此视觉漂移不是唯一解释，下一步冻结 Policy 做 task 4/7 的 action-head 与视觉表示读出分解。详细结果见 [阶段命令诊断](libero-runtime-native-prefix-phase-command-diagnostic-2026-10-28.md)。

补充：paired loss 对照中的 uniform arm 与 phase-balanced arm 都对七个动作维度等权；配置里的 `gripper_action_weight=6.0` 在这两种 loss mode 中未生效，仅用于独立的 gripper-weighted arm。完整配对结果见 [CNN-only phase-loss paired diagnostic](libero-cnn-only-phase-loss-paired-diagnostic-2026-10-25.md)。对应的 phase-uniform CNN-only Policy 已在 5 个 source task、2 个 seed、每组 3 个固定 `candidate_nonregression` 初始状态上完成闭环检查，仅 task 3 / seed `20261021` 的 2/3 状态成功，总计 `2/30`。结果和边界见 [CNN-only phase-uniform closed-loop diagnostic](libero-cnn-only-uniform-closed-loop-diagnostic-2026-10-25.md)。随后抽取 7 条成功/失败轨迹，确认机械臂关节持续响应、部分失败夹爪关闭命令不足，同时也有足够长的 close run 却失败的轨迹；完整表和限制见 [Policy trace diagnostic](libero-cnn-only-uniform-policy-trace-diagnostic-2026-10-25.md)。固定 RGB 快照随后在同一 7 条轨迹上完成，7/7 重放通过来源成功/步数和模型 SHA 校验，共保存 134 帧；画面支持末端接近与初态敏感性是候选瓶颈，但不能区分 CNN、action head、覆盖或夹爪时序根因。结果和边界见 [RGB snapshot diagnostic](libero-cnn-only-uniform-policy-rgb-snapshot-diagnostic-2026-10-26.md)。已有 [分层 expert-state / free-run 误差诊断](libero-stratified-error-diagnostic-2026-10-25.md)、[输入替换敏感性审计](libero-policy-input-replacement-sensitivity-audit-2026-10-25.md) 和 [source-support replay 可行性审计](libero-source-support-replay-feasibility-audit-2026-10-25.md) 已覆盖阶段漂移与 HDF5/runtime 观测分叉，因此不重复这些实验。新的 runtime-native 前缀采集已覆盖 150 条示教，前缀为 63–197 步；其中 17 条完整执行 HDF5 action，133 条在当前 runtime 提前 success，详见 [runtime-native prefix collection](libero-runtime-native-prefix-collection-2026-10-27.md)。随后完成 100-update historical-vs-runtime-native training pilot，两个 observation arm 都是 `0/15` candidate success，详见 [runtime-native prefix training pilot](libero-runtime-native-prefix-training-pilot-2026-10-27.md)。下一步在相同前缀协议上扩大到 500 updates；这仍是机制诊断，不是 qualification。


逐步动作漂移、局部单步反事实、状态分布偏移、RGB/本体感知模态分解、两路相机分解、三 seed 相机复核、单变量 eye-in-hand 平移增强 pilot、固定 expert-state 合成平移敏感性、动作误差—状态偏移对齐、action-head 分解、source-SPI 条件覆盖审计及 relation×geometry×phase 分层误差诊断均已完成。增强臂降低了固定 expert-state 的相机敏感性和前 20 步动作误差，但 joint-state 首次越界仍在第 2–3 步；只训练 action head 的 query MSE 明显高于 full Policy，说明表示学习是必要条件。条件覆盖审计排除了 support/query 缺少 relation、source geometry 或动作阶段类别的简单断裂；分层诊断显示 task 4 open-container 条件的状态/RGB 漂移最大，部分任务的 release 夹爪误差放大。专家 replay 一致性、HDF5—当前重放状态对齐、回放语义冻结、当前 runtime 轨迹和输入对齐审计显示，task 3 demo 1 对 stabilization 敏感、task 7 demo 0 跨 seed/设置稳定失败，四条示教都在首个动作后偏离 HDF5 状态，包括成功控制示教，但已检查的 controller 配置和动作边界没有差异，当前 runtime 三次完整 trace 确定，两种 causal 输入视图从第一个动作后开始分叉；固定 task 3/7 Policy 的输入替换敏感性审计已经完成，shared-hidden action MAE 为 0.02185，独立 recurrent 分支为 0.03544；source-support 全量 replay 进一步显示只有 17/150 条示教具有完整当前-runtime 前缀覆盖。因此 HDF5 causal 已冻结为完整离线 BC 协议，当前 runtime 只用于闭环/对齐诊断。阶段/夹爪损失 targeted diagnostic 显示 phase-balanced 有小幅离线收益，6x gripper weighting 没有收益；容量匹配的 CNN-only/CNN+GRU 对照没有发现稳定的 GRU 优势，CNN-only + phase-balanced 已扩展到 5 个 source task 和 2 个 seed，并完成同结构 uniform-loss 配对；阶段均衡没有总体收益，release 改善以 transport 恶化为代价。CNN-only + uniform loss 的当前 runtime 闭环诊断、action/proprioception 轨迹和 RGB 快照均已完成。所有这些仍是机制诊断，不是视觉修复有效性或正式 SPI 结果，不访问 official candidate、validation、held-out 或 qualification，不选 checkpoint。

后续更新（2026-10-28）：task 4/7 action-head 读出诊断已完成。保持相同初始化、causal source-support 前缀、batch 顺序和 500-update CNN-only uniform-loss 边界，full Policy 的 source-support MSE 为 `0.02656–0.03300`，冻结 visual encoder/fusion/recurrent、只训练 1,799 个 action-head 参数的 MSE 为 `0.18288–0.21104`（高 `5.5–7.9×`）；gripper MSE 高 `7–12×`。在既有 candidate observation 短窗口上，action-head-only 的 10 个固定案例都在第 1 步发出 close，而 full Policy 完整复现旧 trace。该结果支持表示/融合/时序映射学习是必要条件，排除“只重新拟合 action head 即可修复”的解释，但不能把 CNN 单独认定为根因，也没有 candidate expert-action error。详细结果见 [action-head 读出诊断](libero-runtime-native-prefix-task4-task7-action-head-readout-diagnostic-2026-10-28.md)。下一步冻结逐级模块集合，分离 visual encoder、fusion/recurrent 和 action-head 的边际贡献。

后续更新（2026-10-29）：task 4/7 四臂冻结层级读出诊断已完成。每个 task 内对 HDF5/runtime-native 两臂均值显示，source-support MSE 随可训练范围收窄而递增：full Policy 为 `0.02894/0.03295`，冻结 visual encoder 为 `0.04703/0.05101`，再冻结 fusion 为 `0.07046/0.07588`，仅训练 action head 为 `0.21094/0.18296`（task 4/7）。该排序说明当前 support-prefix 拟合需要视觉和 fusion/recurrent 路径参与，但冻结范围同时改变参数量和优化自由度，不能当作模块因果效应或 query 泛化结论。candidate 仅离线读出；action-head-only 在 10 个固定窗口第 1 步均输出 close，冻结策略没有环境 rollout。详细协议和边界见 [task 4/7 冻结层级诊断](libero-runtime-native-prefix-task4-task7-module-freeze-readout-diagnostic-2026-10-29.md)。下一步用两个额外初始化 seed 复核 source-support 拟合排序，再单独设计公平对齐的 query 泛化评价。

后续更新（2026-10-29）：冻结层级读出在两个新增初始化 seed `20261023/20261024` 上复核完成，并与 seed `20261022` 的既有结果合并。task 4/7、HDF5/runtime-native 两臂共 `12/12` 组均保持 source-support MSE 严格排序 `full < freeze_visual < freeze_visual_fusion < action_head_only`；task 7 的 freeze-visual 误差对 seed 更敏感，但排序未反转。这说明读出层级趋势不是单一初始化现象，不表示 query 泛化或模块因果贡献。结果见 [冻结层级 seed 复核](libero-runtime-native-prefix-task4-task7-module-freeze-seed-replication-diagnostic-2026-10-29.md)。

后续更新（2026-10-30）：source `meta_support` → `meta_query` 离线泛化诊断已完成。task 4/7、三个初始化 seed、四种 CNN-only Policy 可训练层级、每模型 500 updates，共 24 个模型；query 数据在每个任务的 support 拟合结束后才创建。严格 query MSE 顺序 `full < freeze_visual < freeze_visual_fusion < action_head_only` 在 `6/6` task×seed 组成立。Full Policy query MSE 为 task 4 `0.03852`、task 7 `0.04263`，action-head-only 为 `0.21473/0.18305`。这是同任务的离线示教泛化，不代表闭环成功、跨 SPI 迁移或模块因果效应。完整结果、seed 波动、实际可训练参数量及边界见 [source meta-query 诊断](libero-source-meta-query-generalization-diagnostic-2026-10-30.md)；配置和 runner 为 [冻结协议](../configs/libero-runtime-native-task4-task7-support-query-hdf5-diagnostic_v1.json) 和 [`run_libero_source_meta_query_generalization_diagnostic.py`](../experiments/run_libero_source_meta_query_generalization_diagnostic.py)。下一项冻结为 task 4/7 Full Policy 三 seed 重训并精确复现 query run 权重，然后在完整 `candidate_nonregression 0–19` 初态集做最多 300 步、固定 environment seed `20261022` 与 stabilization `5` 的配对闭环诊断，记录逐步公开 proprioception 轨迹。该初态集已用于既有诊断，不是新测试集；qualification、held-out、final evaluation 和 SPT/Knowledge 更新仍未触及。协议见 [Full Policy 闭环诊断配置](../configs/libero-task4-task7-full-policy-closed-loop-diagnostic_v1.json)。

后续更新（2026-10-31）：对上述固定闭环 artifact 的 120 条公开轨迹完成只读归因分析，没有重放环境、训练或访问资格初态。7 条成功轨迹全部出现 close；113 条 time-limit 中 32 条没有 close，15 条停在 close 后未出现 release，66 条在出现 release 命令后仍未成功。第 1–20 步 success/time-limit 的公开 joint、gripper offset 和七维 action norm 接近；第 100 步失败组 joint offset 均值较高，但 task 4/7 的方向不同，不能归结为单一状态漂移根因。训练 seed 成功数为 `0/40、3/40、4/40`，只作描述性分层，不能用于选择 checkpoint。详细口径、指定步点均值/中位数和 provenance 见 [闭环轨迹归因诊断](libero-task4-task7-full-policy-closed-loop-attribution-diagnostic-2026-10-31.md)。这仍是 candidate nonregression 的机制诊断，不是 unseen generalization、正式 SPI、Knowledge Evolution 或 Module qualification；下一步保持 Policy 与训练协议不变，做短窗口 observation/state replacement 或首步/首次 close 对齐诊断。

后续更新（2026-10-31）：在同一 120 条公开 trace 上完成 action→公开 state 对齐分析，没有新增 rollout 或训练。首步 joint/gripper 响应在 success/time-limit 间接近；首次 close 后成功组 joint delta 均值从 `0.02218` 降到 `0.01723`，time-limit 组从 `0.02299` 降到 `0.02137`，但成功样本仅 7 条，不能作因果解释。task 7 的 time-limit 轨迹有 `27/56` 没有 close，task 4 为 `5/57`；close 后的响应方向仍具有 task-specific 差异。详细结果见 [闭环状态-动作对齐诊断](libero-task4-task7-full-policy-closed-loop-state-action-alignment-diagnostic-2026-10-31.md)。当前 120 条轨迹预算已用尽，不能把另一套 runtime-native Policy 的 observation archive 混入本 Full Policy 证据；若继续做 observation replacement，必须先冻结新的非 formal 配置和 provenance。

后续更新（2026-10-31）：基于同一状态-动作 artifact 又完成 close/release 连续段分析，没有新增 rollout。time-limit 轨迹的 release 段均值较长（`95.2` 对成功组 `2.75`），但受失败轨迹 300 步右删失影响；close 段长度在 task 4 与 task 7 方向相反，不能推广统一的延长 close 或强制 release 规则。失败模式可描述为无 close、close 后无 release、以及 release 段持续至 time-limit 三类。详细结果见 [闭环 phase-response 诊断](libero-task4-task7-full-policy-closed-loop-phase-response-diagnostic-2026-10-31.md)。现有机制证据尚未分离视觉、action head、数据覆盖或闭环分布偏移的相对贡献，保持不重训、不新增 rollout 和不访问资格数据。

后续更新（2026-10-31）：冻结了一个未来的 Full Policy public joint/gripper state replacement 短窗口协议，但当前不运行。协议固定 task 4/7、三个已有 training seed、source 闭环结果中的六个 checkpoint、candidate state `0..19`、前 20 个 pre-action observation 和 natural/state-reset 两种离线输入分支；当前新增 episode budget 为 `0`，不会自动启动 rollout。配置、六个 checkpoint SHA 和验证边界见 [observation/state replacement protocol](libero-task4-task7-full-policy-observation-state-replacement-protocol-2026-10-31.md)。该协议仍不是正式 SPI、Knowledge Evolution 或 Module qualification。

后续更新（2026-10-31）：用户明确增加预算后执行了上述 v2 短窗口协议。六个固定 Full Policy checkpoint、task 4/7、三个 seed、candidate `0..19` 共采集 `120` 条、每条 `20` 步 natural observation/action 记录；replacement action 只离线前向，不进入环境。第 20 步 public joint/gripper reset replacement 的 action L2 为总体 `0.076515`，task 4 `0.065864`、task 7 `0.087166`，主要来自 translation，gripper action 差异较小。该结果是固定 Policy 的局部输入敏感性，不是闭环成功率、因果结论或 Module qualification。结果和 archive provenance 见 [observation/state replacement diagnostic](libero-task4-task7-full-policy-observation-state-replacement-diagnostic-2026-10-31.md)。

后续更新（2026-10-31）：对 v2 artifact 做 task/seed/outcome 分层 sensitivity 分析。第 20 步 joint drift 与 action delta 的 episode-level 相关在 task 4 为 `0.694`、task 7 为 `0.428`，pooled overall 为 `-0.391`，说明任务间均值差异会反转混合相关方向。该分析只支持 task-conditioned 描述，不支持 pooled 根因或因果结论。结果见 [observation replacement sensitivity 诊断](libero-task4-task7-full-policy-observation-state-replacement-sensitivity-diagnostic-2026-10-31.md)。

后续更新（2026-10-31）：用户明确“增加预算、启动训练”后，按授权 v2 运行了 public joint/gripper state jitter augmentation 诊断。task 4/7、三个 seed、每模型 500 Adam updates，共 6 个模型；causal HDF5 `meta_support` demonstrations `0..29` 训练，`meta_query` demonstrations `30..49` 离线评价，query 在每个 task 的全部 support fit 完成后才创建。task 4 三个 seed 的 query MSE 相对 pinned source Full Policy baseline 均下降，均值差为 `-0.001227`；task 7 均值差为 `+0.000231`，seed 方向混合。训练只改变 joint/gripper 输入 jitter，未做 rollout、未读取 candidate/qualification、未选 checkpoint、未更新 SPT/Knowledge、未注册 Module。实际设备为 RTX 4090 `cuda:0`，runner elapsed `141.24s`；结果、6 个 checkpoint manifest 和完整 provenance 见 [public-state jitter training diagnostic](libero-task4-task7-public-state-jitter-augmentation-diagnostic-2026-10-31.md)，协议见 [public-state jitter augmentation protocol](libero-task4-task7-public-state-jitter-augmentation-protocol-2026-10-31.md)。这些是 source-demo 离线诊断，不能写成闭环收益或因果结论。
补充预算解释：短耗时来自每模型 `500` updates、约 `434–461` 个 support windows、`821,704` 个可训练参数和无 rollout；每模型约重复 support windows `8.7–9.2` 遍。该预算足以完成当前固定离线比较，但没有中间 loss 曲线，不能据此判断收敛，也不能说明闭环失败只是预算不足。若继续验证预算效应，必须同时重训 baseline/jitter 并冻结 `500/1000/2000` budget-sweep，不能把更长 jitter 与当前 500-update baseline 混比。
后续更新（2026-10-31）：按冻结的 `500/1000/2000` budget-sweep 同时重训 baseline 与 public-state-jitter 两臂。task 4/7、三个 seed 共 12 条 endpoint trajectory，每条到 2000 updates，并在三个固定点记录 support/query 曲线。support MSE 从 500 到 2000 持续下降，但 query 不单调；2000 点 jitter 相对 baseline 的 query MSE 均值差为 task 4 `-0.001045`、task 7 `-0.001898`，1000 点两 task 均未稳定改善。500 点与既有结果 hash parity 通过，36 个 checkpoint manifest 和两次保留的 fail-closed partial attempt 都写入 provenance。该结果只说明预算与离线 source-demo 泛化存在 task/arm 交互，不能写成闭环收益、因果结论、正式 SPI、Knowledge Evolution 或 Module qualification。详细结果见 [public-state jitter budget-sweep 诊断](libero-task4-task7-public-state-jitter-budget-sweep-diagnostic-2026-10-31.md)。
后续更新（2026-10-31）：按用户选择固定 `500 updates`，复用 budget-sweep 中 6 个 jitter checkpoint，在相同 candidate nonregression 初态 `0..19`、环境 seed `20261022`、stabilization `5` 和 300-step 上限下新增 120 条闭环 episode；baseline 使用已完成的 120 条 pinned reference。baseline 成功 `7/120`，jitter 成功 `6/120`；paired outcome 为 baseline-only `4`、jitter-only `3`、both `3`、neither `110`，reset digest `120/120` 对齐。该结果不支持 500-update public-state jitter 的总体闭环收益；下一步只做两臂 trace 的首步/close/release/state-drift 机制对齐，不注册 Module、不更新 SPT/Knowledge、不访问资格数据。完整 provenance 见 [500-update jitter closed-loop 诊断](libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic-2026-10-31.md)。
后续更新（2026-10-31）：对上述 baseline reference 与 jitter rollout 做了只读 paired trace alignment。jitter 有 close 的 episode 为 `77/120`，baseline 为 `88/120`；有 close 样本的首次 close 均值为 `80.8` 对 `68.5`。jitter 在 step 50/100 的 joint offset 均值更高，在 step 200/300 更低，gripper offset 全程接近；paired success transition 仍为 baseline-only `4`、jitter-only `3`、both `3`、neither `110`。这些是公开轨迹的描述性差异，不能解释为因果机制；下一步只围绕 7 条转移轨迹做 phase/首步细分，不改变预算或访问资格数据。结果见 [500-update jitter paired alignment 诊断](libero-task4-task7-public-state-jitter-500-closed-loop-alignment-diagnostic-2026-10-31.md)。
后续更新（2026-10-31）：进一步细分 7 条 success transition。jitter-only 的 3 条中，1 条由 jitter 新增 close，2 条在 release heuristic 前完成 success；baseline-only 的 4 条中，1 条丢失 close，1 条出现更早 release/短 close，1 条出现延迟 release/长 close，1 条为 task 4 的 close/release schedule change。首动作和 step 50/100 joint drift 在部分 task 7 转移中差异较大，但没有跨 task/seed 的共同方向。该分析仍是只读描述，不支持因果归因；结果见 [500-update transition mechanisms 诊断](libero-task4-task7-public-state-jitter-500-closed-loop-transition-mechanisms-2026-10-31.md)。
补充 RGB/action 边界：现有两臂闭环 artifact 只保存 action 与公开 proprioception trace，没有 paired agentview/eye-in-hand RGB frame，因此对 7 条 transition 只能完成前 20 步 action alignment，RGB/action 联合对齐不可估计。若后续需要 RGB 证据，必须先冻结新的 RGB-capturing rollout 协议；当前不把缺失帧推断为视觉机制，也不访问资格数据。结果见 [500-update RGB/action alignment 审计](libero-task4-task7-public-state-jitter-500-transition-rgb-action-alignment-diagnostic-2026-11-01.md)。
后续更新（2026-11-01）：按冻结的短窗口协议，对 7 条 transition 的 baseline/jitter 两臂新增采集 14 个 episode、每条前 20 步 RGB/action/proprioception。eye-in-hand RGB MAE 在 jitter-only/baseline-only 分别为 `0.01955/0.02586`，高于 agentview `0.00333/0.00477`；固定 Policy 的 RGB-only swap action delta 也高于 proprio-only swap。该结果只是 outcome-stratified 局部输入敏感性，不是相机因果或闭环收益；若扩大证据，需另行冻结全部 120 paired state 的 RGB capture 协议。详细 provenance 见 [RGB/action capture 诊断](libero-task4-task7-public-state-jitter-500-transition-rgb-action-capture-diagnostic-2026-11-01.md)。

后续证据复核与计划修订（2026-11-01）：只读审计显示，上述补采内部配对 reset 一致，但 task 4 的 4 个窗口未复现源闭环前缀；task 7 的 10 个窗口完全复现。task 4 现阶段只能作为新短窗口案例。同时当前 RGB-only swap 替换两路相机，并未单独测量 eye-in-hand 的 action sensitivity。暂缓全部 120 对 RGB 扩采，先补齐源前缀复现检查，再对现有 archive 做同观察跨 Policy 与逐相机离线分解；保持 500 updates 和原 baseline，jitter 不升为默认，不启动新训练。该轮证据收束后再冻结一个针对性修复变量，随后才推进 SPT A/B/C 与正式实验协议准备。审计 artifact、解释边界和顺序见 [500-update 后续计划](libero-500-update-next-step-plan-2026-11-01.md)。

执行更新（2026-11-01）：保持 500 updates 完成一次严格证据核对与 CPU 离线分解，未训练或新增 rollout。task 7 五对源前缀完全一致，task 4 两对隔离报告。主要分析中，首步观察完全相同但跨 Policy action L2 差均值为 `0.201346`；前 20 步同观察跨 Policy 差异为 `0.100411–0.111334`，固定 Policy 的完整观察替换差为 `0.026819–0.037324`。逐相机替换后，眼在手/公开本体状态/外部相机的局部敏感性排序在五对×两 Policy×两参照方向均成立；各范数不能相加作因果贡献率。保留原 baseline，当前证据仍不足以指定新训练修复；下一步先冻结 task 4 源前缀复现修复协议，明确状态顺序、adapter 历史、环境 horizon 与交互预算。配置、结果 SHA 与 85 项验证见 [证据核对与离线分解](libero-task4-task7-evidence-decomposition-diagnostic-2026-11-01.md)。

执行更新（2026-11-01）：task 4 两对源前缀复现已修复。原补采直接跳到目标状态，遗漏了原环境重置历史；原生抽屉属性采样器配置随 reset 累积。新协议保留状态顺序、固定 500-update 权重及 20 步窗口，新增 56 episodes、1,120 Policy actions 和 280 stabilization actions，`56/56` 前缀精确匹配，四个目标 archive 全部通过，耗时 `127.68s`，相关测试 `100 passed`。新增 optimizer updates 为 0，不修改第三方库或共享 adapter，也不重测最终 success。原 task 4 失败补采仍隔离保存；下一步用四个修复 archive 补齐 task 4 的同观察跨 Policy 和逐相机离线分解，按 task 分开汇报，再决定单一训练变量，预算继续保持 500 updates。配置、实际预算、原生代码及 checkpoint hash 见 [task 4 按原重置顺序复现诊断](libero-task4-ordered-prefix-reproduction-diagnostic-2026-11-01.md)。

执行更新（2026-11-01）：已完成修复后的 task 4 两对与原有效 task 7 五对的离线分解，14 个源前缀全部匹配，task 7 四格 CPU 输出/指标与旧结果精确一致。task 4 同观察换 Policy 的 action L2 为 `0.045174–0.045448`，固定 Policy 换完整观察为 `0.019082–0.022118`；首步动作分叉已在相同观察下出现。eye-in-hand 在两任务中均为局部敏感性最高的输入，但 task 4 的夹爪输出差远小于 task 7，不能直接采用统一的 gripper weighting 或视觉增强。新增训练/交互为 0、CPU elapsed `6.59s`、相关验证 `115 passed`；仍保留原 baseline。下一步固定 task 4/7 三 seed × 两臂的 12 个 500-update checkpoint，冻结一次有专家目标的 source support/query 动作分量及命令段误差诊断；零交互初版只用 `obs[t-1]→action[t], t>=1`，不伪造首帧或声称重算原完整 query MSE。该目标误差证据获得前，不凭输入敏感性大小选择训练修复。结果、限制与完整 provenance 见 [修复前缀离线分解](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md)。

执行更新（2026-11-01）：固定 task 4/7 三 seed × baseline/jitter 的 12 个现有 500-update checkpoint 完成专家目标动作误差诊断。source support/query 共 100 条示教，排除 100 个 `t=0` 后覆盖 14,490 个 causal 专家输入、86,940 行预测，CPU elapsed `219.16s`，新增训练/交互为 0。jitter query 平移 MSE 在两个任务均为三 seed 下降，但 task 7 夹爪 MSE 两 seed 上升、旋转三 seed 上升；不能据此替换默认 baseline。24/24 个 task×role×seed×arm 组合均满足 open 切换窗口夹爪 MSE > close 窗口 > 整体。下一项只检验专家 open 切换 ±5 步内 gripper MSE 权重 2 的单一局部损失候选，其余六维及 500-update 预算不变；全局大权重、视觉增强与更长预算不同时引入。候选设计尚未训练，先接入独立 runner、要求 baseline 500-update hash parity，并冻结完整 source 初帧预算/评价/停止规则，再运行有界诊断。136 项测试及独立输入/目标/误差复核通过；结果、候选设计和 provenance 见 [专家目标动作误差诊断](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)。这仍不是 formal SPI、SPT/Knowledge 更新或 Module qualification。

执行协议更新（2026-11-01）：已接入局部 open-transition gripper MSE，冻结 task 4/7 × 三 seed × baseline/候选、每模型 500 updates 的配对诊断。重训 baseline 必须精确匹配历史权重，两臂初始化/批次顺序一致；完整 causal source 初帧预算为 100 个 reset、4 个 adapter，stabilization/Policy action steps 为 0，不读 official evaluation states。两个任务均须通过三 seed 局部 gripper 改善、局部 sign error 不增及平移/旋转/七维 MSE 非回归规则，失败停止候选分支，不追加预算或搜索系数。专项 41 项及排除 Crafter/smoke 组的 CPU 回归 433 项通过；配置和实际运行记录见 [局部损失配对诊断](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)。通过离线门槛也只允许准备另行冻结的闭环诊断，不自动进入 Module qualification。

执行结果更新（2026-11-01）：局部 2× gripper loss 配对诊断完成全部 12 个 500-update endpoints，六个 baseline 初始/最终权重及六组 batch 顺序均精确复现，实际 6,000 updates、461,482 次有效示教步呈现、100 个 source 首帧 reset、0 Policy action steps。query open 窗口 gripper MSE 均值虽下降，但两个任务都只有 2/3 seed 下降；task 4 平移均值退化 7.03%，task 4/7 七维 MSE 退化 4.19%/6.63%，均未通过冻结门槛。按协议停止该分支，不搜索系数、不加 updates、不进入该候选闭环，保留原 baseline。下一步先收束已有输入对齐、表示、jitter 与损失证据，明确未分离的机制，再冻结不同的诊断。完整公开首帧、checkpoint、预测 archive 和独立复核均留存；26 个输入 pin、168 个误差分层单元、505 个输入摘要及停止判断已验证。实际成本、原 helper 索引 access record 的说明及负结果详见 [局部损失配对诊断](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)。

执行更新（2026-11-01）：证据收束后的梯度诊断完成并独立复核，六个 baseline500 的共享整体局部余弦中位数均为正，不满足一致整体冲突主指标；窗口级冲突和 task 7 逐帧投影层负向结果单列。保留 float32 v1 数值失败及两精度审计，完整 float64 v2 用原门槛执行，新增 optimizer/环境交互为 0。下一项转向 H2 的跨示教条件可分性设计，仍不训练新 Policy。源输入、原始梯度、134 项测试及实际计算成本见[梯度诊断](libero-task4-task7-gradient-interference-diagnostic-2026-11-01.md)。

执行更新（2026-11-02）：按用户夜间推进授权，完成 H2/解析饱和、专家命令时序、已有 120 条前缀的输入覆盖、有标签 source 首步控制及公开夹爪输入替换。错误符号强饱和占比 task 4 为 1.54%–2.41%、task 7 为 8.33%–12.40%，均低于预设训练条件；两个任务的 BCE 分支均停止。open 匹配命令偏迟中位数为 1.5–5 步，仍不能等同物理 release 或闭环因果。相对 source 前 20 步的初始覆盖差异在首个动作之前存在，但 task 4 的全时序参照大多不超线；公开夹爪输入固定虽使局部 MSE 六模型下降，sign error 各 task 一 seed 增加。实际 996 次完整前向和 2,706 次部分读出链，0 新 optimizer/环境步骤；反馈 JSON 类型错误通过完整缓存恢复，不重跑模型，历史失败保留。三份独立审计与 532 项 CPU 回归通过；下一步先冻结 source 稳定化匹配对照，保留 500 updates。结果、完整设置、计算/读取预算、封存边界及下一项范围见[晨间汇总](libero-overnight-diagnostic-progress-2026-11-02.md)。

后续执行更新（2026-11-02）：上述稳定化对照完成原计划 source 采集，零步精确复现门槛全部通过；哈希 helper 的 Path 类型错误发生在模型读出前，保留失败并仅从四份完整公共 cache 续算，没有重复初态读取/环境采集。主分析六模型的初始距离和超线比例均下降，task 4 一 seed 的 reference95 亦大幅增加，故不把零超线称作完全覆盖。固定 Policy 的 action L2 差 task 4 为 `0.104–0.125`、task 7 为 `0.335–0.562`，均无夹爪符号切换；不评分旧 action[0]，不新增成功率或自动训练。全部近邻/原读出及三 seed 判断经独立复核；实际预算、55 项专项相关及 566 项 CPU 回归见[source 稳定化执行报告](libero-task4-task7-source-stabilization-control-diagnostic-2026-11-02.md)。下一项先冻结公开缓存上的 RGB/本体响应分解，保持 500 updates。

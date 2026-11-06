# LIBERO 500-update 后续计划与证据复核（2026-11-01）

几何/接触记录核对完成（2026-11-06）：[字段、证据与下一步](libero-formal-G-geometry-record-availability-audit-2026-11-06.md)。G逐步归档只有动作及joint/gripper，没有末端、物体、接触或RGB帧；历史专家ee_pos及task4/7图像不能拼接成当前36对的几何真值，因此尚不能检验close与实际接近是否协调。本次0原轨迹重读/训练/前向/环境/私有状态，115项重点测试、1648项CPU回归通过（不含可选Crafter收集项）。保持500/5步/现有loss、G阴性停止及封存角色；下一步先另行设计来源candidate的有限几何观测与时钟对齐，尚未冻结或采集，不追加validation或训练。

36对较晚close反例分段完成（2026-11-06）：[只读结果与事件邻域](libero-formal-G-close-runs-late36-diagnostic-2026-11-06.md)。31对输家首段close仍持续至赢家首次close，4对为已结束的单步脉冲，1对为已结束的两步段；首次多步段口径下赢家仍较晚33对、较早2对、同步1对。30条赢家首段因成功终止右删失，不能用观察长度决定保持时长。72条旧轨迹、254段、858行邻域，84项相关测试及15904项独立表级核对通过；原始第二遍复核无完成记录，未计通过、未重试，详见报告。0新训练/Policy前向/环境，G未通过及held-out/qualification封存保持。下一步先明确close与接近物体的几何协调假设和合法观测需求，现有joint/gripper不能替代目标/接触证据，不立即改loss或重训。

validation B/C 800对只读轨迹诊断完成（2026-11-06）：[完整结果与反例](libero-formal-G-validation-BC-trace-diagnostic-2026-11-06.md)。1600条公开轨迹校验通过；相同reset下800对首步平移动作已不同，162对成败不同中155对前20步夹爪符号始终相同。赢家较早close较常见，但36对较晚、9对同一步；83对open因提前终止不可判序，264/326条成功轨迹终止前没有close后的open命令。93对赢家前20步joint路径更短、69对更长，不能直接统一提前close或抑制运动。39项新增、109项相关测试与66010项独立检查通过；0新训练/Policy前向/环境重放，G未通过及held-out/qualification封存保持。后续36对close段与事件邻域已完成，见上方新报告；原G结果保持。

G任务/seed只读诊断完成（2026-11-06）：[分析与全部seed表](libero-formal-G-task-seed-differences-2026-11-06.md)。保留全部20个seed，task2的B−C为7胜/3平/10负、task6为9胜/9平/2负；只有4个seed一任务输一任务赢，3个两任务都赢，均值抵消不是统一根因。17个source护栏违反分布在12个seed；验证表现存在组间差异，也保留违反组5个B−A正向及未违反组1个负向反例。固定专家MSE与闭环方向混合，不据此改loss或加预算。31项新增、80项相关测试及21318项独立核对通过；0新预测/训练/环境/初态，G未通过与H/qualification封存保持。后续800对公开轨迹诊断已完成，见上方新报告，原G结果保持。

G阶段完整复核、准入未通过（2026-11-06）：[最终结果与停止依据](libero-formal-G-results-2026-11-06.md)。320个Policy各500次适应、6400条闭环齐全，原始及本次独立训练/完整审计均通过。validation任务2/6的A/B/C等权成功率为21.125%/20.375%/20.375%；B−A为−0.75个百分点、B−C为0，严格胜出11/20和10/20，均未达预设+5个百分点及16/20门槛。source另有17/100个task×seed配对退化超过10个百分点。累计活动约8小时3分37秒，167000次优化更新及固定预算完成；原审计字段故障与修复记录保留，0重复source训练。按冻结规则结束当前配置，H主检验未开展，held-out/qualification/reserved继续封存；不追加训练、不接受活动SPT/Knowledge/Module。后续只读任务/seed解释已完成，见上方新诊断；原准入和停止决定保持。

以下为启动前冻结快照，实际执行进度以上方记录为准。

正式验证方案v2已冻结（2026-11-05）：[当前有效协议](libero-formal-transfer-protocol-v2-2026-11-05.md)修正v1复用准入/最终训练seed的条件筛选风险；v1未执行且原样保留。G/H各20个独立seed及全新来源生成器，保持每Policy500和5步稳定化；先source非退化+validation2/6，准入通过才做held-out5/8/9。H以20个训练重复作单侧精确胜出概率检验+Holmα0.05，另要求B−A/B−C观测平均增益≥5个百分点和逐task退化护栏。全部阶段上限500适应Policy、250000适应+4000生成器+10000临时SGD=264000更新，6400+5400条闭环。80项相关、1459项CPU回归及32组独立数学核对通过；仅metadata预检，0研究训练/环境/封存payload。执行入口尚待实现/绑定；qualification预算0，不自动写入SPT/Knowledge/Module。


四任务A/B/C收尾完成（2026-11-05）：[完整结果](libero-four-task-abc-completion-2026-11-05.md)已补齐task4/7的18个500-update Policy及360条fresh闭环，两份独立审计通过；task0=8/12/12; task1=0/11/1; task4=0/1/1; task7=0/0/0（A/B/C，各60条）。维持5步稳定化/现有loss，task0/1几何覆盖与task4/7几何外推分层，不合并总成功率或挑seed。48项训练相关、133项闭环相关、1351项CPU回归及12项汇总测试通过；已交付36条seed端点、24行四格及360行曲线CSV。本周source内部验证收尾，正式validation/held-out/qualification仍封存；下一步冻结正式方法/预算/统计与停止规则，不自动接受SPT/Knowledge/Module或继续零散调参。



最新横向整理决定（2026-11-05）：[task0/1/4/7结果总表](libero-task0-task1-task4-task7-results-comparison-2026-11-05.md)已只读核对6个原始JSON，按episode重算10个方法×任务格、30条seed记录，另导出中文四任务CSV。task0/1 A/B/C为8/12/12及0/11/1，task4/7 BC/jitter为3/3及4/3（各60条），协议、source更新、初始化和帧口径不同；不填缺项、不合并总成功率或给任务排难度。无新训练/rollout或封存访问；原观测层诊断仍待冻结。若要回答四task的同方法稳定优势，需另冻补齐task4/7 A/B/C及来源几何/预算，当前不自动追加实验。

最新表示敏感性决定（2026-11-05）：[固定候选头60批读出](libero-task1-b-history-input-sensitivity-results-2026-11-05.md)已完成并审计；0.05全窗口同能量的历史−重复当前平移成本三seed为−7.38e-7/−8.93e-7/−1.89e-6，0.01亦同向。未支持此代理下历史更脆弱，局部分层反证和所有负对照保留；相同线性logit变化不等于相同tanh动作响应。4221928项独立检查、63项相关测试通过，0新训练/CNN/环境，不扩大探针幅度搜索。下一步建议原始公开观测层另冻小窗口、单分量及CNN预算，再决定训练修改；500/5步/现有loss和封存角色保持。

前置敏感性冻结快照（2026-11-05，执行结果见上）：[同一候选头历史输入表示敏感性](libero-task1-b-history-input-sensitivity-diagnostic-2026-11-05.md)已锁定三seed和全部2518帧/seed，正负成对的0.01/0.05固定Rademacher方向分别作用于当前/过去/全窗口。主比较全窗口0.05同能量下相对干净目标误差成本，同时报告动作能量，保留线性logit一致性与过去路由负对照。60批151080行、84张逻辑表，58项相关测试及metadata通过，未解码缓存或执行研究读出。下一步按冻结入口运行并独立审计；不把表示探针写成真实传感器/部署shift根因，继续500/5步/现有loss及封存角色。

最新专家读出决定（2026-11-04）：[固定候选头双输入目标误差](libero-task1-b-fixed-head-expert-target-error-results-2026-11-04.md)6批15108行完成且独立审计通过。重复当前平移三seed增加+40.07%/+31.05%/+20.91%，均值+30.74%，七维均值+12.47%；每seed20个demo均同向，已知open段仍有反向局部收益。497380项独立检查、85项相关测试通过；0训练/CNN/环境。当前保持500/5步/现有loss，不用seed24四条闭环成功推断专家动作更准确；后续机制建议另冻观测分布变化下的历史敏感性，尚未冻结或运行，不接受SPT/Knowledge/Module，正式角色继续封存。

前置冻结决定快照（2026-11-04，执行结果见上）：[同一候选头双输入专家目标误差](libero-task1-b-fixed-head-expert-target-error-diagnostic-2026-11-04.md)已固定task1三seed候选500头、全部source-query2518个完整历史帧/seed，使用同一专家target，夹爪始终当前并精确复用。平移主指标、旋转/七维与逐demo次指标，保留三seed及全部预定分层反证；6批/15108行矩阵预算，0 CNN/训练/环境。77项相关测试、metadata预检通过，实际研究读出/缓存解码0。下一步按绑定入口运行并独立复核，再决定训练修改；继续500/5步/现有loss，旧候选停止及封存角色保持。

前序执行决定（2026-11-04）：[同一候选头真实历史/重复当前闭环](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)完整120条及独立审计通过：0/60对4/60，四个repeat-only全在seed24，三seed0→0/0→0/0→4，不接受稳定改进。60对第5步运动与公开状态分叉、当步夹爪相同；56对仍共同失败，不能用close提前或内容项大作为统一修复。1147101项独立检查、77项相关专项通过，实际35470动作+600零动作、625.70s，无新训练。下一步建议另冻固定合法专家观察的同头两输入目标误差对照，仍500/5步/现有loss；不自动重训或恢复候选接受，封存角色保持。

前置冻结决定快照（2026-11-04，执行结果见上）：[同一候选权重、真实历史/重复当前闭环](libero-task1-b-fixed-head-input-closed-loop-diagnostic-2026-11-04.md)已独立冻结120条fresh配对、最多36000 Policy动作+600零动作；三seed同500端点、前4动作共同启动，第5步只改运动输入，夹爪当前。metadata预检、65项相关专项及1212项CPU回归通过；研究前向/训练/环境/初态解码0，未运行。下一步检查GPU1后按已绑定入口执行，再核对四格、完整动作/终止差异和参数预算；500/5步/现有loss、旧候选停止及封存角色保持。

最新交叉读出决定（2026-11-04）：[共同第5步固定候选头结果](libero-task1-b-step5-crossed-readout-diagnostic-2026-11-04.md)已完成3批60行H(C)和6批120行重放，运动内容项在50/60范数更大，seed23则两项各10；主夹爪内容0，原记录舍入单列。66项测试与280781项独立检查通过，0 CNN/训练/环境。当前不改500/5步/loss，不把较大内容项当闭环根因或自动删历史。下一建议另冻同一候选权重下真实历史/重复当前fresh配对闭环，共120 episodes、0训练，同seed/场景/四步启动；它是独立机制对照，不改变旧候选失败及停止决定，现已独立冻结（见上），尚未执行，封存角色保持。

最新只读决定（2026-11-04）：[60对路由轨迹分解](libero-task1-b-routed-paired-traces-diagnostic-2026-11-04.md)已完成。60/60对第5步共同观察下出现主要平移差，前20步夹爪符号仍相同；3个baseline-only close时机分别晚2/166、早1步，完整open缺失与joint距离也不构成通用失败规则。全部57对共同失败保留，57项测试与558511项独立检查通过，0新模型/训练/环境/NPZ输入读取。当前不改loss或500/5步、继续停止路由候选；下一项建议另冻共同第5步固定候选运动头下真实历史/重复当前读出，复用已记录动作，区分已学权重差与内容影响，不能把代数参照当物理因果。

最新闭环决定（2026-11-04）：[路由读出120条fresh配对](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)已完成，baseline3/60、候选0/60，三seed2→0、0→0、1→0；同场景baseline-only3、candidate-only0、共同失败57。技术门槛及前置runtime接口全部通过，原权重不变，无补跑；停止该路由配置，不以离线MSE选继续训练或扩大预算。1149项CPU回归与1184391项独立检查通过，实际35489个Policy动作、600零动作、639.92s。下一步先另冻仅已有全部60对的只读分析，单列3个baseline-only，检查共同前5个观察后的动作/close/drift分歧；不新增CNN/环境/训练、保留500/5步/现有loss及正式封存角色。

最新重跑决定（2026-11-04）：[运动历史／夹爪当前配对训练](libero-task1-b-routed-history-readout-diagnostic-2026-11-04.md)完成24个新读出各500 updates。先三折支持gate通过，再6个完整fit完成后读取固定query；三seed平移/七维均改善且错号持平，query均值−11.66%/−5.14%、事件仍28/69。新非退化条件通过，但未修复切换或追认旧完整历史候选。12个baseline逐位复现、178项测试和4052292项独立检查通过；CPU9.49s、0原Policy更新/编码器/环境。下一步只固定6个全support500端点，先冻runtime历史缓存及前4步启动接口并数值核对，随后另冻candidate闭环；暂不再搜loss/预算，正式角色继续封存。

最新只读决定（2026-11-04）：[query事件margin/完整时序](libero-task1-b-query-event-margin-timing-diagnostic-2026-11-04.md)完成69事件配对及696完整段帧。6个新增、4个纠正，MSE主要由24个仍错病例的幅度变化降低；新增错号当前margin0.078–0.312，恢复含lag1/3/5及一条段尾未见，不可统称单帧噪声。58项测试、59399项独立检查通过，0新增模型/训练/HDF5/feature调用。保持500/5步/现有loss和此前query失败停止决定。下一机制先冻固定历史头下“重复当前/真实历史”的缓存读出，分开内容与权重差；拟新增3批7554行，尚未执行，不进入loss或历史长度搜索。

最新停止决定（2026-11-04）：[task1 B完整support/query结果](libero-task1-b-support-query-history-readout-diagnostic-2026-11-04.md)已完成6个读出各500 updates，并在所有fit/CPU桥接通过后读取source-query。query事件错误三seed11→9、7→9、10→12，只有1/3改善，joint_query_cue=false；其他主误差改善不能补救。按冻结规则不推进本配置闭环、不选seed22或增加预算。CPU26.13s、98项测试与2371项独立检查通过，原Policy更新/环境/初态/private读取0。下一项先冻并只读23个已有query事件的margin/前后时序，解释连续MSE与符号错误的分歧；不让query成为loss调参集。保持500/5步/现有loss及所有封存角色。

最新训练决定（2026-11-04）：[固定表示当前/历史读出](libero-learned-phase-history-readout-diagnostic-2026-11-04.md)已冻结并实际完成108个小模型×500，共54000更新，CPU25.65s，0原Policy重训/完整前向/环境调用。source-support内部三折按完整demo隔离新读出fit/probe，编码器仍已见全部support；18/18端点七维和平移改善，六组仅task1 B满足联合条件，事件错误23/23/21→22/20/19，仍不可靠。91项相关测试和独立权重/预测/角色核对通过。只保留task1 B受限线索，其余五组不按本配置推进历史；下一实验先另冻全support两臂500拟合及合法source-query检查，不能挑最好一折直接闭环。原500/5步/现有loss及封存角色保持。

最新决定（2026-11-04）：[切换点近邻分解](libero-phase-history-transition-neighbors-diagnostic-2026-11-04.md)已完成全部65事件和585个模型×事件，93个由对变错、76个由错变对。整体BA主要由非切换帧改善；历史距离可能用过去4帧的匹配优势换入close邻居，也有反向纠正，不能由此认定记忆根因。原Policy在两种近邻分析下不变，变差组49/93原本Policy就正确，不能将近邻变化当训练结果。v1状态字符串错误在数组读取前停止，v2定义不变并完成；72项相关测试、165968项独立核对通过，0新前向/训练/环境/全库搜索。保持500/5步/现有loss；下一实验建议先冻固定表示上同结构/同预算的可学习当前/历史读出，本轮未启动该训练，封存角色保持。

最新决定（2026-11-04）：[阶段／历史可辨性诊断](libero-source-internal-phase-history-observability-diagnostic-2026-11-04.md)已冻结、绑定并完成819主前向+18重复。总体命令BA在18/18端点提高，但切换响应不一致；仅task1 B/C满足三seed联合条件，task0三臂均未满足。高条件phase BA只说明expert-open上下文易区分，不证明何时切换可辨、必要记忆或闭环根因。56项相关测试和独立近邻/指标审计通过，0新训练/环境调用；继续500/5步/现有loss。下一步只读全部既有事件近邻，检查由对变错/由错变对和原Policy margin，不立即修改网络、loss或恢复已停止分支。

夜间完成决定（2026-11-04）：[错误复核与完整open段结果](libero-source-internal-overnight-error-review-and-full-open-results-2026-11-04.md)完成既定65事件、1755模型×事件段及540初始前缀的冻结分析；原局部15处连接修正为13延长、1桥接、1段尾未解，另2处孤立。原预测和主要阴性判断保持，未发现需重训的实现错误。20/51段未首次正确、25/62段未形成负号尾段，缺失不填0；更少重入不是单独收益。259项专项/回归与150436项独立标量复核通过，0新增研究前向/训练/交互/源数据调用。继续500/5步/现有loss，收束输入回退，后续检验阶段/历史可辨性而不是盲调loss；封存角色保持，当前无遗留训练作业。

前序只读决定（2026-11-04）：[新增翻号margin与事件时序](libero-source-internal-gripper-sign-margin-temporal-diagnostic-2026-11-04.md)完成18模型两条件全部帧/事件核对。主17处原margin0.000803–0.031598，lag3/4/5为1/4/12；可见窗内单帧新增，15处连接既有错close段、2处孤立，lag1五处完全为主条件子集。原错误连续段仍保留，两类margin范围重叠；12处右端截断，不外推窗外持续时间。78项专项、574940项独立标量检查通过，0新前向/训练/源数据调用。其后65事件完整段跟踪与范围更正已完成，见上方；继续500/5步/现有loss，封存角色保持。

前序执行决定（2026-11-04）：[局部gripper受控读出](libero-source-internal-gripper-state-local-control-diagnostic-2026-11-04.md)已实际完成2457+54批次及独立审计。事件锚点和lag1各18模型的平移/夹爪MSE均数值增加，符号错误率无模型下降；主条件17个模型×帧由正确变错、0纠正，六组主cue全部false。原预测、未处理/padding/sham和重复均精确通过；90项专项、605232项独立标量检查，41.77s、0训练/环境。不采用该反馈回退修复，保持500/5步/现有loss。其后新增错号margin与时序核对已完成，见上方结果，封存角色保持。

前序设计决定（2026-11-03）：[输入审计复核和局部gripper受控读出协议](libero-source-internal-input-sensitivity-review-and-gripper-state-control-protocol-2026-11-03.md)在执行前独立冻结。全部18个500端点、task0/1共306个open后expert-open帧，保留原正确/错close/neutral分层；仅替换两维公开gripper，主事件pre-action锚点、次级lag1，RGB/joint/原目标/权重保持。固定2457主前向+54重复，48项专项与独立样本/预算核验通过；冻结时实际新研究调用0。该快照保留，后续入口实现/绑定及实际执行均已完成，见上方结果。继续500/5步/现有loss，封存角色和旧分支停止决定保持。

前序参照决定（2026-11-03）：[非切换帧参照核对](libero-source-internal-nontransition-translation-reference-diagnostic-2026-11-03.md)已完成，0新模型/训练/环境或源数据调用。扣除固定每seed均值后，open后B平移差+0.047595/+0.081137，同专家符号参照+0.049245/+0.064848；A/C同向，简单加性均值差未消除关联。其他阶段存在反号，不能当因果机制或直接改loss；40项专项和482110项独立检查通过。保持500/5步/现有loss，其后输入审计和设计冻结已完成，见上方进度，新受控读出尚未执行。

前序只读决定（2026-11-03）：[事件前后和同帧误差](libero-source-internal-event-phase-direction-translation-diagnostic-2026-11-03.md)已完成。open后错号与平移偏高在两task三臂均有同帧跨seed正均值；B共同错号12/17帧、来自4/8个demo，排除重叠后仍见。open当步和close后有反例，不统一改gripper loss。42项专项、191092项独立检查通过，0新研究调用；保持500/5步。该阶段提出的非切换帧参照核对现已完成，见上方进度；未追加训练或开放封存角色。

前序读出决定（2026-11-03）：[逐帧专家误差定位](libero-source-internal-fixed500-expert-frame-target-error-diagnostic-2026-11-03.md)先冻819主批次+18重复并绑定代码后完成。所有臂的open窗口四项主误差均集中，close窗口夹爪误差集中；早期符号0错误伴随专家目标全open，不能推到runtime首动作。B open附近两种错误方向随seed反转，保持500/5步、现有loss和已停止分支决定。该阶段提出的事件前后只读分解现已完成，见上方进度；61项专项和294770项独立检查通过，0训练/交互、封存角色保持。

前序只读决定（2026-11-03）：[固定专家输入0/500点误差分解](libero-source-internal-initialization-endpoint-target-error-diagnostic-2026-11-03.md)已冻结并执行，发现原产物已有36组同support输入指标，故新增前向预算冻结为0。B-A初始平移/夹爪优势在500步后平均缩小，B-C初始夹爪六组均更差、最终各task只有2/3 seed更低；误差与闭环方向相反案例保留，不据此改loss或预算。59项相关测试和2930项独立检查通过。保持500与5步；后续逐帧读出已按新冻结协议执行，见上方进度。

前序只读决定（2026-11-03）：[六组正反配对轨迹分析](libero-source-internal-six-group-paired-trace-diagnostic-2026-11-03.md)已完成并独立复核。首步映射、命令进入/维持和公开状态路径均有seed差异，但早close、持续close或小位移没有统一成功规律。因此保持500与当前loss/观测，不追加训练；该阶段建议先冻固定合法专家观察上的0/500点目标动作误差对照。分析读取原360条轨迹，新增Policy/环境/训练/HDF5/RGB/初态/checkpoint调用0，813项CPU回归通过；后续误差分解见上。

最新闭环结果（2026-11-03）：[全部360 episodes](libero-source-internal-spt-spi-closed-loop-diagnostic-2026-11-03.md)已实际执行并独立复核。task0 A/B/C为8/60、12/60、12/60，task1为0/60、11/60、1/60；B收益有明显seed依赖，没有三seed一致优势。固定18个500端点、5步稳定化和封存角色不变，实际99457个Policy动作与1800个零动作，0新训练。下一步只读六组完整正反配对轨迹定位seed差异；不追加预算或挑选checkpoint。

冻结阶段记录（2026-11-03）：[360 episodes协议](libero-source-internal-spt-spi-closed-loop-protocol-2026-11-03.md)当时已冻结并接入入口，实际交互尚为0。18个500端点、5步稳定化、300步上限与candidate初态0..19均固定；74项专项、781项CPU回归通过。该执行前快照保留，完整闭环结果见上，未追加训练或选择模型。

主线推进更新（2026-11-03）：[夹爪映射](libero-gripper-action-mapping-audit-2026-11-03.md)已核对，同号幅值差在相同控制历史下不改变增量；按用户同意的计划进入[source内部SPT/SPI诊断](libero-source-internal-spt-spi-diagnostic-2026-11-03.md)，并完成18个Policy各500 updates及独立审计。B相对A有早期曲线收益，但未形成相对普通预训练C的稳定优势，不接受活动SPT。当前下一步先冻结同18个端点的source candidate闭环协议；本轮环境交互0，正式角色仍封存。

前序闭环快照（2026-11-02）：[当前环境下0/5闭环成功率对照](libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic-2026-11-02.md)已完成并复核，两个arm均为fresh 7/120。没有三seed一致的零步改善，保留现行5步、原baseline500。下一步只读全部120对公开action/state轨迹，将6组相反结果单列，检查首动作、共同仿真时钟下的初始变化与close/release；无需新环境。此前两次完整历史复现failed保持，不改判；本次主分母不复用旧成功率。实际预算、614项CPU回归与独立审计见新报告，封存角色和SPT/Knowledge/Module边界保持。

更新至2026-11-03：[task7缓存分层定位](libero-task7-expert-camera-component-tradeoff-diagnostic-2026-11-03.md)已完成并独立复核。seed24的夹爪恶化主要见于expert close，open及open切换窗口整体改善；三个seed在非切换close帧上有幅值偏离的共同局部模式，但全局平移/夹爪方向不同。新模型/训练/交互/HDF5调用0，690项CPU回归通过；保持500 updates与闭环5步。下一项先只读核对公开夹爪动作映射和既有审计，确认同号幅值差是否改变实际命令，暂不改训练。原设计、结果和所有停止决定保持。

## 初次只读核对发现（历史快照，修复见下文）

检查了原 baseline/jitter 闭环 JSON 与 RGB capture v3 的全部 14 个 archive。先核对源文件与 archive SHA-256，再比较 action steps `1..20` 和对应 pre-action joint/gripper steps `0..19`；没有读取环境初态、执行 rollout 或加载训练数据。

| 检查 | 结果 |
|---|---:|
| 新采集两臂之间 reset digest 一致 | `7/7` 对 |
| 与原始闭环 reset digest、action、joint/gripper 前缀完全一致 | `10/14` 个窗口 |
| task 7 与原始前缀完全一致 | `10/10` 个窗口，即 `5` 对 |
| task 4 与原始前缀完全一致 | `0/4` 个窗口，即 `2` 对未复现 |

task 4 两个选中案例为 seed `20261023` / state `9`、seed `20261024` / state `17`，两臂均未复现源前缀，最大 action 分量绝对差 `0.099773`。新采集使用选中状态的 20 步窗口，源闭环按全部 `0..19` 状态复用 adapter、每集最多 300 步；adapter 历史不同是需要核对的协议差异，目前未确认它是偏差原因。

证据：`/tmp/libero-task4-task7-transition-rgb-capture-source-prefix-audit-20261101-v1.json`，SHA-256 `73a89e09b0e1286327592d7d9d2a68d80b7b6ff9514f9108b5ccced0f3bbc31a`。原始 capture、analysis、checkpoint 均保留且未改写。

因此，旧报告的两组 RGB/action 汇总仍是新采集窗口上的局部描述，但不能全部归属于原来 7 条 success transition 的精确复现；task 4 需要单独标注。眼在手图像像素差较大也不能直接证明该相机对动作更重要：当前 analyzer 的 RGB-only swap 同时替换两路相机，未完成逐相机 action swap。

## 下一步顺序

1. **复现检查已补齐。** 源 reset、action、pre-action 公共状态和权重 hash 检查已接入新版本采集；不能通过放宽容差认定复现成功。task 4 另行冻结有界协议后，恢复原状态顺序，新增 56 个短窗口，四个目标 archive 均通过门槛。旧失败窗口仍保留。

2. **修复后的 task 4 离线分解已完成。** task 4 两对与 task 7 五对均完成 `baseline/jitter Policy × baseline/jitter observation` 四格读出及逐相机/公开状态替换，task 7 重算与旧结果精确一致。新旧 archive 来源明确，旧 14 窗口结果未改写。Policy 变化项与观察变化项均已报告；L2 范数不能直接相加作贡献百分比。这些是输入敏感性诊断，不是环境因果机制。

3. **专家目标误差与局部损失对照均已完成。** 12 个 task 4/7 × 三 seed × baseline/jitter 的既有模型比较揭示 open 切换误差热点，但后续 baseline/局部 2× gripper 的 12 模型配对训练未通过预设门槛，已停止该分支。当前 jitter 闭环 `6/120`、baseline `7/120` 的结论不变；不因单 seed、局部均值或 query 曲线选择 checkpoint，也不继续搜索附近权重。详见[专家目标误差](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)与[局部损失负结果](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)。

4. **固定权重梯度诊断已完成。** 六个 baseline 500 checkpoints 在 source-support `t>=1` 的完整读出与独立复核通过。两个任务的共享整体主指标均未形成三 seed 负向结构；task 7 逐帧投影层及部分窗口的负向结果保留为次级线索，不据此覆盖主指标或改训练。虚拟裁剪与原 Adam 轨迹分开解释，详见[梯度诊断结果](libero-task4-task7-gradient-interference-diagnostic-2026-11-01.md)。

5. **H2 及后续只读分解已完成。** 六个固定 baseline500 没有形成预设的 projection 区分一致下降，错误 tanh 强饱和未达到 BCE 分支条件，故不训练该候选。专家观察上的 open 命令偏迟、初始覆盖差异和夹爪反馈敏感性均已分开记录；参照依赖、种子差异和符号错误反证保留。完整矩阵、失败导出恢复和独立复核见[夜间汇总](libero-overnight-diagnostic-progress-2026-11-02.md)，不把近邻混叠或输入替换称为闭环根因。

6. **source 稳定化对照已完成。** 初态、horizon、顺序、bank/reference 规模匹配，60 个零步公开观察和六模型旧读出精确复现，两个任务三 seed 的共同尺度初始距离均下降。保留 task 4 reference95 膨胀的反证，不称完全覆盖或闭环因果；原 Path 类型错误和零环境续算单列 provenance。实际成本与判断见[稳定化执行报告](libero-task4-task7-source-stabilization-control-diagnostic-2026-11-02.md)。

7. **已完成当前栈两臂均fresh的闭环对照。** 完整历史位级复现两次失败后保持原失败，另冻同设备、同权重、同初态和horizon的当前比较，不复用旧成功率。0/5均为7/120，各有3组初态只在一臂成功，任务与seed方向不一致；不支持去掉五步即可修复整体可靠性。完整结果与成本见[当前闭环对照](libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic-2026-11-02.md)。

8. **现有120对公开轨迹只读分解已完成。** [报告](libero-task4-task7-stabilization-trace-alignment-diagnostic-2026-11-02.md)单列六组相反结果；首动作、close早晚和首段持续长度没有统一成功规律。首步符号变化12对中11对共同失败，不足以区分结果。Policy序号、共同时钟、事件缺失及结束截断均分开，独立复算和631项CPU回归通过。新增研究训练/交互0，保留5步与500 updates，不据outcome-stratified六组调参。

9. **source缓存输入分解已完成。** [结果](libero-task4-task7-source-stabilization-input-decomposition-diagnostic-2026-11-02.md)显示三seed、两参照下眼在手替换响应最强，RGB大于本体；本轮全无夹爪符号变化，不能解释六组闭环反转。48主前向和12重复完成、自然端点精确匹配，649项CPU回归通过，训练/环境新增0。原4像素增强及局部夹爪候选停止决定不变。

10. **合法专家相机—目标误差诊断已执行。** [报告](libero-task4-task7-expert-camera-target-error-diagnostic-2026-11-02.md)完成原定7200主前向+48重复、208656行预测；两个任务、两个相机的平移主flag均未形成三seed一致，强信息损失probe变差但不证明几何必要性。七维局部线索与平移/夹爪取舍单列，不覆盖主指标。原科学设计和算子hash不变，0训练/环境，677项CPU回归及独立复算通过。保持baseline500与闭环5步，下一步只读已有命令/逐demo分层定位task7分量取舍，不追加前向、标签或自动恢复已停止训练。

11. **task7既有分层只读定位已完成。** [报告](libero-task7-expert-camera-component-tradeoff-diagnostic-2026-11-03.md)区分同帧取舍、帧组抵消、窗口重叠、符号与幅值；seed24夹爪净恶化主要见于expert close，非open切换窗口整体恶化。三个seed有共同局部模式，仍无共同全局方向；原主flag、4像素增强和局部夹爪候选停止决定保持。新模型/训练/环境调用0，690项CPU回归通过。下一步先只读核对runtime公开夹爪动作映射和既有审计，确认同号幅值差是否改变真实命令；暂不改训练。

12. **动作映射核对与source内部SPT/SPI对照已完成。** 根据用户同意推进主线的指示，收束局部幅值诊断，复用当前CNN-only设置执行原始SPT、source反馈更新的隔离FOMAML候选、匹配数据/容量/梯度预算的普通预训练。task0/1为已暴露source内部探针，18个Policy各500 updates，完整曲线、108份权重与180组指标独立核对，739项CPU回归通过；B相对C无稳定优势，不接受候选版本。失败和执行修订均保留，详见[结果](libero-source-internal-spt-spi-diagnostic-2026-11-03.md)。这不是正式迁移或Module qualification。

13. **source闭环360 episodes已完成。** 固定全部18个500端点、candidate状态0..19、5步稳定化、300步horizon和环境seed20261022；实际99457个Policy动作、1800个零动作和320次reset比较通过，无技术中止或补样本。task0 A/B/C为8/12/12、task1为0/11/1（各60条），B收益集中于个别seed；完整seed和两方向配对见[结果](libero-source-internal-spt-spi-closed-loop-diagnostic-2026-11-03.md)。独立复核全部360份轨迹和18份权重，792项CPU回归通过。下一步只读全部六组action/close/公开状态漂移，解释不稳定性，不新增训练或环境交互、不选择seed或checkpoint；不用qualification状态。

14. **六组配对轨迹只读分析已完成，不改训练。** 全部六组B-A/B-C与三臂全部seed两两对照均保留，360条episode特征、240条主配对、360条次级配对和空/删失分层均复核。task0 seed22缺少close与收益相伴，但task1 seed24的B-only逐对更晚close；同首步符号、相同close/release时刻及相近早期位移也有成败差异。详见[报告](libero-source-internal-six-group-paired-trace-diagnostic-2026-11-03.md)。保持500和当前loss；下一项先准备合法source-support专家观察0/500点动作目标误差诊断，明确前向预算与hash后再执行。本轮没有新增模型/交互或训练。

15. **0/500点目标误差分解已完成。** 固定六组、三臂及36组历史source-support误差，先冻结定义/代码再做算术分解，新增主/重复前向及训练/数据payload调用0。B-A优势平均缩小、B-C夹爪相对下降未形成三seed稳定端点收益，全部反证和原闭环上下文保留，详见[报告](libero-source-internal-initialization-endpoint-target-error-diagnostic-2026-11-03.md)。当前不改训练；若继续，需另冻逐帧专家命令分层读出，才能查早期/close/open窗口，而不是由汇总猜测时序根因。

16. **固定500端点的逐帧专家误差已完成。** 科学配置/窗口/算子与执行代码在读取本轮数值payload前冻结，实际819+18批次、62586+1440有效预测行。全部三臂两任务的open窗口四项主误差三seed一致集中，close夹爪集中、早期符号全对但标签全open；错误方向反转与action0独立复核范围单列，详见[报告](libero-source-internal-fixed500-expert-frame-target-error-diagnostic-2026-11-03.md)。不改训练，下一步只读现有NPZ查事件前后、两种符号错误及机械臂同步误差，不追加前向。

17. **事件前后、错号方向和同帧平移分解已完成。** 仅既有NPZ，主open/次close、原窗口/排除重叠口径均保留。open后expert open预测close与更高平移误差的同帧跨seed均值两task三臂均正，open当步与close后有反向结果；全部分母、空组、demo和交集见[报告](libero-source-internal-event-phase-direction-translation-diagnostic-2026-11-03.md)。保持500/现有loss/5步，下一项可只读非切换帧参照排查seed整体拟合混杂，尚未执行。

18. **非切换帧参照排查已完成，暂不改训练。** 原192个同帧组/两种固定参照均保留，未改变帧ID或seed集合，open后关联在两task三臂仍为正均值；不能由该结果排除所有整体拟合差异。控制组反号、缺失与全部基线见[报告](libero-source-internal-nontransition-translation-reference-diagnostic-2026-11-03.md)。下一项应复核已有输入敏感性审计，再冻结固定Policy单一观测分量受控读出，而不是继续调整loss或选择参照；保持500/5步，尚未运行新前向。

19. **输入审计复核与局部gripper控制设计已冻结。** [协议](libero-source-internal-input-sensitivity-review-and-gripper-state-control-protocol-2026-11-03.md)保留旧审计反证和停止规则，不按最大响应或连续夹爪MSE选训练。固定task0/1、18个500端点、全部306个open后expert-open帧；只替换公开gripper为原事件/lag1过去观察，主次条件、原符号分层及2457+54批次均锁定。48项专项和独立样本/预算核对通过，本轮实际新Policy/源观察/训练/环境调用0。下一步实现并绑定执行runner，再运行冻结读出，仍不直接改训练。

20. **局部gripper控制读出已绑定并执行。** [结果](libero-source-internal-gripper-state-local-control-diagnostic-2026-11-04.md)完成原2457+54批次，无新增训练/交互。主事件锚点和lag1各18模型都没有符号错误率改善，平移/夹爪MSE均数值上升；主六组cue全部false，原预测与所有精确对照通过。90项专项和605232项独立标量核对通过；保持500/5步/现有loss。下一步只读本次新增错号的原输出margin、lag及原正确/错误组响应，区分临界翻号和持续错误，不立即转为训练或新前向。

21. **新增错号的margin与事件内连续性已只读核对。** [结果](libero-source-internal-gripper-sign-margin-temporal-diagnostic-2026-11-04.md)保留全部18模型与两条件，主17处在冻结1..5步内均单帧新增、15处延长原错段，原错仍错和连续段另列；次级五处不重复计作独立新病例。无事后近0阈值、无跨模型/缺帧拼接；78项专项与574940项独立标量核对通过，新增研究调用0。12处新增在右端，下一项先冻完整expert-open命令段的只读跟踪定义，核对恢复、重入及持续时间截断；继续500/5步/现有loss。

22. **完整expert-open命令段及错误复核已完成。** [修正报告](libero-source-internal-overnight-error-review-and-full-open-results-2026-11-04.md)补入事件当步、完整命令段、初始open前缀和恢复/重入/边界指标；原17处主新增细分为13延长、1桥接、2孤立、1末端未解。未发现需重训的数值实现错误，不采用当前反馈回退修复。259项专项/回归与150436项独立标量核验通过，0新增研究调用；保持500/5步/现有loss。下一机制应检验阶段/历史可辨性，不由此宣称记忆根因或继续搜索局部loss。

23. **阶段／历史可辨性已按新协议执行。** [结果](libero-source-internal-phase-history-observability-diagnostic-2026-11-04.md)固定全部18个500端点、source-support共同t>=5和同一5NN规则，当前表示与当前加过去4帧配对。全部模型总体命令BA提高，但六组只有task1 B/C符合联合条件；task0三臂平均事件当步错误增加，不能由整体读出推断记忆缺失是根因。原预测62586行与18批动作/表示重复逐位一致，56项相关测试和独立审计通过。下一步仅利用已保存的65事件近邻和margin解释取舍，任何新分层先冻结，不搜索k/历史长度或直接改训练；500/5步/现有loss保持。

24. **全部切换点近邻只读分解已完成。** [报告](libero-phase-history-transition-neighbors-diagnostic-2026-11-04.md)保留585个模型×事件、60426个整体命令帧和195个三seed事件模式。93个变差、76个改善；非切换帧BA贡献覆盖部分事件损失，全部10175个进入/离开配对显示过去距离优势可以反转当前距离偏好。其为固定5NN规则的算术机制，不等于真实Policy采用滞后控制或记忆缺失。v1父状态检查错误保留，v2科学定义不变；72项相关测试和165968项独立检查通过，0新研究前向/训练/环境。当前不改Policy，若推进下一训练先另冻固定表示的可学习当前/历史读出，匹配结构/参数/样本/seed/500预算并同时约束切换和全动作误差，不搜索近邻参数或自动追加闭环。

25. **固定表示当前/历史监督读出已实际训练。** [报告](libero-learned-phase-history-readout-diagnostic-2026-11-04.md)保留108个读出各500 updates、3折fit/probe曲线、全部18端点合并评价和六组三seed门槛。整体动作MSE和平移改善18/18，task0切换不稳定，仅task1 B满足联合条件（1/6），仍有61/102事件当步错号。没有新的原Policy前向、更新或环境交互，91项相关测试和45029项独立检查通过；原500/5步/现有loss保持。下一实验只保留task1 B线索，先另冻全support当前/历史两臂三seed500拟合及source-query固定端点检查，再决定是否另冻candidate闭环；不选择3折中最好模型或按整体MSE自动推广历史。

26. **task1 B完整support/query检查已执行并停止候选推进。** [结果](libero-task1-b-support-query-history-readout-diagnostic-2026-11-04.md)新训练三seed×当前/历史两臂各500，所有6个fit完成后经数值桥接才读query。23事件/seed的错误11→9、7→9、10→12，虽其他三项主指标均改善，事件flag失败、联合cue=false。没有新闭环或原Policy更新，98项相关测试和独立审计通过。下一步只读已有query预测的事件margin与前后时序，不选有利seed或用此query搜索loss/历史长度；继续500/5步/现有loss、封存角色。

27. **query事件margin与完整时序已只读分解。** [报告](libero-task1-b-query-event-margin-timing-diagnostic-2026-11-04.md)把69个事件配对的MSE精确分为35都对、6新增、4纠正、24仍错，核算父事件误差无失配。新增当前|g|0.078–0.312，不以near0阈值分组；完整时序分开额外错号前缀、首次正确、尾段保持及重入资格，避免把一帧额外错号当成已恢复。58项相关测试与59399项独立检查通过，0新模型/训练/源数据/feature调用。下一步先冻固定读出权重的内容/权重分解，仅增加缓存表示读出，不用query搜索loss；500/5步/现有loss及不推进闭环决定保持。

28. **固定权重交叉读出已完成。** [结果](libero-task1-b-crossed-history-readout-diagnostic-2026-11-04.md)新增3批7554行H(C)，原6新增错号全部在同头重复当前参照下纠正；真实历史同时保留整体运动拟合优势。精确动作/logit分解不识别独立物理因果；CPU0.287s、0编码器/训练/环境调用。支持进一步检验分量路由，未直接替换全历史或放宽旧停止规则。

29. **分量路由两臂已按新冻结协议重训。** [结果](libero-task1-b-routed-history-readout-diagnostic-2026-11-04.md)先18个三折500读出、后6个完整support500读出，共12000更新；支持和query全部三seed非退化条件通过。query平移/七维均值改善11.66%/5.14%，事件11/7/10保持、28/69未解决。12个baseline逐位复现，178项测试及4052292项独立核验通过；审计v1字段名错误保留并另存v2，未重训。下一步冻结runtime历史cache/pre-action接口与前4步启动规则并检查数值一致，再独立冻结candidate闭环预算。当前无新成功率、不扩大离线参数搜索，500/5步/现有loss和全部封存角色保持。

30. **runtime历史接口已冻结并通过。** [结果](libero-task1-b-routed-runtime-interface-diagnostic-2026-11-04.md)验证24834次流式读出、246个原编码器前向，源数值输入限80行source-support，0环境/训练。前4步共享baseline、第五步候选成熟历史、reset清空cache，全部既有预测/符号与CPU单步桥接误差<=1e-5；最大1.0431e-6。471463项独立检查通过，接口检查本身不提供成功率。

31. **120条fresh路由配对闭环已完成并停止候选推进。** [结果](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)固定6个500端点、candidate0..19、5步稳定化，三seed两臂均完整执行；baseline3/60、候选0/60，无候选新增成功。100次reset、240次共同四步动作和300次共同前5观察精确比较通过，独立重放全部35489行动作/特征与35609行公开状态，1184391项检查通过。新训练0，639.92s，无技术中止；不因离线收益继续增加updates或改loss。下一步先另冻全部60对的只读比较，保留57对共同失败和3对baseline-only，从共同观察下第5步动作及后续close/drift定位失配，不立即指定唯一根因。

32. **60对路由轨迹只读分解已完成，不改训练。** [报告](libero-task1-b-routed-paired-traces-diagnostic-2026-11-04.md)在全部60对第5步共同观察下定位到动作映射差，前20步夹爪符号均相同；3个baseline-only含close晚2/166与早1，close相近、较小joint距离仍有丢失成功。事件缺失、首段右删失和成功终止按同钟比较，不以57共同失败作独立验证。57项相关测试、558511项独立检查通过，0新研究调用。下一步建议另冻固定候选运动头第5步真实历史/重复当前交叉读出，拟3批60行、0 CNN/环境/训练，复用已记录两臂动作；本轮未读feature/head payload，未运行新读出，不恢复路由候选或改变500/5步/loss。

33. **第5步固定头交叉读出已冻结并完成。** [结果](libero-task1-b-step5-crossed-readout-diagnostic-2026-11-04.md)保留60个场景及三seed，新增H(C)3批60行，已有格6批120行；动作/Logit逐维分解通过。运动内容项50/60更大、43/60两项点积负，夹爪内容严格0；比例可负/超过1，不称因果贡献。66项相关测试、280781项独立审计通过，0新CNN/训练/环境。下一建议另冻同一候选权重、真实历史/重复当前两输入的120条fresh闭环机制对照，保持500/5步/四步启动和封存角色；不自动恢复候选接受、不因接近baseline直接宣称修复，本轮未冻结/执行后续。

34. **同一候选权重双输入闭环的执行前冻结已完成。** [协议](libero-task1-b-fixed-head-input-closed-loop-protocol-2026-11-04.md)固定三seed和全部已用candidate0..19，真实历史/重复当前两臂同一候选500头；共同四步启动、夹爪当前、5步稳定化。120条全新轨迹，最多36000 Policy动作、600零动作、53760内部Linear；技术失配保留并使主汇总无效。入口/新helper/测试/协议已hash绑定并通过metadata、65项相关专项及1212项CPU回归，冻结阶段新增研究调用0；后续执行及审计见下一项，不复用旧0/60或自动恢复候选接受。

35. **同权重双输入120条闭环已完成并独立审计。** [结果](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)保留全部60对，真实历史0/60、重复当前4/60，四例仅seed24的state4/7/11/12，未通过三seed一致条件。60对第5步共同观察下运动分叉，夹爪连续值首差6、符号首差53对；56对共同失败保留。实际35470 Policy动作+600零动作，1147101项独立检查通过、最大动作复算差6.67e-7；0新训练，不据此删历史/加预算。下一项建议另冻固定专家观察的同头两输入目标误差，区分输出变化与准确性，继续500/5步/现有loss和全部封存角色。

36. **同一候选头双输入专家目标误差的执行前冻结已完成。** [协议](libero-task1-b-fixed-head-expert-target-error-protocol-2026-11-04.md)固定全部三seed的候选500头与2518个完整历史专家query帧，三seed7554配对；每seed仅当前与历史两次矩阵读出，共6批15108行。平移MSE为主，旋转/七维、逐demo与原专家分层及方向—能量分解为次；夹爪当前精确复用。77项相关测试与元数据预检通过，未解码缓存或运行研究读出，0 CNN/训练/环境；下一步执行并独立审计，500/5步/现有loss及正式封存角色保持。

37. **同一候选头双输入专家目标误差已完成并独立审计。** [结果](libero-task1-b-fixed-head-expert-target-error-results-2026-11-04.md)保留三seed全部7554专家帧配对，重复当前平移均值增加30.74%、七维12.47%，60/60个demo×模型均平移恶化；已知open段存在反向收益，不统一定义历史有害。497380项独立检查及85项相关专项通过，复算差0、旧预测桥接最大6.78e-7；严格6批15108行，0新CNN/训练/环境。专家分布与闭环排序不同，当前不删历史或改loss/updates；下一机制建议独立冻结历史输入对观测分布变化的敏感性，保持500/5步和全部封存角色。

38. **固定历史输入表示敏感性的执行前冻结已完成。** [协议](libero-task1-b-history-input-sensitivity-protocol-2026-11-05.md)保持三个候选500头和全部source-query帧，固定测量seed及两个幅度的正负成对probe，主比较全窗口同能量扰动下相对干净平移MSE成本。身份/sham/过去负对照/同头logit代数门槛已锁；60批151080行，84张逻辑表有显式复用，58项相关测试和metadata通过，0科学调用。下一步执行后独立审计，再决定原观测级验证或训练修改；不把表示probe当部署shift证明，500/5步/loss及全部封存角色保持。

39. **固定历史输入表示探针已完成并独立审计。** [结果](libero-task1-b-history-input-sensitivity-results-2026-11-05.md)主条件三seed均历史平移成本略低，两个幅度同向；事件/初始open局部反证保留，不解释为部署shift证明或全局鲁棒性胜出。60批151080行、84张逻辑表，4221928项独立检查和63项专项通过，全部重建差0；0新训练/CNN/环境。下一步建议原始公开观测层另冻固定权重的小窗口单分量及CNN预算，500/5步/现有loss和全部封存角色保持。

40. **四个source任务的现有结果已只读横向整理。** [报告](libero-task0-task1-task4-task7-results-comparison-2026-11-05.md)保留task0/1 A/B/C和task4/7 BC/jitter的不同协议、全部seed及missing格；重算10个方法×任务单元、30条seed记录，独立列出fresh稳定化复核与query指标口径。未追加训练或rollout，不用跨协议成功率排名任务难度。若后续补齐四task同方法对比，需另冻source fold、几何覆盖和总预算；原观测机制建议仍待冻结，qualification等继续封存。

41. **正式迁移仍需独立冻结。** 任务角色、内外循环/适应预算、checkpoint、候选接受、qualification阈值与统计方法须在正式评价前冻结；先解决闭环指标的分辨力与可复现性。validation、qualification、held-out和final evaluation继续封存；不注册Module、不更新Knowledge。Knowledge Evolution仍须独立的环境机制证据。

执行进度：严格的源前缀检查及固定 archive 离线分解现已完成。task 7 五对通过，task 4 两对单列；同观察 Policy 差异均值 `0.100411–0.111334`，高于固定 Policy 完整观察替换的 `0.026819–0.037324`。分别替换相机后，眼在手局部动作敏感性也高于外部相机。新脚本没有创建环境或 optimizer，新增 14 个专项测试、相关 85 个测试通过；证据见 [500-update 证据核对与离线分解](libero-task4-task7-evidence-decomposition-diagnostic-2026-11-01.md)。

后续执行进度：task 4 复现修复已完成。恢复从 state 0 开始的环境复用顺序后，`56/56` 个短窗口和四个目标 archive 精确匹配源 reset/action/公开状态；实测原生抽屉属性采样器配置项逐次累积，目标 state 9/17 分别为 11/19 项。没有改变权重、500-update 预算或第三方代码。完整运行 provenance、独立复核索引及 `100 passed` 验证见 [task 4 按原重置顺序复现诊断](libero-task4-ordered-prefix-reproduction-diagnostic-2026-11-01.md)。

后续执行进度：修复后的 task 4 两对离线分解已完成，同观察 Policy 输出差 `0.045174–0.045448` 高于完整观察替换 `0.019082–0.022118`。task 7 五对重算与旧结果精确一致。14 个 archive、8 个 checkpoint 已核验，CPU elapsed `6.59s`，新增训练和交互为 0，相关测试 `115 passed`。详细结果和解释边界见 [修复前缀离线分解](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md)。

后续执行进度：有专家目标的误差诊断已完成，14,490 个 causal 专家输入、86,940 行预测，新增训练/交互为 0，elapsed `219.16s`，相关测试 `136 passed`。168 个分层结果、505 个共享输入批次摘要及 12 个 checkpoint 文件经独立复核。结果、下一候选的精确定义与解释边界见 [专家目标动作误差诊断](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)。

后续执行进度：上述候选已通过独立冻结配置运行，完成 12 模型 × 500 updates。baseline 权重精确匹配；query 局部 open-window gripper MSE 在两个 task 均只改善 2/3 seed，task 4 平移 MSE 均值增加 7.03%，两个 task 的七维 MSE 分别增加 4.19%/6.63%，因此按规则停止该候选分支。没有追加训练或闭环；设计稿和旧 artifact 保留。实际预算、433 项 CPU 回归与独立复核见 [局部损失配对诊断](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)。

后续执行进度：证据收束已完成，11 个既有结果 SHA-256 与 6,000 条梯度范数日志核验一致。局部候选的裁剪次数在 5/6 组增加、1/6 减少，仍没有梯度方向证据。下一项选择共享表示优化干扰的零更新诊断，而不是继续假设夹爪权重不足；当前只完成设计，尚未运行。不会把失败变成附近权重搜索、延长 updates 或选择有利 seed；不提前进入 qualification 或完整 KECRL 主实验。详见[机制证据汇总与下一假设](libero-task4-task7-mechanism-evidence-review-2026-11-01.md)。

后续执行进度：上述梯度诊断已完成。v1 的 14 个已完成窗口与后续数值门槛失败单独保留，同窗口两精度审计后另冻结 v2，用原门槛完成 2,670 个主窗口、6 个重复窗口与 11,454 次反向；全过程 optimizer/环境更新为 0。六个共享整体中位数均为正，30%–43% 局部负向与 task 7 投影层负向另列。16 个输入 pin、890 个 source 摘要、六份原始梯度及全部分层经独立复核，相关测试 134 项通过；下一项按原规则转向 H2 设计，不搜索局部权重或选择 checkpoint。结果、数值修订与完整成本见[梯度诊断](libero-task4-task7-gradient-interference-diagnostic-2026-11-01.md)。

后续执行进度（2026-11-02）：H2、饱和、时序、覆盖、source t0 与夹爪反馈输入检查完成，条件训练均未获完整依据。实际模型前向和部分读出预算、三份独立审计、导出失败的保留与缓存恢复及 532 项 CPU 回归见[晨间报告](libero-overnight-diagnostic-progress-2026-11-02.md)。未新增训练或环境交互，未改变 500-update baseline；下一项先分离稳定化/参照时相的混杂，不提前启动 qualification 或主实验。

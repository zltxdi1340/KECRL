# task7当前／历史方向读出的只读对错转移方案（2026-11-08）

本次冻结科学口径、缓存清单、分析预算及停止规则，**尚未执行真实数组分析**。依据[上轮完整结果](libero-source-expert-component3-observability-results-2026-11-07.md)，task7末20步history5的正召回提高、负召回下降；现有汇总只有净变化，不能告诉我们纠正和新增错误各有多少、分布在哪些示教和步骤。本轮检验这些变化是否集中于相同的目标方向历史、时间位置或参考邻居模式，不预设“缺历史”是根因。

[冻结配置](../configs/libero-task7-component3-history-transition-readonly_v1.json)与[仅元数据验证器](../experiments/libero_task7_component3_history_transition_protocol.py)配套。保持500 updates、未来闭环稳定化5步和原均匀七维masked action MSE；本次无闭环，稳定化不产生额外离线步骤。G未通过/H未执行不变，source meta_query及validation、held-out、qualification、reserved、final evaluation payload继续封存；不更新SPT/Knowledge或注册Module。

## 固定模型、缓存与标签

仅task7 × seed20261106/07/08 × A_original/B_fomaml，全部六个现有500端点。A/B指G原始/候选初始化适应臂，不能混同早期baseline/jitter。不选择模型、seed、checkpoint、帧或示教。六份probe NPZ和一份task7共享ID/标签NPZ逐一绑定路径及SHA；checkpoint只保留父清单中的路径、文件SHA与weight hash，**不打开checkpoint**。

父科学配置、执行绑定、capture、pending-audit结果、已通过独立audit及completion六份JSON均绑定历史SHA，核对整个来源链，不修改其状态或字节。pending-audit主文件由单独audit/completion证明完成，不能回写为新状态。编码器已训练全部30条source meta_support示教，留一仅隔离旧参考库/scaler，因此本分析是**来源训练拟合diagnostic**，不是未见示教泛化、正式迁移、闭环成功或Module资格结果。

沿用`obs[t−1] → action[t]`；t≥5且`abs(action[t,2])≥0.025`的共同material帧为唯一方向评分集。第3分量为数组index2，不直接称世界上下或米制位移。共3598个共享帧，负2273、正1325；末20为477帧，负402、正75；补集为3121帧，负1871、正1250。每个模型使用相同标签/ID，六份预测不是六倍独立示教。原t≥1共享缓存4188行、共同t≥5含中性4068行；470个共同中性帧不评分，但合法历史描述保留中性。

只解码共享文件的targets、demo_indices、action_steps、demo_lengths、tail20、common_rows、history_rows、selected_rows、direction_labels；probe只解码source_rows、labels、policy_sign及current/history5的balanced prediction/neighbors/distances/ties。禁止解码同包public_proprio、ordinary规则、features/logits/全动作预测或public9缓存。本轮不重新提取表示、不进行近邻搜索、不重算特征距离/全库距离、不重放环境或原CNN。

## 四格转移、分母与全部示教

主比较为旧主规则current_balanced与history5_balanced：负/正各5个近邻的平均平方距离，正类均值严格更小才预测正，完全相等固定负；规则、scaler、参考库和阈值均沿用旧冻结定义。

| 类别 | current | history5 | 解释 |
|---|---|---|---|
| stays_correct | 对 | 对 | 保持正确 |
| corrected | 错 | 对 | 被历史纠正 |
| new_error | 对 | 错 | 历史新增错误 |
| stays_wrong | 错 | 错 | 持续错误 |

以上四格在每个固定模型、每个目标正负方向和时间层中互斥且完备。重点单列正目标corrected和负目标new_error，同时保留正目标new_error与负目标corrected，不能把反向变化隐藏在净收益中。核对`history正确数−current正确数=corrected−new_error`，用两类各自分母重建BA变化；BA不是闭环成功率。

固定报告all_common、tail20、complement三层，主解释窗口为tail20。每模型及每条demo0..29均保留四格计数、占同方向总帧比例；纠正率另除以current错误帧数，新增错误率另除以current正确帧数，避免不同分母的百分比混比。空格保留0，零分母比例为null，不删示教/seed或把null置0。逐demo必须重新从本次ID和逐帧预测计算，不能把旧per_demo all_common聚合表当成逐demo尾窗。

逐帧输出全部21588个模型帧的task/seed/arm、source_row、demo、action_step、方向、目标第3分量及绝对值、两预测/四格标签、原Policy符号和是否正确。原Policy输出0单列为错误，仅作既定次级描述；它不是近邻控制器，也不替代current/history主比较。

每臂对同一3598帧对齐三个seed，保留全部三个转移码及每类0/1/2/3个seed的投票数、帧ID与分布；全3共同和seed分歧同时报告，不只挑多数一致子集。两臂分别汇总全部seed，不把相同标签或相邻帧当独立统计重复，不做显著性检验、seed选择或正式接受判断。

## 定位时间、目标方向历史与邻居变化

时间坐标仅为action step、距最后合法action的步数和相对末20起点位置。末20不是物理接近/放置/释放阶段，本缓存没有原生成功时刻、RGB、末端/物体几何或接触信息，不能据此判断正向动作是成功前必需动作。

为每个评分帧记录同demo的五个合法目标方向标签`[y_t,…,y_(t−4)]`，中性为0；这只是分析标签，不进入Policy/读出输入。material run定义为原始连续action steps上同一非零material符号的最大连续段，中性或相反符号均截断，不能跳过中性制造直接切换。记录自观察到的run起点的年龄、窗口是否含相反/中性标签；run从t=1开始时标左删失，不读取action0或声称知道更早真实切换时刻。年龄0..3与≥4对应五帧窗口是否完全处于观察到的当前run，左删失须单列。报告目标绝对幅值及上述条件的计数、均值/中位数/Q25/Q75（float64线性插值），不搜索新的幅值/距离阈值。

对current/history分别从已保存的负5/正5距离重建两类均值与`gap=负均值−正均值`，gap>0预测正，否则负；另存gap除以两均值之和，两者和为0时为null。gap是固定读出规则下的距离差，不是原action head输出、校准置信度或因果证据；只在同模型内对照，不跨模型比较绝对距离，也不假装五个lag的分项距离已保存。

全部21588个模型帧均核对两模式各10个已存邻居：ID合法且属于其它demo、t≥5、material、前5负/后5正、距离有限非负且类内按旧规则有序。每帧记录每类current/history邻居ID交集数0..5、各自参考demo数和参考尾窗数。对**全部2862个末20模型帧**导出57240行邻居明细，保留四格和两方向、参考demo/step/符号/末窗、距离及类内rank，不只选改变帧。邻居的时间位置是参考示教坐标，不能拼成同一物理轨迹。

比较正目标纠正与负目标新增错误的目标历史、时间分布、邻居ID替换及距离差，同时保留未纠正/未变错对照。若同样的历史模式既纠正又损害，只能记录取舍；若跨seed/demo共同集中，也只提供后续机制假设。当前六文件没有匹配原始feature，禁止声称本轮重证了“全局最近”或完成了全库距离复算；旧独立审计是此项来源证据。

## 预算、独立审计和停止规则

| 项目 | 未来主分析 | 未来独立审计 |
|---|---:|---:|
| 共享ID/标签NPZ加载 | 1 | 1 |
| 固定probe NPZ加载 | 6 | 6 |
| 数组文件加载总数 | 7 | 7 |
| 同帧模型记录 | 21588 | 21588核对 |
| 已存邻居记录 | 431760 | 431760核对 |
| 新训练/前向/环境/近邻搜索/模型或原始数据读取 | 0 | 0 |

两阶段共14次数值数组文件加载，**本轮冻结未消费**；每阶段每文件一次np.load，禁pickle/mmap/fromfile和逐成员重复解码。固定文件SHA字节校验与数值数组加载分开记账，不以旧205次剩余额度重新执行原采集。未来表行数固定：frame-transitions21588、model-transitions144、demo-transitions4320、seed-agreement-frames7196、seed-agreement-counts192、tail20-neighbors57240；统计/重用行数不是独立样本数。

未来仅CPU、Python3.10.22/NumPy1.22.4、CUDA隐藏及OMP/OPENBLAS/MKL单线程；工作内存≤512MiB，主分析/独立审计各≤120秒，外部总watchdog300秒。运行记录配置/代码/输入/输出SHA、设备与软件、PID、开始结束时间、实际读取attempted/completed、耗时及异常，不用真实试跑决定预算；环境seed/初态不适用。

主分析与独立审计分进程，独立审计不导入主分析统计实现，使用相同七个固定数组重新计算标签/ID/历史、四格、父汇总混淆计数、全部保存邻居/距离规则、逐demo/seed一致性与所有导出表。整数严格相等，浮点绝对容差1e−10、相对0；不能因审计追加feature或全库距离预算。两遍输入字节SHA前后不变，所有允许字段/预算attempted=completed且无越权，才标完成。

任一来源/角色、哈希、ID、dtype/shape、非有限值、邻居、父计数或预算/时间不一致立即停止，保留partial及失败，不重试、不换seed、不覆盖旧缓存、不修父结果。输出只能新建于`/tmp/libero-task7-component3-history-transition-20261108-v1.json`及对应artifacts目录；无论阳性/阴性，均不自动启动训练或解封正式角色。

## 本轮冻结状态与下一步

本轮只创建配置、上述口径、stdlib元数据验证器与合成测试，并核对指定父JSON/源码与七文件字节摘要；真实研究数组解码、前向、训练、环境、初态/封存payload均0。实际测试数、命令、配置/代码/文档摘要与仓库保护核对记入`/tmp/libero-task7-component3-history-transition-freeze-20261108-v1.json`，不在文档里预填未运行的分析结果。

执行入口、七文件/成员访问guard和独立数值审计**尚未实现或绑定**。元数据验证器的`--require-executable`应拒绝，普通预检不解码数组；未来须单独绑定指定analysis、independent_audit、payload_guard、runtime_tests、metadata_protocol、protocol_tests、protocol七角色、当前科学配置SHA与完整预算后，才能消费14次额度。

下一步实现上述执行入口/guard/独立审计，用合成数据验证四格守恒、净变化不可替代总纠正/新增数、中性断run/左删失、等距/零gap、空组、逐demo/seed对齐和越权/重复读取拒绝，再完成代码绑定并执行。真实结果应另写dated report，保留全部正反证后再决定单变量训练候选；本轮不改500/5步/现有loss。

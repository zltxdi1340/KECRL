# task7残差输出的错号转移与幅值只读协议（2026-11-08）

**本轮仅冻结方案，尚未投影真实CSV数据或执行新分解。** 承接[固定当前表示的可纠正性对照](libero-task7-current-residual-correctability-results-2026-11-08.md)：仿射共同正方向MSE/错号数5/6改善，同时共同负错号5/6增加；bias与current额外三seed联合cue均false。本次解释净错号变化中“纠正了哪些、又新增哪些”，检查其与参考输出距0的幅值、残差位移及同帧连续误差的关系；不能用后验解释放宽原护栏或恢复候选。

保持task7、seed20261106/07/08、A_original/B_fomaml全部六个原Policy500端点与已停止的辅助500端点、未来稳定化5步和原loss。原四个最终cue=false、G失败/H未开展、source-query及正式评价payload封存均不改变；无训练/新读出/优化器/环境/近邻搜索/原数组读取，不更新SPT/Knowledge或注册Module。这是给定已暴露support表示的OOF输出诊断，不是未见泛化、正式结果或闭环因果实验。

## 唯二CSV与投影权限

[科学配置](../configs/libero-task7-residual-sign-transition-readonly_v1.json)及[元数据门控](../experiments/libero_task7_residual_sign_transition_protocol.py)只授权上一轮artifacts的两张CSV，固定父science/binding/result/audit/completion五份JSON SHA用于身份和完整审计验证，不用JSON里的指标作为新增分析数据。**不重新打开或哈希NPZ/NPY、头归档、checkpoint、HDF5、RGB、features、公开proprio或private simulator state。** 不读其它CSV、旧NN/历史标签，不读当前之外的实验输出。

| 输入 | 每阶段词法行 | 允许投影 |
|---|---:|---|
| oof-frames.csv | 75384 | 17列，全部行 |
| curves.csv | 4608 | 全部行仅9个ID列；update=500且role=probe的576行额外14指标 |

frames的17列为task、seed、arm、mode、update、source_row、probe_fold、demo、action_step、target_direction、tail20、component3_category、raw_residual、clip、target_2、prediction_2、error_2。CSV原35列中其它动作轴的target/prediction/error不转换或暴露给分析；第3分量是action index2，不解释为世界z或夹爪close/open。

curves的9个ID列为task、seed、arm、fold、mode、update、role、phase、direction。识别全部4608个预期ID、重复/缺失及合法枚举，但不转换0/100/250点或fit行的指标。仅final500 probe的14指标为n、material_n、component3_mse、component3_sse、component3_mae、component3_bias、material_sign_error_n、material_sign_error_rate、clip_n、clip_fraction、raw_residual_mean、raw_residual_abs_mean、raw_residual_min、raw_residual_max。不得利用较早曲线选择checkpoint或补读fit预测。

两表header须与已审计manifest完全一致且无重名，行宽/字段类型/身份必须严格合法；源字节SHA在前后核对，与词法遍历/投影分开记账。冻结阶段只核对这两表的字节，不打开CSV reader或投影真实行。

## 配对范围、ID及标量复现

source meta_support demos0..29，obs[t−1]→action[t]、t≥1每模型4188行，三个读出各一份final500 OOF预测，共75384行。编码器已见全部30条support；probe仅为辅助读出内部整demo三折，probe_fold=demo//10，每帧仅作为一个头的probe。由固定30条示教长度重建source_row、demo、step、probe_fold、tail20，不能仅按CSV行序或相近标签拼接。

逐(task,seed,arm,mode,source_row)严格唯一且完备；同帧三个mode、全部seed的目标/ID须相同。target和保存预测须为有限、范围[−1,1]、可精确表示的float32值；error_2复现prediction_2−target_2，六种类别、目标方向标签及clip一致。original残差0、预测为原输出、clip=false；另用CSV原输出和raw_residual独立复现float32(clip(original+raw_residual,−1,1))，无需W/特征前向。每端点/fold的bias_only raw_residual须为同一float32常数。

原始t≥1及neutral帧保留供完整性和probe摘要复现；错号转移主集合固定t≥5且abs(target_2)≥0.025，共3598共享帧（负2273/正1325），末20负402/正75。时间层固定all_common、tail20、complement，共同neutral470帧和首次4步不进入二元错号分母。末20仅为每demo最后20动作，不解释为物理接近/运输/释放。

三组候选−参照全部保留：bias_only−original、current_affine−original、current_affine−bias_only。后者是两个分别学得的读出之差，不是对同一仿射头去掉Wz的干预。每模型×比较3598帧，共64764配对行。

## 四格、零、幅值与连续误差

本次transition是**已保存连续动作相对专家目标的符号对错变化**，不是之前NN读出的transition，更不是episode成功转移。material目标下，target_sign×prediction>0才算正确；预测0也算错，严格反号/恰0分开。四格为：

| 参照 | 候选 | 名称 |
|---|---|---|
| 对 | 对 | stays_correct |
| 错 | 对 | corrected |
| 对 | 错 | new_error |
| 错 | 错 | stays_wrong |

每目标方向/时间层四格互斥完备，候选错误数−参照错误数=new_error数−corrected数。分别报告参照/候选符号−/0/+的九格、零输出计数，以及原六种严格错号/零/同号不足/过大/精确相等/零目标非零类别的6×6表；material下最后一种应为空，仍保留。净变化少不代表没有大量相反转移。

参考输出和候选输出距0分别固定为zero、0<abs(p)<0.025、abs(p)≥0.025三档，保留3×3输出幅值档及所有空格；阈值沿用已有对称0.025，不扫描阈值或用它替代严格符号判定。正纠正、负新增为关注点，同时完整保留正新增、负纠正及两个不变格。

每帧计算：

- signed_margin_ref=s×p_ref、signed_margin_cand=s×p_cand、signed_effective_shift=s×(p_cand−p_ref)。新增错号需ref margin>0、candidate margin≤0；纠正需ref margin≤0、candidate margin>0，跨到恰0仍属错误。
- raw_shift=raw_residual_cand−raw_residual_ref；effective_shift=p_cand−p_ref；gap=effective_shift−raw_shift。**gap同时含clip和保存float32的舍入差**，非零gap不能直接归为裁剪。分别记录两头原clip标记、none/ref_only/candidate_only/both计数与gap分布。
- abs(effective_shift)/abs(p_ref)、abs(raw_shift)/abs(p_ref)。参照恰0时比值null并单列分母0；不加epsilon。比值≥1本身不表示损害，仍须结合signed shift、参考正确性和候选符号。
- 同帧参考/候选的SSE、MSE、MAE、bias、目标能量及相对零输出误差，逐帧ΔSSE的lower/tie/higher计数和能量贡献。新增错号可能伴随MSE下降，例如小正目标从过大正输出移到轻微负输出；错号变化不等于连续误差方向。

每组并列target_abs、ref/candidate_abs、两头raw残差、raw/effective shift、gap、signed margins和比值的n/均值/中位数/Q25/Q75/min/max（float64线性分位数）。不能只凭错号数断定残差普遍过大或当前表示缺少信息；目标尺度和原本的幅值误差同时保留。所有组的SSE和ΔSSE须与整类守恒。

## 固定参照、示教构成与跨seed

固定两种组间描述：corrected−stays_wrong（原本都错）、new_error−stays_correct（原本都对），同目标方向/时间层/端点/比较；并列组n、目标幅值和误差尺度，不作因果或同分布假设。全部30个demo保留，仅同demo两组均非空才定义配对差；共同demo等权均值、配对数、缺项原因和higher/tie/lower/null计数单列，不能删空seed或改用更好看权重。

同一来源arm/比较的3598帧按三个seed对齐，保留各seed四格、参考/候选错误、原幅值档及ΔSSE方向；统计参考错误、候选错误及四格成员的0..3票数、至少一次/三seed全共同。**共同新增错号少不代表没有共同参考错误**，两者分开报告。A/B来源不合成六个独立seed，不把邻帧或同帧模型重复当独立样本，不做p值或显著性。

另按同一端点/源帧，交叉original→bias和original→affine的四格，完整4×4矩阵检查纠正/新增错误是否重合，保留全部16格。该交集只是两种输出的关联；CSV不含W/b/features，不能拆分current_affine内部Wz与它自己的bias，更不能把affine−bias_only当作纯Wz因果贡献。

## probe摘要复现与严格预算

从对应端点/fold/mode的OOF帧独立重算全部576个final500 probe行的14指标，含all_source/all_common/tail20/complement与all/negative/positive/neutral分组，共8064个标量/阶段。neutral的material分母为0、率null；空组n/SSE0、MSE和分位数null，0目标能量比null。原始行亦用于all_source及每折覆盖，不进入主四格。fit预测不在本CSV中，不能为了复现fit摘要回读头NPZ。

| 每阶段读取/投影 | 精确值 |
|---|---:|
| 固定CSV数据遍历 | 2 |
| frames / curves词法行 | 75384 / 4608 |
| frames投影列值 | 1281528（17列） |
| curves ID列值 | 41472（9列×全部行） |
| 选中probe指标列值 | 8064（14列×576） |
| 合计词法行 / 投影列值 | 79992 / 1331064 |
| material配对 / 跨seed帧 | 64764 / 21588 |
| 原数组/归档读取或字节哈希 | 0 |
| 训练/前向/环境/近邻 | 0 |

主分析＋独立审计总4次CSV数据遍历、159984词法行、2662128投影列值；不借用已消费的50次NPZ额度。12张输出CSV固定：paired-frames64764、model-transitions432、demo-transitions12960、margin-cells3888、error-cells3888、contrasts216、demo-contrasts6480、demo-standardized216、crossseed-frames21588、crossseed-summary144、head-overlap576、probe-parity576，共115728行；另有72行完整三seed对比JSON。组/phase重复统计不是新独立样本。

CPU Python3.10.22/NumPy1.22.4、CUDA隐藏、单线程、PYTHONHASHSEED0；主/审计各≤180秒，总watchdog420秒，当前地址空间VmHWM/VmRSS≤512MiB，输出CSV≤256MiB。记录配置/代码/父证据/两表SHA、seed、设备/软件、实际attempted/completed遍数/行/投影列值、时间/PID/成本、异常；无环境seed或初态索引。

未来字段/CSV guard必须限制一文件一遍、上述条件投影，不允许直接raw/其它CSV绕读或np.load/fromfile/mmap、Torch/HDF5/环境入口。源码、七角色和预算先绑定；独立审计另进程，不导入主数值实现，重复同两表/投影并独立核对ID/标量/14项probe摘要、四格/九格/类别/幅值/SSE守恒、示教构成、重合与三seed及全部输出。浮点abs1e−10/rel0，ID/计数/类别/float32预测复现严格；科学符号、零和幅值阈值无epsilon。

任一SHA/父审计/ID/header/重复缺失/字段越权/标量复现/probe parity/预算或独立审计不一致，停止并保留partial，技术重试0，不回写历史结果或修旧CSV。主pending-audit字节不回写，audit/completion另存。没有新接受门槛；所有父cue仍false，分析只汇报支持证据、反证和未解决问题，不自动调参、训新头或启动闭环。

## 本轮状态与下一步

本轮只读取源码/当前报告、五份指定JSON元数据及两CSV字节SHA，创建科学配置、本方案、stdlib门控与合成测试。新增104项合成测试、相关协议回归共277项通过，py_compile及环境检查通过；CUDA为本轮CPU验证主动隐藏。真实CSV数值投影行、原数组读取/哈希、训练/前向/环境均0；没有新错号分解结果。实际验证和工作区保护证据另存/tmp/libero-task7-residual-sign-transition-freeze-20261108-v1.json，并同步README/roadmap/500计划。没有修改训练或共享运行框架，本轮只运行相关协议测试，未重跑全量pytest。

下一步实现入口、两CSV条件投影/字段guard及独立审计，合成测试覆盖四格守恒、零分母、near-zero边界、MSE/错号反向、clip与舍入、ID联接、probe摘要与跨seed后绑定代码，再按本额度执行。本轮不创建执行绑定或启动分析/训练，正式评价数据继续封存。

# task7固定当前表示残差读出执行契约（2026-11-08）

本契约落实[科学冻结方案](libero-task7-current-residual-correctability-plan-2026-11-08.md)，不改变任务、源端点、整demo三折、500更新、loss、对照、判据及预算。科学配置SHA保持`722b626c34f7faff67356b3d7e385784536aad4b95abf80084f685079855c58e`。原方案、配置、元数据门控和83项元数据测试保持原字节。独立执行绑定另存`configs/libero-task7-current-residual-correctability-execution-binding_v1.json`，绑定科学配置、七角色、源码/关键本地Torch实现依赖及本契约后才可读取真实数组或辅助训练。

执行入口`python -m experiments.run_libero_task7_current_residual_correctability`默认仅预检，`--mode run`依次启动全新CPU训练和NumPy独立审计进程。CUDA隐藏、固定软件/线程/确定性；阶段活动各180秒、外层各195秒上限且总watchdog420秒。每阶段当前地址空间VmHWM/VmRSS≤512MiB；CSV≤256MiB，新头NPZ≤128MiB，技术重试0。已有launch/结果/artifacts或partial即拒绝，失败保留日志/状态并不启动下一阶段。中断仅清理本次启动的进程组。

## 数值实现与原Policy边界

只有36个新辅助头可以创建Adam、接收梯度或更新参数。bias_only只有一个bias；current_affine只有256权重和一个bias。角色guard登记参数身份、零初态、CPU/float32、有效参数量和每个head的500个batch→forward→backward→step；探测到未登记参数、probe行、重复step、重复optimizer或额外更新即失败。torch.load、HDF5、原模型/环境模块入口禁止，原Policy只以缓存预测和checkpoint JSON身份出现，无新原Policy前向/训练。

fit阶段仅将授权fit源行交给训练worker；均值/std只用fit，float64原地中心化/缩放后转float32。三seed×三折只有9张PCG64采样表，两个来源臂及两个读出共用绝对source_row表。保存全部初始/100/250/500权重、500步loss/裁剪前后norm及Adam m/v/步数；全部36头精确到500前，不允许probe变换或快照读出。

本机CUDA版Torch仅导入约435MiB，默认Adam的Dynamo编译器包装会额外导入大量模块。为遵守原512MiB预算，在guard作用域内调用本地固定2.4.1 Adam的原始eager add_param_group/zero_grad/step函数，step置于no_grad；保留原Adam类、状态、偏差修正、单tensor数值内核和全部optimizer设置，仅绕过编译器graph-break包装。退出作用域恢复原函数，不修改安装文件。独立合成单测以相同500组梯度比较标准Adam与eager路径，权重/bias及全部m/v逐位相同；实际全部36头仍使用真实180秒/512MiB上限。Torch optimizer/Adam/包装与clip实现文件随执行绑定固定SHA。

快照评分在fit门打开后，用保存的float32权重与标准化特征按float64数学函数计算残差，clip后只将第3分量保存为float32，其余六分量逐位复制p0。主阶段采用流式einsum避免整份float64矩阵临时副本，审计另行用逐行乘积求和重建。共144个主读出和144个审计重建，不额外调用Torch/CNN/原Policy来打分；不宣称Torch前向位级重放。原float32预测/目标先转float64算指标，审计对保存预测算严格符号和指标，再用重建做1e−5数值/符号稳定性门。

归档没有新增权限字段：每头19字段按科学配置保存，bias_only的weight/m/v仅零占位。完整诊断表流式写出，临时对照缓存只保留判据/差值需要的14指标与分母；已处理的源表示及时释放。阶段之间回收临时对象、将未使用CPU heap页面归还系统；不重置或隐藏VmHWM，仍检查真实历史峰值。

## 读取、角色、独立审计与最终cue

主阶段只有7源NPZ/17字段；审计7源＋36新头NPZ共43次/701字段。合计50次/718字段；源额度14次/34字段、新头审计36次/684字段分开记录。np.load必须为path+allow_pickle=False，每文件/成员一次；非授权字段、重复、mmap、直接fromfile/read_array/Zip/raw读取及旧NN标签CSV均拒绝。输入及新产物SHA在阶段前后核对，字节SHA不混计数值解码。该guard是研究访问契约，不是OS安全沙箱。

独立审计不导入主训练数值实现或Torch；重新构造完整demo角色/scaler/精确批次，核对实际optimizer字段/步数/状态与角色日志，独立重建144批读出、全部OOF帧、七CSV共151434行、三seed384行JSON及legacy216标量。它不独立重跑18000步optimizer，不将有限loss/norm核对称为训练数学重演。其它六维位级不变、初态数值等同原Policy、source_row完整/唯一均单独核验。

最终数值符号门依赖独立审计的144批重建。主pending-audit JSON因此保存18组provisional flags，numerical_sign_stable=False且显式标记pending，4组provisional cue不得解释为科学失败；不在主阶段额外做一次重建。审计独立核对这些暂定值，再根据final500共同material帧的实际重建分歧数生成最终18组比较/4组arm cue，写入audit/completion；主JSON不回写。读取结果须跟随completion→audit获取最终cue。

科学比较仍是保存float32预测的直接<0/≤0，无epsilon退化容忍。重建误差≤1e−5但final500 material符号有分歧时，相应cue=false，不改符号或放宽阈值。任何guard/来源/角色/optimizer/预算/表/数值审计错误为技术停止；cue=false是该候选的诊断结束，不追加预算或选择其它曲线点。cue=true也不部署读出、启动闭环、恢复G/H或修改SPT/Knowledge/Module。

## 绑定前验证与后续记录

绑定前保留三次合成内存失败记录及修正，均未解码真实研究缓存或训练真实辅助头：首次为默认Dynamo导入，之后为大矩阵临时数组/CPU heap和完整指标字典缓存。先后保持25项访问/数值单测通过而完整测试触发内存护栏；修正后27项测试（含完整36×500、全新审计及500步标准Adam位级对照）通过，完整合成主/审计约24.4/33.8秒、峰值501.60/111.61MiB、13769914项独立核对。日志与`/tmp/libero-task7-current-residual-correctability-prebinding-memory-failure-20261108-v{1,2,3}.json`保留；后续新增拒绝测试和最终CPU回归结果写入执行绑定/结果报告，不回写科学冻结。

最终实际配置、种子、文件/代码SHA、逐步/曲线、18k预算、时间/RSS、失败/重试状态及结论另写dated results报告，并同步README/roadmap/500计划。保持原Policy500、未来稳定化5步及现有loss，source-query和正式评价payload封存，G停止/H未运行不变。

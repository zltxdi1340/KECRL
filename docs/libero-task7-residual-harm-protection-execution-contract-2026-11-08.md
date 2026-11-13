# task7原动作误差保护配对读出执行契约（2026-11-08）

本契约落实[已冻结科学方案](libero-task7-residual-harm-protection-plan-2026-11-08.md)，科学配置SHA保持`aabfd4a198a0e59e96394ca788dc1bf0ab9c4612afb8e7d57eac2880531955b5`。原方案、配置、元数据门控及172项元数据测试不改字节。新增入口、角色/字段guard、NumPy独立审计、运行测试和本契约，另存执行绑定后才开放真实来源字节SHA、数值读取和辅助训练。

## 入口、单变量和fit/probe边界

`python -m experiments.run_libero_task7_residual_harm_protection`默认只预检；`--mode run`依次启动全新CPU训练、NumPy独立审计进程。仅新建36个257参数辅助头，每头500 updates；同源同折的λ=0/1均256权重＋bias、全零初始化、同scaler/绝对source_row采样/Adam/裁剪/设备。唯一变量为既定新增误差惩罚系数0或1。主MSE仍在64×7全部元素上平均，P=mean(ReLU(e−e0))/7；e0脱离训练图，ReLU零点梯度为0。λ=0直接返回原L0张量，惩罚遥测detach/no_grad计算，不添加0×penalty，也没有遥测额外前向。

角色guard只允许登记的CPU float32、requires_grad、零初值且形状256＋1的参数创建Adam、执行fit batch→forward→backward→step各500次；禁止原模型/未登记参数、probe梯度、重复/额外optimizer或step。只有全部36头到500且fit检查通过后才能打开probe、完整标准化和144批固定快照。fit仅收到20示教的行，包含neutral，原good/方向/phase不参与抽样或训练权重。编码器已见全部30条support，缓存物理解码也包括probe，不能称未见编码器泛化。

沿用固定Torch2.4.1 eager Adam路径和实际超参数，绑定本地Adam/Optimizer/_compile/clip源码SHA。fit标准化以复用float64方差缓冲计算相同population std，再重填缓冲作变换；合成测试要求与历史mean/std/标准化float32逐位相同。Torch共享只读使用的fit输入内存，免去额外完整副本；不修改输入或数值精度。每对首个Adam更新的保存W/b及初始loss逐位核对，不增加更新或前向。

## 有界归档、读取和预算

为满足512MiB上限，每头fit完成后一次打开新NPZ，顺序写入26个fit/角色/逐步字段，并只保留快照W/b等小数组；归档流保持打开。全部fit完成、probe门打开后，追加唯一`predictions.npy`成员并关闭，共27字段，每字段只写一次。主阶段没有重新打开或读取这些新归档。失败则关闭并保留不完整归档；禁止覆盖、删除、续训或重试。此为新产物的分阶段写入，不改历史头，也不新增输入解码。

主阶段解码7源NPZ/17字段；先后字节SHA核对7源与18历史仿射头的可用性和不变性，主阶段对历史头仅允许SHA作用域，禁止np.load/成员/raw读取。审计解码同7源＋36个新头×27字段＋18个历史头×14字段，共61NPZ/1241字段；两阶段合计68NPZ/1258字段。历史14字段固定，不可访问predictions/角色字段/旧bias-only头。0输入CSV遍历；原Policy checkpoint仅保留JSON身份，不读取或哈希其字节。

guard核对每文件/成员一次及allow_pickle=False；拒绝未授权文件/字段、重复、mmap/fromfile/read_array/Zip/raw旁路、checkpoint、HDF5、原环境/Policy包和输入CSV。源与历史文件不可写；所有输出写新路径。它是研究访问契约，不是OS安全沙箱。

固定18000批fit及1152000行，144主＋144审计快照；审计另重建18000批/1152000行保存的更新前线性输出和训练loss。14CSV共191994行、384份seed摘要、6个候选flags、2个来源cue与18个历史头复现记录，数量按科学配置核对。所有30示教/空组和三seed保留，500唯一主端点，不按曲线选点。

CPU固定Python3.10.22/NumPy1.22.4/Torch2.4.1+cu124、CUDA隐藏、单线程/确定性；阶段活动各≤180秒，子进程各≤195秒且总watchdog≤420秒；真实VmHWM/VmRSS≤512MiB，不重置峰值。CSV≤256MiB，新头NPZ≤128MiB。运行前检查GPU占用并记录CPU实际设备；UTC/PID/软件/配置/代码/文件SHA、角色、attempted/completed更新/行/解码、时间/RSS均保留，无环境seed或初态索引。

## 独立重建、历史控制和最终判据

独立审计不导入新主训练数值实现或Torch，不重新训练。分别重建整demo三折、fit-only scaler、9张精确采样表、逐步loss/active_harm计数、144批快照、OOF/七轴及全部14表/seed摘要；最终500严格符号和原good损害判定的重建分歧进入数值门。训练线性重建abs≤1e−5，loss abs≤1e−6，指标abs≤1e−10/rel0；严格符号/损害判定无科学epsilon。保存float32训练预测用于独立重算active_harm，线性数值重建造成的临界差异另记，不冒称完整optimizer或梯度轨迹重演。

新鲜λ=0的18个头，updates/全部4快照W/b/500批次/scaler/500 loss/梯度norm/step/最终Adam m/v等14字段须与历史归档dtype、shape、位级全部相同。任一不符即技术停止，不能当作λ=1效应。候选其它六维仍逐位复制p0；首步配对相同另行核对。

原good与wrong从固定原p0和专家动作定义；按全部common material逐帧保存一次成员，主/审计分别用数组和标量实现。负good common/末20的可检验损害须降低，所有原good组不新增被损害帧；正方向保留同一批已纠正帧，不能以新增纠正抵消丢失；所有MSE/错号/六维/数值护栏同时满足。两来源分别要求三seed全部通过，不按均值挽救。主JSON保存数值/历史门pending的暂定flags，不能解释为最终科学失败；最终6模型flags/2来源cue只由audit/completion给出，主JSON不回写。

来源/角色/字段/optimizer/预算/复现/审计失败为技术停止，真实重试0；科学cue=false按500终点结束候选，不调λ、good定义、预算或checkpoint。即使cue=true，也不自动部署、闭环、恢复G/H、更新原Policy/SPT/Knowledge或注册Module。原Policy500、未来稳定化5步/现有loss、父4 cue=false和source-query/validation/held-out/qualification/reserved/final封存保持。

## 绑定前验证记录

已保留两次合成内存失败（516.61/513.49MiB）及一次新增JSON摘要字段范围不一致，均发生在真实输入访问前。修正为复用方差缓冲/fit Tensor及单流分阶段归档，未放宽上限、删字段或改科学设计；摘要仅对齐已选指标字段，原CSV和判据不变。失败日志与`/tmp/libero-task7-residual-harm-protection-prebinding-{memory,schema}-failure-20261108-v*.json`保留。最终合成/重点/CPU回归、成本和SHA写入执行绑定；真实执行的结果与结论另写dated报告。

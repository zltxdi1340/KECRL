# 第3分量方向可辨性：执行绑定契约（2026-11-07）

[科学协议](libero-source-expert-component3-observability-protocol-2026-11-07.md)和配置SHA `acac57118851969ece67cd2fe606927678e34a50647db78343aaf7280f513550`保持。用户已授权实现、绑定后运行。本契约明确实现与审计细节，不更改数据角色、30个固定500端点、历史窗口、读出规则、408批预算、统计或停止条件。所有结果仍是source meta_support diagnostic。

## 入口与阶段

- [执行入口](../experiments/run_libero_source_expert_component3_observability.py)默认只做metadata预检；`--mode run`顺序启动capture、CPU probe、独立audit三个子进程，记录PID、命令、环境、wall time、exit code、watchdog与日志SHA。任一阶段失败即停止，无重试。
- [方向算子](../experiments/libero_component3_direction_neighbors.py)实现t≥5共同行、同demo合法历史、整demo scaler/参考隔离、每类5近邻与普通5NN、两类召回/BA、逐demo分组及基于整数混淆计数的联合cue。距离复用旧已冻结direct squared-distance primitive，不改原夹爪/phase算子。
- [访问guard](../experiments/libero_source_expert_component3_observability_access.py)继承既有精确causal HDF5窗口限制，新加阶段数组allowlist；[独立审计](../experiments/audit_libero_source_expert_component3_observability.py)不用生产方向算子重算标签/历史、scaler、全部选中邻居距离/投票、每fold首query全库搜索与全部指标/计数。
- 绑定文件必须含科学协议要求的八个角色及其指定路径/SHA；另固定全部src/experiments/scripts Python依赖、环境、采集ledger和本契约。未绑定或代码/上下文hash变化，入口在payload前拒绝。

运行命令：`conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_source_expert_component3_observability --mode run`。启动环境和外部8100秒watchdog另记绑定/launch；不用该命令绕过绑定或资源检查。

## 采集与parity

固定GPU1 UUID、任何其它compute进程即等待、≤512MiB/≤5%利用率等门控不变。等待最多3600秒，不在等待时构造模型或读数值payload。采集计时600秒自实际payload阶段开始；先核验并加载全部30份weights-only checkpoint，再一次读取五份旧标签，确定全部material query/reference数、历史和读出预算，并在首个CNN前写`label-plan-before-first-CNN.json`。

每task原source窗口只读取一次，仍为5个唯一HDF5、10次显式index/payload File打开、1933窗口、每字段18648行。精确切片只允许meta_support demos0..29的四个原观察字段和t≥1动作；action0、attributes/init_state、states、query或其它任务字段拒绝。旧数组参考仅五label和30prediction，共35次np.load。

所有378主批次须匹配旧批次输入摘要；六模型输入配对及前后摘要不变。非修改hook在每次完整前向捕获256维pre-head与7维logits；GPU上`torch.tanh(logits)`与原输出作字节一致检查。原首批repeat同样比较含padding的feature/logit/prediction字节，非仅float数值相等；全111888预测行与旧预测及标签ID也作字节一致检查。G500六项汇总继续abs1e−10/rel0，原权重前后hash与gradNone检查不变。采集随机种子20261022沿用原读出，与checkpoint训练seed区分记录；无真实warm-up、额外CUDA pool探针、梯度或优化器。

输出为30个模型NPZ（features/logits/predictions）、30个完整repeat NPZ、5个共享NPZ（原labels/IDs、公开joint7+gripper2、共同/历史/选中行及方向标签），共65。没有raw RGB、物体/末端/接触或private state。共享公开9维取同一批pre-action输入，不另取环境状态。

## CPU读出与独立审计

采集子进程退出释放GPU后，CPU子进程隐藏CUDA，仅加载上述30模型+5共享文件，共35次np.load。当前/history与平衡/普通规则共享五lag距离，不新增Policy前向。输出30主probe和5公开9维probe，共35个NPZ；fold scaler/分母和全部模型/示教指标另存JSON。空间/模型不混合近邻库，公开9维每task只读出一次。

每阶段`np.load`只允许所列路径、pickle=False、无mmap，逐文件校验SHA并拒绝重复或额外加载。输出目录及数组用独占新建，failed-partial或已有probe也阻止重跑。新输出固定路径的**字节SHA校验**单列允许，但不能由此np.load输出；字节hash与数值数组解码不同，后者严格保持35+35+135=205次。直接np.fromfile、未知array、checkpoint/HDF5/初态文件与禁止runtime导入均拒绝。

独立审计只加载100个新NPZ及35旧参考，共135次。它独立重建causal IDs/历史和方向标签，核对全七维旧预测、完整首批重复、原生采集角色/窗口/weight证据及完整预算；重新拟合每fold scaler，重算全部选中邻居距离和投票，并对每fold固定首个material query在全部参考库重做五lag搜索（最多1050fold）。距离abs1e−8、指标abs1e−10；两模式/两规则同分母，空/单类BA为null，不丢fold。

GPU tanh的字节检查由采集408次逐一完成；CPU独立审计核对该ledger、重复logits/actions和旧动作parity，**不在CPU冒充逐位重放CUDA tanh或重新计算CNN**。pre-head原始表示由已绑定hook和权重/输入parity定位；本审计不进行额外checkpoint读取来重建编码器。此限制随最终结果记录。

CPU读出2400秒、独立审计1200秒，观察缓存2GiB与CPU working 1GiB限制保持；所有精确计数attempted/completed相等，上界项另报实际值。监督进程只终止自己的超时子进程组，保留其退出状态和partial，不影响其它GPU作业。

## 软件验证及保留

[专项测试](../tests/test_libero_source_expert_component3_observability.py)含因果ID/近零边界、全demo隔离及中性scaler、正负每类邻居/等距规则、单类/空组、整数cue、非修改hook、字节parity、阶段array/HDF5 guard、篡改拒绝及完整30模型/35空间的合成artifact链路。全部合成数据，不消耗真实研究预算。

加入完整链路后，`execution-focused-20261107-v2.log`发现一次guard错漏：新输出的SHA字节读取被误拒。已单独授权固定输出hash-only路径，仍禁止额外np.load，并补齐failed-partial拒绝；科学配置/预算未变。v2失败日志与前后通过日志均保留在/tmp，真实研究payload尚未因此读取或重跑。最终专项、CPU回归、编译/环境验证和所有代码hash记入执行绑定及结果，不把历史回归当本轮验证。

结果JSON保持pending-audit原字节；通过后另写independent-audit/completion，不回写旧结果或checkpoint。研究artifact位于`/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/`及同前缀结果。完成后另写dated result report并更新README/roadmap/500计划；不改本科学协议、G停止或角色封存，不自动改变训练或接受SPT/Knowledge/Module。

# 固定500端点的source专家末段平移动作读出（2026-11-07）

用户要求继续[408批预算草案](libero-source-expert-prediction-cache-availability-2026-11-07.md)的下一步。本次冻结、实现并在设备空闲后执行有界专家读出，不是重训、正式评价或新的方法接受。依据[原生目标核对](libero-source-candidate-native-goal-pose-results-2026-11-07.md)，要检查当前端点在相同合法专家观察上的末段平移拟合，不能把最终物体XY差直接归为夹爪或某个head根因。

## 固定模型、数据与比较

[科学配置](../configs/libero-source-expert-late-translation-readout_v1.json)SHA为`0349b99cae5b9e79019de54f7e8422b5b850ae52309544f870aa5c51443453e6`。固定task0/1/3/4/7、seed20261106/07/08、A_original/B_fomaml全部30个现有500端点及文件/张量SHA，不能选checkpoint或补训。保持CNN-only、float32、原dual-RGB/joint/gripper四输入、无增强、原均匀七维masked action MSE，0梯度/优化更新。离线读出不运行环境或额外稳定化；未来闭环的5步设置不变。

只读取source meta_support demos0..29；输入`obs[t−1]`对应`action[t]`，t≥1，sequence10/stride10、batch32、固定顺序、无shuffle。全部合法帧都输出预测，不只保留末段。末段为每demo最后min(20,L−1)个合法动作，其余为补集；这是示教时间分层，不是物理接触/运输/放置标签。数据角色为训练拟合诊断，不能称未见泛化；G的source meta_query已参与外循环反馈，本次不读query。validation、held-out、qualification、reserved、final payload与所有初态/private simulator数据访问预算0。

主描述量为每个task×seed×arm的末20帧、补集及全support平移action MSE，另保留七维/旋转/夹爪误差、三维平移动作分量MSE和有符号bias。动作误差不等于世界位置厘米，不能用原生3cm阈值给它分类。A/B按同专家帧配对，保留全部15个task×seed组和30个模型，不合并为独立样本，不做显著性、距离/phase阈值搜索、loss或方法选择。阴性与反向结果同样输出。

## 精确预算

| task | 合法专家行 | 窗口 | 每模型主批次 | 六模型主批次 |
|---|---:|---:|---:|---:|
| 0 | 2895 | 304 | 10 | 60 |
| 1 | 4059 | 422 | 14 | 84 |
| 3 | 3064 | 317 | 10 | 60 |
| 4 | 4442 | 458 | 15 | 90 |
| 7 | 4188 | 432 | 14 | 84 |

全部主批次378，每模型首个原批次重复一次，30重复，合计408个Policy前向。主有效预测111888行/115980计算槽，重复9210行/9600槽，包含padding；每模态共享合法观察18648行，目标动作18648行，唯一末段帧3000。checkpoint读取30次、Policy对象30个，无额外真实smoke、warm-up或合成CUDA pool前向/backward。沿用已验证的deterministic pool实现，不在本轮重做CUDA梯度探针。

5个source HDF5各先index、再payload，原数据集契约共10次显式File打开；不因5个唯一文件而漏计重开。1933个合法window只读一次，五个字段各18648行；action0不用于标签或评分。SHA校验整文件字节与指定角色数值解码分开记录。每task缓存≤2GiB，不同时缓存全部任务。运行活动上限600秒，含权重核验、文件校验、数据解码和读出；无预算追加或技术补跑。

设备固定物理GPU1、UUID `GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff`、逻辑cuda:0。CUDA_VISIBLE_DEVICES=1、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP/OPENBLAS/MKL单线程，deterministic开启、TF32关闭。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、h5py3.11.0；与G运行一致。只在任何数值payload前检查GPU，显存>512MiB、利用率>5%或存在其他compute进程均等待；212MiB上下文也不能按空闲处理。不换设备或回退CPU。

资源等待最多3600秒、每30秒重检，不分配Policy/数据/GPU上下文，不属于研究重试；释放后先重新核对全部绑定，再启动唯一读出。等待用时与600秒活动分开记录，外部OS watchdog4320秒。超出等待或技术门槛即停止，不能抢占、杀其他作业或放宽门控。

## 复现门槛与实现隔离

[纯metadata协议](../experiments/libero_source_expert_late_translation_protocol.py)对G training JSON只做结构扫描，解码指定30个source模型与effective Policy配置；其余290个模型记录及validation曲线内容不数值解码。新配置中固定这30份G500全support参考量，不用它们选择模型。

[执行入口](../experiments/run_libero_source_expert_late_translation_readout.py)在首个HDF5数值读取前核验全部30个文件/张量SHA，使用原deterministic CNN Policy及原causal数据集，不修改共享实现。全support有效行数和gripper sign accuracy必须精确匹配G；七维/平移/旋转/夹爪MSE的abs容差1e−10、rel=0。首批重复包括padding必须逐值相等。63个原始批次输入摘要共享，315个跨模型批次比较、378个输入前后不变检查和30个权重/无梯度后验凭据必须齐全。

HDF5 guard按文件、task、demo、字段及每个causal slice的预定顺序限制读取；拒绝query/demo30+、states、attrs/init_state、整数组转换与read_direct。torch.load仅允许指定30份权重各一次且weights_only；拒绝旧预测数组读取、优化器构造、autograd.grad/backward。任何拒绝尝试保留在失败记录中，不以零预算计数掩盖。全部资源attempted/completed明确记录，失败即保留partial并停止。

结果保存5份label NPZ、30份全合法帧prediction NPY、30份原首批/重复/mask NPZ、输入摘要和事件日志；不存原始RGB、本体输入缓存、private simulator state或复制checkpoint。独立[审计器](../experiments/audit_libero_source_expert_late_translation_readout.py)只用NumPy与metadata，禁止Torch、HDF5、checkpoint和环境操作，独立重算mask/指标、65份数组、G parity、全槽重复与预算，导出90行model×stratum、45行配对、30行task×arm×stratum和900行demo分层CSV。审计不能替代重新观察真实环境，也不能证明闭环根因。

## 执行绑定、产物与停止

所有实现、测试、共享依赖与本说明须在真实payload前写入[执行绑定](../configs/libero-source-expert-late-translation-execution-binding_v1.json)，默认CLI只做metadata预检，run模式才按预算执行。测试使用合成字典、HDF5与CPU Policy，不是额外真实读出。若后续技术修复，保留历史绑定/日志并明确版本，不能重复消耗408批。

执行前24项新增测试、165项重点测试通过（7.95秒）；CPU回归1797项通过（50.73秒），排除`tests/*crafter*.py`、`test_spatial_gather_spt.py`和`test_smoke.py`，不声称包含可选Crafter的全环境pytest。四份Python文件编译通过；CPU屏蔽CUDA的环境检查通过，不将device_count=0误记成GPU实验失败。首次合成Policy测试漏填spatial_pool_size/action_dimension两项，补齐后通过；修正发生在真实payload和执行绑定前，初版失败日志保留，没有额外真实前向。

固定主结果为`/tmp/libero-source-expert-late-translation-readout-20261107-v1.json`，归档目录为`/tmp/libero-source-expert-late-translation-artifacts-20261107-v1`，独立审计与completion另存其中。主结果写入时为pending-audit，之后不回写原字节；独立进程通过后才写完成收据。dated结果报告另记实际进度、预算与结论，不能以冻结代替已运行。

独立审计另生成归档内`readable-results.md`，包含全部task×arm三seed均值及15组末段配对，便于资源等待结束后的直接检查；最终通过状态以completion.json为准。软件验证记录及初版失败日志均在新的/tmp路径保留。

任何SHA、角色、帧身份、重复、输入、权重、G parity或预算不一致均技术停止，不形成完整主结论、不补跑、不选择checkpoint。完整数值即使没有支持末段误差集中也按阴性结果结束。G准入未通过、H未开展及SPT/Knowledge/Module边界保持；不把本次source-support读出升级为正式SPI结果或Module qualification。

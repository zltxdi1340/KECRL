# LIBERO来源candidate几何采集执行绑定说明（2026-11-07）

本执行层实现[已冻结v1科学协议](libero-source-candidate-geometry-capture-protocol-2026-11-06.md)，不修改科学配置、500端点、5步稳定化、现有loss或停止规则。该文件是运行前代码/接口说明；实际运行和结论另记于dated结果报告，避免改写执行绑定。

## 执行范围和计数

固定task0/1/3/4/7、seed20261106/07/08、A_original/B_fomaml、candidate0..3，30个既有500端点、60对、120条新短轨迹。每条最多150个Policy动作，原生成功可提前停止；到前缀上限不标为环境time_limit，也不当作300步失败。

资源逐项记录attempted/completed：权重反序列化30次、模型30个、初态容器5个和数值行20行、新鲜adapter与底层reset各120次、Policy前向/动作各≤18000、零动作600、原生控制步≤18600、物理积分≤465000、影子MjData≤120、影子位置计算及两类状态行各≤18120、bowl碰撞记录≤4638720且每行≤256。活动时间≤3600秒；OS执行命令另设3600秒watchdog。无技术重试、补样本、额外真实smoke或训练。

## 实现与隔离

- [采集入口](../experiments/run_libero_source_candidate_geometry_capture.py)：先仅metadata核对科学和执行绑定，再验证全部30份checkpoint文件/权重hash；此后才创建环境。使用既有确定性pool、原有效CNN配置和原predictor格式，eval/inference模式，不构造优化器、不反向传播。torch.load只准读取这30个未重复的固定路径；h5py.File被拒绝。每个模型使用后再次检查权重与梯度标志。
- [有界adapter](../src/environments/libero_geometry_capture_adapter.py)：调用已审计的injected builder，仅传单个授权初态，禁止默认整库读取；绕过ControlEnv.reset的finally/continue重试循环，直调一次底层reset。global/selected-bank-local/adapter-local分别记录。每条环境关闭并回收，不复用其他初态的控制器历史。
- [独立observer](../src/environments/libero_geometry_observer.py)：每条新建影子MjData，按科学协议复制当前状态、只调用mj_fwdPosition。原生MjSim.step/forward在observer阶段被硬拒绝；每次检查原生buffer字节指纹和原Policy观测摘要不变。全场景原始状态不序列化，几何不进入Policy。只保存bowl有关的碰撞几何，不读取或输出接触力数值作为测量。
- [独立审计](../experiments/audit_libero_source_candidate_geometry_capture.py)：仅标准库和科学metadata模块，不导入采集器、observer、Torch、MuJoCo或环境。独立重算距离、旋转、接触分组、时间、删失、60对/120条身份、30个权重前后凭据、20个初态行边界及全部调用计数，并导出逐条、配对、事件邻域和task×seed×arm表。它不重新运行物理环境，也不声称复核碰撞力或重新求解几何。

实际编译模型无法在零环境调用的科学冻结阶段先取得。本执行绑定固定五个source BDDL字节hash、源码与解析规则；每条真实reset后、首个Policy动作前保存实际编译XML SHA、body/site/geom名称与ID、维度和坐标约定。相同task/global初态的六个模型必须通过公开reset、几何reset和编译映射的逐位比较，共20个参照、100次比较。缺失或不一致按原规则停止，不用猜测的ID代替运行记录。

## 两个实现核对点

合成测试发现，MuJoCo3.2.3会把未启用island/dual-solver等部分C NULL指针呈现为新分配、未初始化的NumPy数组。它们没有原生buffer存储，反复读取就可能不同。指纹现检查全部原生有存储的数值buffer和contact/诊断视图；对这种非原生分配只哈希字段名、类型、形状和“无原生存储”标志，不读取未定义内容。关键复制字段须有真实存储且与影子数组无别名。未放宽状态或观测的一致性容差；原生buffer的字节校验不是力或完整私有状态的测量输出。

代码还确认原Policy本体观测采用原有Observable采样缓存。按同一控制边界归档，不意味着缓存值等于该边界当前qpos。按冻结要求保留原Policy输入、禁止刷新；几何仍由当前qpos在影子数据重算，sim.time标明其物理边界，并额外记录当前native与缓存joint/gripper的L2差。这只是时序质量标注，不从差值推断精确延迟，也不将未知采样时刻冒充当前几何真值；不改变主要几何量、事件起始时刻或采集预算。

## 验证、GPU与运行产物

[专项测试](../tests/test_libero_source_candidate_geometry_capture.py)只使用合成MuJoCo模型和合成metadata，覆盖120条完整流程、原生缓存陈旧时的影子几何、正margin区别、接触上限、动作前后时钟、原生写入拒绝、reset失败不重试、整库路径拒绝、独立审计反例及预算/顺序漂移。31项新增执行测试，连同科学协议和初态读取器共82项通过（7.36s）；排除可选Crafter相关收集项的CPU回归1714项通过（49.21s）。排除项为tests/*crafter*.py、test_spatial_gather_spt.py、test_smoke.py，不声称包含Crafter的全量pytest。日志和原始修正说明在`/tmp/libero-source-candidate-geometry-execution-software-tests-20261107-v3.json`及其引用文件。

选用物理GPU1，UUID为GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff；CUDA_VISIBLE_DEVICES=1、MUJOCO_EGL_DEVICE_ID=1、MUJOCO_GL=egl，Policy逻辑设备cuda:0，CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0和单线程。绑定和启动均检查GPU占用，其他作业占用则不启动，不切GPU或CPU。运行前metadata预检不读取checkpoint、初态数值或真实几何。

执行绑定：[execution v1](../configs/libero-source-candidate-geometry-capture-execution-binding_v1.json)。执行命令和GPU占用快照保存在其中；所有源码和本说明在真实payload读取前hash绑定。结果路径固定为`/tmp/libero-source-candidate-geometry-capture-20261107-v1.json`，原始公开轨迹/geometry sidecar/事件日志目录为`/tmp/libero-source-candidate-geometry-capture-artifacts-20261107-v1`；独立审计与CSV使用新的/tmp路径，均拒绝覆盖。

后续只按已冻结事件定义描述连续几何和接触代理，不用本次结果选checkpoint、seed、loss或方法，不推断单一根因。G阴性决定、H未开展、held-out/qualification/reserved封存以及SPT/Knowledge/Module边界保持。

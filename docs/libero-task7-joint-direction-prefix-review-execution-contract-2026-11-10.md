# task7 前35步数值复核：执行契约（2026-11-10）

对应[冻结方案](libero-task7-joint-direction-prefix-review-plan-2026-11-10.md)和[科学配置](../configs/libero-task7-joint-direction-prefix-review-diagnostic_v1.json)。科学SHA保持 `d86709b75f938060f8c252c4743e6a9f90c7ce58b35f2519a0191cb56a1b5bf7`；不修改原方向v1的失败状态、源码、容差或pending数据。用户已授权实现、绑定并执行本次新诊断，冻结方案中“本轮只冻结”描述的是上一冻结阶段。

## 入口和独立性

- P及watchdog：[run_libero_task7_joint_direction_prefix_numerical_review.py](../experiments/run_libero_task7_joint_direction_prefix_numerical_review.py)。默认preflight，不默认启动研究。
- C：[run_libero_task7_joint_direction_prefix_cross_review.py](../experiments/run_libero_task7_joint_direction_prefix_cross_review.py)。U自由路径先完成，再执行S36、N36。
- V：[audit_libero_task7_joint_direction_prefix_numerical_review.py](../experiments/audit_libero_task7_joint_direction_prefix_numerical_review.py)。标准库独立读出派生算术，不导入NumPy/Torch，不读原数组或新trace。
- guard：[libero_task7_joint_direction_prefix_review_access.py](../experiments/libero_task7_joint_direction_prefix_review_access.py)；[执行测试](../tests/test_libero_task7_joint_direction_prefix_numerical_review.py)。

P直接复用原绑定NumPy measure/trial，C直接复用原独立Torch measure/trial/construct_direction；不重写loss、梯度或归一化。新Jobs适配器和内核钩子记录实际调用。V不调用P/C比较或方向函数，独立实现标量分解、误差范数、参数更新核对、网格/接受/停止规则及事件对账。共用的只是冻结身份/字段和IO/格式契约。

新binding钉住8角色、此契约、科学配置及原785依赖闭包；必须先完成合成检查与资源包络。每个直接payload API也重新验证配置、绑定与调用pin，禁止仅在CLI设门。绑定后的研究执行不修改代码，0重试、0覆盖；单头k0..35，k35禁止试步。无Policy/CNN/环境调用，不开放probe/query/正式封存角色，无SPT/Knowledge/Module变更。

## 访问和调用门

NumericGuard限制三个原NPZ（P/C各一次）及C的一次新trace；np.load必须allow_pickle=False，字段各解码一次。原归档可包含未准入成员，但不能读取；特别current的logits未解码。新trace必须恰好10成员。原数组逐字段只读，fit统计和上下文只用固定2777行/own20示教/t≥1；C独立重建标准化并精确核对P的z/y/p0摘要，编码器曾见全部30示教这一边界保留。

原head的0/100/250/500权重和optimizer_steps均物理解码；0仅一致性，100/250不评分，不恢复原Adam状态；只复制500作为P/U起点。S只读P实际参数，N只读P实际Gb/G_T；S/N禁止试步、接受更新或额外梯度。U按自身接受路径前进，不能接收P的更新。提前停止保持原原因，之后U数值null、不填0。

P只准原受控einsum内核；C只准当前注册fit张量上的Float64 linear与注册w/b的autograd.grad。每梯度状态2线性形式/3梯度，每试步1形式，方向构造与norm/dot/vector另计账。禁止其他reader、mmap、原始zip/文件描述符绕过、PT/HDF5、Module调用、优化器构造、backward、原Policy前向、环境导入。该guard是研究访问契约，不声称OS安全沙箱。

P先把state/参数/梯度写入内存及逐状态CSV并flush、写after事件，再核对历史锚；36参数SHA/258试步须复现。C先保存所有U/S/N实际值再形成比较，无跨实现q差异门。before事件含参数输入摘要；N含Gb/G_T输入摘要，after含实际q/hex及其梯度摘要、计算/方向账本和资源数据。失败JSON只使用已在内存值，不在handler中新增读取、SHA、范数、前向/梯度或资源监测；访问尝试与完成次数分别保留。

## 预算和派生数据

沿用科学配置：P36梯度状态/35更新/258试步；C≤72梯度状态/216autograd/35更新/700试步，另N36纯方向；合计≤108状态/70辅助更新/958试步/1174线性形式。方向合计≤144构造/576norm/288dot/576向量形式/148032坐标。P/C各重建一次fit标准化。

归档读取7次/解码40成员/SHA15次（P7、C8），新trace1写10成员；禁止旧CSV读取。5新CSV≤10282行/64MiB，trace≤16MiB，2事件JSONL≤292行/8MiB。V每CSV/JSONL各一遍，CSV前后10SHA/事件前后4SHA；连同生产端5CSV/2事件各一次SHA，合计15CSV SHA/6事件SHA。

C和V分别核算冻结比较字段所需的派生代数：每阶段≤504误差L2、504 maxabs、129528误差坐标；relative_L2还需每点至多10个不同被减向量范数（≤360），复用同点重复分母，0分母为null。这360是生成既定relative_L2字段的分母计算，单列计账，不冒充已计入方向norm预算，也不新增前向/梯度或科学优化预算。V另≤37008坐标望远镜核对、108个q参考/324norm/324向量形式。V的q参考与P/N/S/U实值差异是观测，不能用它丢弃本轮要解释的超差。

V核对P历史锚、CSV/JSON一致性、raw参数/梯度摘要、14类误差、带符号q分解、实际参数接受更新、全部试步首接受、固定M500和终止原因、事件顺序/计数及读出预算。两端近似范数用原绝对1e-10比较；超差标志严格abs(delta)>1e-10，无rel容差。成功仅表示新观测的独立一致性检查完成，不认证旧v1或主14/18，不将该诊断解释成正式结果。

## 环境、资源和启动

Python3.10.22/NumPy1.22.4/Torch2.4.1+cu124，CPU。CUDA隐藏，PYTHONHASHSEED0，OMP/OPENBLAS/MKL各1线程；C实际设置/记录Torch线程、interop线程与确定性选项，记录NumPy/Torch后端和CPU/平台。无新随机抽样，环境seed=null、初态=[]。P≤180s/512MiB，C≤300s/768MiB，V≤120s/256MiB；active资源每个算术任务监测，watchdog总≤720s，记录命令/PID/UTC/退出码及实际墙时。720s包含启动和元数据校验；子进程超时另保留总门，不转移预算。

完成绑定后只启动一次：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n libero-kecrl python -m experiments.run_libero_task7_joint_direction_prefix_numerical_review --mode run
```

watchdog依次启动P/C/V，写launch/process与各阶段日志；P写primary-result/trace/2CSV/事件，C写cross-result/3CSV/事件，V通过后写本轮independent-audit、新result和completion。任何阶段技术失败保留partial并停止，不修代码重试旧v1。研究执行完成后另写dated结果报告；README/roadmap/500计划只前置新状态，保持旧文本和绑定文件。

## 绑定前验证记录

42项新增执行检查已通过，包含完整三进程合成与P/C最大数值包络；完整合成的P试步35是合成条件，不冒称研究258试步复现，最大包络另实际执行P258/C700试步。合成独立审计93401项通过，V没有NumPy/Torch或归档调用。最大P36状态/258试步用时0.126s、51.328MiB；C72状态/216autograd/700试步/N36用时0.290s、450.082MiB。完整合成C峰值578.672MiB，在768MiB内。

首版合成因/proc/self/status解析为/proc/<pid>/status而被新guard误拦，32通过/2失败；绑定前统一为当前进程解析路径后34通过，再补充8项输入隔离/超差保留/失败证据检查后42通过。失败日志/XML保留，此处没有研究payload或真实重试。CPU回归、环境与最终编译记录将在执行绑定validation中钉住；合成通过不替代真实数据上的复核与终检。

末轮复查补充U_states严格1..36/整数及U_updates对应关系、P/C记录数、CSV冻结列/路径/行数准入；新增5项后47项通过。CPU回归3759通过、1项测试宿主RSS失败（单测直接继承整个pytest进程的高水位，不能当独立研究进程RSS）。将该单测移入独立子进程，保持512MiB原门，并对最终47项（含完整三阶段及资源包络）再次复核。CPU失败原XML/日志保留，不改写为全通过；其余共享代码未改，不重复整套9分钟回归。绑定validation区分原CPU失败与最终定向解决记录，先确认唯一失败已解决再准入研究。

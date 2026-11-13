# task7 λ1 起点MSE接受条件对照：执行契约（2026-11-09）

本契约实现[已冻结科学方案](libero-task7-mse-nonregression-acceptance-plan-2026-11-09.md)，不改变科学配置 SHA `b470d22c51b11848e2aa920cd2969dbf5cfe6520e8a36f481ce3be55b433515b`。只在原 λ1 Armijo 接受规则上增加固定辅助500起点 MSE 上界；18头、两臂、各最多50更新/每轮12试步，主1800/独立审计另1800上限。没有新原Policy训练或闭环，原Policy500/未来5步、表示和loss不变，正式角色封存。

## 执行和不可变性

主入口 `experiments/run_libero_task7_mse_nonregression_acceptance.py` 的 CLI 分 preflight/evaluate/run；默认只做元数据检查。`run` 以新进程顺序启动主计算与独立审计，记录命令、PID、UTC起止、退出码、环境和总耗时，主/审计均成功且总≤960秒才完成。主活跃≤300秒/512MiB、审计≤600秒/768MiB；Linux当前地址空间VmHWM/VmRSS计账。CPU、CUDA隐藏、PYTHONHASHSEED=0与BLAS单线程，版本锁定在科学配置。技术重试0，任何已有主结果、产物目录、launch或process记录即拒绝，日志/部分失败不覆盖。

CLI 和两个 payload API 都调用新 science/binding 校验，绑定七角色、本文、科学配置及父736依赖闭包后才创建研究目录或读取研究归档。配置、父provenance、代码字节或预算不符即拒绝。仅合成测试在隔离子进程显式替换 `validate_call`，替换函数必须确认所有归档和历史JSON均由本次pytest临时目录拥有；生产代码没有绕过开关。

两个优化臂由同一个已保存 λ1 辅助500头拷贝，先 original_armijo，后 start500_mse_guarded_armijo；每个头的表示/目标批次同一只读对象。source准入核对 weights/bias 等八个成员的归档来源，提取保存索引3的500参数；每个来源只准入一次，两条路径重复或改起点均拒绝。原输入、归一化表示和起点用进程内摘要核对不可变性；审计不要求跨实现参数字节相等，以绝对1e-10核对数值坐标。

## 数值和预算guard

`experiments/libero_task7_mse_nonregression_acceptance_access.py` 沿用既有归档白名单和更新状态机，新增单来源×优化臂准入、固定起点上界、MSE/P守恒及两臂终点保存门控。每个当前状态只准两种线性形式、三梯度；试步只准一种线性形式且无梯度。试步必须是当前参数沿完整负梯度的第j个固定α，不允许跳点或在第一次接受后继续。`trial_end`同时核对MSE3/P/T/F并记录原Armijo、固定MSE、应用门、接受和原因；拒绝试步也计入前向、fit行数、两项误差计算与试步总量。

MSE上界在每个头首次baseline状态建立，候选首次状态必须精确相同，后续不能改成上一状态或原Policy误差。候选所有接受状态严格不超过固定上界，无epsilon；两条件均失败与单条件失败分别记录。每轮12次全部拒绝为正常 `line_search_exhausted`，再细分有原Armijo可行试步但被MSE否决、原Armijo全不通过；零更新保留、不算进展。正常停止优先级及唯一最后接受终点保持原规则，不能转移预算或挑checkpoint。

每阶段18次fit标准化、18次表示复用、18个源头和36条路径、≤1836状态/5508梯度/21600试步，行数和坐标量全部按冻结额度核对。源归档25个/161成员每阶段各读一次；审计额外读18个新归档/126成员。主写18个新NPZ/126成员，每个只存[500起点、baseline终点、candidate终点]。保存前必须两条路径都完成，参数逐值及scaler摘要匹配注册来源；重复保存、快照选择或附加字段拒绝。

归档IO guard覆盖np.load/成员/ZIP、raw open/Path/io.FileIO/os.open、mmap/fromfile/替代CSV读取、torch模型/优化器及环境导入。source仅允许声明的7个缓存及18个λ1头，λ0/历史优化头、原.pt、RGB/HDF5和其它路径均拒绝；缓存虽包含30个support示教，数值工作只接收经注册的本头fit行。审计只读主表，不得写主表或改头；该guard是实验访问契约，非操作系统安全沙箱。

归档字节SHA：主68、审计86，总154；读取68次/448成员；主写18NPZ/126成员。新七CSV在审计前后核SHA，审计逐行重新核对，主JSON保持原始pending状态，以独立audit/completion与两进程exit0表示完成，不回写主结果。来源CSV读取/SHA为0；新产物CSV属于审计输出核验。

## 历史复现和独立审计

`History`仅把旧固定优化JSON中的λ1投影为声明标量列；不把旧参数向量交给数值计算，不读历史CSV或新增50头NPZ。每阶段核对全部918个baseline状态、7030个试步、36个端点和18个头，记录精确投影数量与标量比较数；任何缺行/错序/分支/决定不符即技术停止。新的baseline复现不使用其未来query或闭环结果。

`experiments/audit_libero_task7_mse_nonregression_acceptance.py` 不导入新/旧主数值实现，复用既有独立Torch Float64测量和fit归一化验证。审计从自身source500副本独立生成两条路径，用autograd得到三梯度，另算单侧方向导数、trial MSE/P和两个接受条件，逐项核对guard返回决定；两臂各自更新、全部拒绝试步、所有新表及三点NPZ均重演核对。完整端点参数、梯度向量、clip/hinge摘要及每步动作目标数值按科学容差核对；分支/接受/ID/计数容差0，接近阈值时不放宽。

派生配对差值、三折值/等折均值、原联合优化线索及新受保护fit线索独立重建，保留全部18头。候选MSE保持是算法不变量，不等于优化成功；0更新单列，probe/query/闭环改善均不在本轮可回答范围。旧cue/G失败/H未运行、SPT/Knowledge/Module状态保持。

## 验证与启动

运行前完成新局部测试、同规模完整合成主/审计、最坏1836状态/21600试步的NumPy和Torch资源包络、相关重点与CPU回归、环境检查、编译及git diff --check。具体XML/log/选择清单与资源证据在新绑定的validation字段记录；合成优化仅用pytest临时数组，不记作研究结果。工作区不变性快照仅覆盖非results代码/文档/设置；不重复上一轮的通用历史归档哈希偏差。

```bash
conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_task7_mse_nonregression_acceptance --mode preflight
conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_task7_mse_nonregression_acceptance --mode run
```

默认产物 `/tmp/libero-task7-mse-nonregression-acceptance-20261109-v1.json` 和同前缀 artifacts目录；原研究产物不改动。结果与实际provenance另记dated结果报告，更新README/roadmap/500计划入口，不修改已哈希的科学方案、配置及历史报告。

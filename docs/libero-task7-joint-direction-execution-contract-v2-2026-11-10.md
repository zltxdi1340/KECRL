# task7 方向对照v2执行契约（2026-11-10）

[科学方案](libero-task7-joint-direction-plan-v2-2026-11-10.md)和[冻结配置](../configs/libero-task7-joint-direction-diagnostic_v2.json)SHA保持 `253061977b288bbec50a232e49b630f0f1e825370bdafeeabdc248e0106ac50c`。用户已授权冻结后执行完整18头；本轮不打开任何正式评价角色。旧方向v1及单头数值复核文件均保留原字节。

## 代码与角色

主入口：[run_libero_task7_joint_direction_diagnostic_v2.py](../experiments/run_libero_task7_joint_direction_diagnostic_v2.py)；独立审计：[audit_libero_task7_joint_direction_diagnostic_v2.py](../experiments/audit_libero_task7_joint_direction_diagnostic_v2.py)；[访问、trace与预算guard](../experiments/libero_task7_joint_direction_access_v2.py)；[元数据](../experiments/libero_task7_joint_direction_protocol_v2.py)、[冻结测试](../tests/test_libero_task7_joint_direction_protocol_v2.py)、[执行测试](../tests/test_libero_task7_joint_direction_diagnostic_v2.py)。绑定7角色+契约+science和父795依赖闭包，预计804个唯一文件；不包括研究数组。绑定检查实际集合和每个SHA，payload API各自复验，默认CLI只预检。

P复用原NumPy measure/trial/optimize及历史控制规则，只新增有界trace和上下文摘要。C禁止导入主数值实现，复用原独立Torch measure/trial/replay，分别执行U完整自身路径和S同P参数读出。没有第三条S优化路径、额外网格搜索或新方向/λ/预算选择；P/U每臂最多50，原Policy保持500。

每头先运行U两臂、再运行S该头全部实际P参数。S参数必须是新P trace中被参数SHA精确锁定的Float64值，fit索引/scaler与原500逐字节匹配，z/y/p0摘要与P相同。额外沿P方向切线使用已缓存raw线性输出，仅增加一个切线形式。每个P实际trial参数严格按Pθ+αdP重建，禁止跳α、重复读出、超过20级或在k50试步。S只有明确注册的当前参数w/b可求梯度，0更新/优化器/梯度累积。

NumPy/Torch工作guard验证当前fit张量来源、参数/方向/alpha、梯度参数ID、每状态形式/梯度数、真实调用次数及资源；源NPZ、成员、zip/文件描述符、替代reader、mmap、pickle、HDF5、原Policy/Module、环境导入均限制。范围是研究契约，不声称OS沙箱。schema的dtype/形状/有限值/范围检查可以覆盖原成员；梯度、目标、标准化矩和影子读出只投影本头fit。不称原成员解码的字段级物理隔离。

## 同参数门与离散分歧

原绝对1e-10/相对0用于全部同参数浮点字段、三梯度/方向坐标、P方向精确导数和P试步目标。raw clip/hinge分支摘要、计数以及身份/来源/shape严格一致；不把不同P/U参数的记录放进这个门。局部真实超差或数据/预算损坏技术停止，记录两端实际值、P参数摘要和可用坐标，0重试。

方向status/selected_joint、正常停止条件、严格接受影子的差异逐项记录。stop_S是同P点正常停止条件和已访问P网格的影子耗尽标签；沿P方向的影子使用自身S起点M500、S当前目标及精确沿P方向sT。它不是S方向独立优化结果，也不推断P未访问的网格。任何非等价严格决定都不能因数值接近改判。

C独立校验P的七CSV与P JSON逐位Float64序列化一致性、每个实际P参数/分支/目标/梯度/方向、每次P接受参数转移、首接受/固定M500/Armijo/停止和唯一末点。P头的端点指标和科学线索从其实际状态重建。U按自身精确规则执行，独立完整记录P/U的科学结果、更新/停止差异，不强制跨路径或跨科学cue一致。稳健线索另依冻结定义报告。局部门全过不意味方向主假设通过，更不意味着Module资格。

## 预算与归档

冻结上限：P/U/S各1836梯度状态；P/U各1800辅助更新、S0；P/U/已访问P trial影子各36000目标读出。C真实autograd≤11016，P/C线性形式≤39672/81180，总≤120852；每种原始调用、方向和行数分别计账。P/C各18次fit标准化，S只复用上下文。P保持历史控制398次更新门；U不被强制历史逐步对齐。

原25个NPZ每阶段各一次，P新增18个14成员文件、C读取这些文件后另写18个21成员文件。总68读取/574解码/172归档SHA、36新写/630成员写；每阶段新增NPZ≤80MiB。保存P/U所有测量参数及三梯度/方向，另保存S参数/三梯度/方向；新trace不作Policy加载或checkpoint选择。每种trace逐状态登记坐标单列，序列化/打包内存由RSS与字节上限约束。

P七CSV≤51840行，C两CSV≤37836行，合计≤89676。C对P CSV各一遍及前后14 SHA；加生产端共23 CSV SHA。P scalar JSON是新主程序pending记录；C独立过门后新写independent-audit、最终result及completion，不覆盖P或旧v1。C JSON≤192MiB。失败JSON≤1MiB，仅写已在内存值、资源快照、访问和计算/方向计账；handler不新读数据/前向/梯度。

环境Python3.10.22/NumPy1.22.4/Torch2.4.1+cu124、CPU，CUDA隐藏，PYTHONHASHSEED0、OMP/OPENBLAS/MKL和实际Torch/interop各1线程，记录后端/设备/seed/UTC。无新随机采样，环境seed=null、初态=[]。P≤300s/512MiB，C≤900s/1024MiB，watchdog从元数据预检前开始计总≤1500s，记录命令/PID/退出码/墙时。原训练/正式评价预算不变。

绑定后只运行一次：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n libero-kecrl python -m experiments.run_libero_task7_joint_direction_diagnostic_v2 --mode run
```

## 绑定前验证

最终31项定向检查通过，含完整18头合成双进程；另1项最大C读出包络通过（3672状态/11016autograd/72000trial/1836额外形式）。合成P1797更新/1833状态、U另1797更新/1833状态、S1833状态，全部同参数和路径合同过门；2125670数值/规则核对、559481 CSV精确核对通过。合成不是研究结果，不要求合成复现真实398控制次数。

首轮24通过/2失败：新主入口缺失math导入、未注册S角色n未初始化；均在绑定前修正。随后补齐S重复/跳α/重复状态门、P科学头端点重建，最终完整合成通过。全部失败日志/XML保留。独立最大包络证明另存固定/tmp路径，不依赖pytest临时目录保留期。CPU回归、环境、编译与工作区不变证明在binding.validation中钉住；可选Crafter相关文件仍显式排除，不混写为LIBERO失败。

最终验证：CPU选择3795 passed/31 deselected（564.78秒），69个Crafter关联文件排除；先前pytest宿主RSS单测已使用独立子进程并保持原512MiB门，此次整套选择无失败。最后补足同参数门之前的在内存参数/P梯度坐标日志后，完整18头合成另复核1项通过（38.64秒），数值内核/预算未改。最大C包络21.602秒、450.461MiB。Python3.10编译、CUDA隐藏环境、git diff --check通过，日志和XML在绑定validation逐项钉住。

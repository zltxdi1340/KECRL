# LIBERO task4/7 A/B/C几何外推诊断冻结协议（2026-11-05）

本项补齐[四任务现有结果表](libero-task0-task1-task4-task7-results-comparison-2026-11-05.md)中task4/7的A/B/C缺项，沿用task0/1的生成器、CNN-only Policy、优化器、固定预算和评价接口。是已暴露source内部的diagnostic，不是正式held-out迁移、SPT接受或Module qualification。用户要求继续实验并在本周完成LIBERO这一轮；本项不访问封存角色，不为达到好看的成功率追加训练。

## 问题、分层和三臂

检验来源经验和元目标在两类不同接近几何上，能否改善固定预算适应和闭环。A保留原始生成器；B以source support一步SGD后的query梯度做上下文FOMAML；C在相同source批次/容量上普通pooled BC。B/C每fold、每seed各50次生成器Adam；每次四个source SPI，B每SPI一步SGD，lr0.001；外循环lr0.0003。

| 探针 | source更新任务 | 来源覆盖几何 | 探针独有、来源缺失的几何 |
|---|---|---|---|
| 4 | 0,1,3,7 | table_region / movable_object_surface / fixture_surface | open_container |
| 7 | 0,1,3,4 | table_region / movable_object_surface / open_container | fixture_surface |

这与task0/1的四类几何均覆盖不同。task4/7须单列为**来源中缺少探针几何的外推诊断**，不能补表后声称四任务完全同分布，不能将差异归于任务固有难度。task4初态抽屉已经打开，不新增开抽屉目标。task3仍只作来源，未扩充任务划分。

## 训练与独立读出

- source角色仍是0/1/3/4/7。demos0..29为support、30..49为query；来源query是训练反馈，探针query只作评价。仅causal `obs[t-1] -> action[t]`、t>=1，不重建t0、不访问simulator state。
- seeds固定20261022/23/24。每fold、seed、臂从配对生成器初始化；B/C来源批次相同，三臂探针适应批次顺序相同。每个探针Policy为500次Adam，lr0.0003、clip1、batch8、sequence10、stride10；均匀七维MSE，无jitter、历史头、局部loss或新权重。
- 保存0/50/150/300/500的support/query曲线；每fold全部9个fit完成后才创建并读取该探针query。使用fold顺序计数，不把非连续task编号当已完成fold数量。固定500端点，曲线不用于选checkpoint或改预算。
- 新增18个Policy、9000次探针适应更新、600次生成器更新、1200次临时SGD、4800次source数据梯度、2400次VJP、90个曲线checkpoint。B独有临时Policy写入，不能宣称全部FLOP或耗时匹配。
- 只使用物理GPU1，启动前检查无计算进程、显存<=512MiB、利用率<=5%；GPU0不占用。确定性等价池化、关闭TF32、CUBLAS `:4096:8`，CPU线程1，host tensor cache上限8GiB。环境/Python/依赖/设备、实际前向与梯度、分段耗时、输入/代码/权重hash写入provenance。

新的版本化入口继承历史v2执行逻辑，旧代码、配置和产物不变。新离线配置为[geometry-extension v1](../configs/libero-source-internal-geometry-extension-diagnostic_v1.json)，入口为[训练runner](../experiments/run_libero_source_internal_geometry_extension_diagnostic.py)，[独立审计](../experiments/audit_libero_source_internal_geometry_extension_diagnostic.py)不导入训练runner或Policy，仅复查权重/指标/事件/合法source action长度元数据。测试通过后、首次训练前绑定本协议、入口、审计、核心依赖及测试hash。

## 固定500端点配对闭环

训练完整且独立审计通过，才能从全部18个500端点生成闭环manifest；不按离线效果筛模型。随后把来源结果/审计、全部端点hash、初态bank元数据和执行代码绑定，才解码candidate前20行。闭环条件预先固定如下：

- tasks4/7 × 三seed × A/B/C × candidate0..19，共360条**新增**episodes、18个adapter；每模型独立adapter，内部按0..19递增复用，task→seed→A/B/C→初态顺序。
- 环境seed20261022，稳定化5个零动作，最多300个Policy动作，native horizon305；最多108000个Policy动作+1800个稳定化动作，native step上限109800。
- 只数值解码两份source bank的candidate前20行，共40行。qualification20..39、reserved40..49及validation/held-out/final evaluation不得打开；不使用全bank反序列化默认路径。
- CNN-only确定性推断，物理GPU1/EGL1、现有动作接口。每task/state九个模型的post-stabilization公开reset digest须逐位相同，共40个reference、320次比较。task4严格保留reset历史，禁止跳过前面状态。
- 保存逐步action及公开joint/gripper状态、相对reset偏移、终止原因和完整provenance，不保存private live state、初态向量或RGB像素。闭环0训练、0反向、0示教payload读取。
- 独立审计从360份episode重算task×seed计数、B-A/B-C四格、动作数、公开offset和reset等式，并核对全部18份权重。不得将失败中止计作300步失败或合法主结果。

## 停止、解释和交付

只因hash/角色/预算/数值/资源或reset门槛失败中止，保留partial与失败状态，不放松门槛、不按效果重试。全部seed及正反结果报告；不做事后显著性检验，不将多轮candidate使用称未见泛化，不混合跨任务总成功率。

无论B是否优于C，都交付训练/闭环两份独立审计、逐seed和五点误差曲线、四任务方法结果表与几何覆盖分层。单个seed获益不能当稳定元学习优势；低成功率也不单独证明视觉、历史、数据覆盖或动作头是根因。继续封存validation、held-out、qualification、reserved和final evaluation，不接受活动SPT、不更新Knowledge、不注册Module。本周完成范围和后续正式协议门槛见[周内收尾计划](libero-weekly-completion-plan-2026-11-05.md)。

# task7 MSE门停止点方向/网格只读分析结果（2026-11-10）

已完成冻结的全部11头只读分析和独立审计。**6个零更新头属于约束边界上的当前方向局部冲突；5条短路径仍有正误差余量，支持连续数学目标下存在更小可行步的条件性推论，但既有12步长网格没有找到。** 两类不能统一归因为“更新预算不够”，也不能据此认定全局无解或表示/loss唯一失败。

这是保存的source fit辅助优化记录分析，不是新训练、SPT更新、Knowledge Evolution、SPI正式结果或Module qualification。没有评估新α、计算新梯度、读取原数组/checkpoint或访问任何封存角色。

## 冻结范围与实现

依据[科学冻结](libero-task7-mse-gate-stop-direction-grid-plan-2026-11-09.md)和[执行契约](libero-task7-mse-gate-stop-direction-grid-execution-contract-2026-11-09.md)。固定task7、A_original/B_fomaml来源、训练seed20261106/07/08；11个既有candidate末轮耗尽头（6零更新/5短路径）全纳入。7个完成50步头只留metadata分母，未读取其梯度做新对照。结局已知分层不能当随机因果实验；每头使用已见source-support0..29中的本折20个fit示教、t≥1，不构成未见泛化。

只投影昨日endpoints/parameters/trials三表：22起止点、5654个坐标行（每点257个参数坐标的四种已保存向量）、132个末轮试步。parameters表的parameter权重列不转数值，未选行只做词法角色匹配；CSV整行文本会被解析，不能宣称这些列字节未接触。原Policy仍500 updates、未来stabilization5步、表示/λ1 loss/搜索方向/网格/训练预算及checkpoint规则不变。probe/source-query/validation/held-out/qualification/reserved/final封存，旧cue/G失败/H未运行及SPT/Knowledge/Module状态保持。

## 结果：两类停止的证据不同

沿用 `A=MSE3/7`、`T=A+P`，d为保存的负总梯度。M500是该头辅助500起点专家MSE，余量 `S=M500−Mstop`；sM是沿d的专家MSE单侧导数，sT是组合目标T的保存精确单侧导数。这里MSE是action[2]的fit误差，不是成功率、夹爪误差或真实机器人阶段标签。

| 停止组 | 头数 | 起点→停止点的MSE方向导数 | 停止点余量 | 冻结判据支持什么 |
| --- | ---: | --- | --- | --- |
| 零更新 | 6 | 均为正，点/向量未变 | 全部0 | 当前d在边界附近提高MSE；沿同方向把α缩到足够小，仍违反约束。 |
| 短路径 | 5 | 全部从负变正 | 全部>1e-10 | 总目标仍有负方向导数且约束有余量，支持局部小步存在性；没有实际验证小步。 |

11个停止点均有sM>0、sT<0，且Gb与Gp的cosine均为负（−0.995494至−0.609861）。在这些点，当前方向继续降低T的同时局部提高专家MSE；5条短路径先降低MSE后方向转变，6个零更新头一开始就存在这一取舍。这是固定表示/λ1目标下这些保存点的优化几何证据，不是环境因果机制或闭环根因。

反证须同时保留：B_fomaml/20261106/fold0起点的Gb/Gp cosine已经为−0.925640，但当时sM=−0.00375010、sT=−0.00077089，两个量沿原方向都下降；只看梯度相互夹角为负不足以判定当前方向必定提高MSE。其它4条短路径起点的cosine为正，停止时全部为负；判定依赖实际方向导数及约束余量。

全部22点保存的clip边界tie数与hinge零点数均为0，因此本组可用 `sM=7 Gb·d`，且保存sT与G·d的一致误差最大约2.08e-17，远小于冻结数值带1e-10；没有近零/clip未知分类。这只描述这些点本身，不说明有限步长途中没有跨越clip/hinge分支。实现仍保留拐点资格guard和合成反例，不能把本组的光滑性当所有未来点的默认属性。

## 六个边界头：局部方向冲突

| 来源 | seed | fold | sM（正值） | sT（负值） | Gb/Gp cosine |
| --- | --- | --- | ---: | ---: | ---: |
| A_original | 20261106 | 2 | 0.00484363 | -0.00347685 | -0.713930 |
| B_fomaml | 20261106 | 1 | 0.01647502 | -0.00081999 | -0.995494 |
| B_fomaml | 20261106 | 2 | 0.00704357 | -0.00369915 | -0.774804 |
| B_fomaml | 20261107 | 1 | 0.00826121 | -0.00495183 | -0.916667 |
| A_original | 20261108 | 2 | 0.00173100 | -0.00332052 | -0.609861 |
| B_fomaml | 20261108 | 2 | 0.00310393 | -0.00217905 | -0.772816 |

六个头S=0，sM为0.001731–0.016475，sT严格负，全部四个起止向量变化范数为0。它们不是因用满50步而停止；原方向在足够小正步长邻域内会增加MSE，单纯沿同方向继续减小α不能在该局部邻域满足现有固定起点上界。不排除更大非局部步、其它方向或全局可行解，也没有为它们搜索新方向。

## 五条短路径：有限网格与正余量

| 来源 | seed | fold | 已接受步 | 起点sM | 停止sM | 余量S | 线性α尺度S/sM | 尺度/原最小α |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_original | 20261106 | 1 | 3 | -0.04394976 | 0.01858576 | 1.062345e-06 | 5.715909e-05 | 0.117062 |
| B_fomaml | 20261106 | 0 | 3 | -0.00375010 | 0.00615984 | 2.598607e-06 | 4.218627e-04 | 0.863975 |
| A_original | 20261107 | 1 | 5 | -0.01263900 | 0.00110780 | 1.867343e-07 | 1.685629e-04 | 0.345217 |
| A_original | 20261108 | 0 | 11 | -0.02096337 | 0.00047935 | 1.450650e-07 | 3.026295e-04 | 0.619785 |
| B_fomaml | 20261108 | 0 | 4 | -0.03319268 | 0.00320385 | 1.353283e-06 | 4.223926e-04 | 0.865060 |

正余量为1.45e-7–2.60e-6；起点sM全部为负，停止时均为正，总目标方向导数始终为负。因为MSE连续且S>0，总目标真实右导数sT<0，在由保存数据定义的连续数学目标中，足够小α可同时保持误差约束并满足Armijo；现有有限网格没有找到。这是条件性存在推论，不是已运行更小步长、未证明浮点可接受步或后续优化收益。

原最小α为1/2048=4.8828125e-4；五个线性尺度S/sM为5.72e-5–4.22e-4，只有原最小步的0.117–0.865倍，均落在原网格最小值以下。线性尺度不是安全步长、上界证明或新α选择；有限步长的实际高阶/分支变化仍存在，不能直接把该尺度设为下一次步长。各已有试步的MSE割线和一阶残差已逐行保存，没有做任何未试点插值前向。

## 最后一轮12试步逐头复核

| 来源 | seed | fold | 组 | 原Armijo通过 | MSE门通过 | 接受 | 最小MSE越界量 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A_original | 20261106 | 1 | short | 4/12 | 0/12 | 0/12 | 8.594161e-06 |
| A_original | 20261106 | 2 | zero | 7/12 | 0/12 | 0/12 | 2.408355e-06 |
| B_fomaml | 20261106 | 0 | short | 6/12 | 0/12 | 0/12 | 4.153754e-07 |
| B_fomaml | 20261106 | 1 | zero | 5/12 | 0/12 | 0/12 | 8.077789e-06 |
| B_fomaml | 20261106 | 2 | zero | 6/12 | 0/12 | 0/12 | 3.490131e-06 |
| A_original | 20261107 | 1 | short | 7/12 | 0/12 | 0/12 | 3.549920e-07 |
| B_fomaml | 20261107 | 1 | zero | 5/12 | 0/12 | 0/12 | 4.242345e-06 |
| A_original | 20261108 | 0 | short | 7/12 | 0/12 | 0/12 | 8.923096e-08 |
| A_original | 20261108 | 2 | zero | 6/12 | 0/12 | 0/12 | 9.723917e-07 |
| B_fomaml | 20261108 | 0 | short | 8/12 | 0/12 | 0/12 | 2.137952e-07 |
| B_fomaml | 20261108 | 2 | zero | 6/12 | 0/12 | 0/12 | 1.603771e-06 |

132个已有试步全部因MSE上界而不可能接受；其中67个通过原Armijo，另外65个同时未过原Armijo和MSE门。原接受/拒绝均按严格规则复核，不以1e-10描述带放宽。ΔT=ΔMSE3/7+ΔP、固定C6、缓存sT及Armijo bound均一致。正越界不是“近零可视为通过”；本轮不重新训练、不试新α、不重放环境。

## 实际provenance与审计

科学配置[readonly_v1.json](../configs/libero-task7-mse-gate-stop-direction-grid-readonly_v1.json)，SHA `ab870764b427d77bfc95fe3c6757e87b16d94d9c0232ffcf756079e3c1526c8f`；新绑定[execution-binding_v1.json](../configs/libero-task7-mse-gate-stop-direction-grid-execution-binding_v1.json)，754依赖、SHA `c17ac95d96fa3b19e5bf3ce083b020851ff1da3848359e1ae4d7d8a4957709eb`，在真实来源CSV SHA/数值读取前完成。既有冻结配置、协议、代码/报告和来源产物保持原字节。

```bash
conda run --no-capture-output -n libero-kecrl python -m experiments.analyze_libero_task7_mse_gate_stop_direction_grid --mode preflight
conda run --no-capture-output -n libero-kecrl python -m experiments.analyze_libero_task7_mse_gate_stop_direction_grid --mode run
```

实际Asia/Shanghai 2026-11-10T01:04:46.546167+08:00 → 2026-11-10T01:04:50.282568+08:00；UTC 2026-11-09T17:04:46.546167+00:00 → 2026-11-09T17:04:50.282568+00:00，总墙钟3.736157秒。产物20261109前缀是冻结标识，实际本地执行日为2026-11-10。两子进程exit0，技术失败/重试0。Python3.10.22、纯stdlib、CUDA隐藏、PYTHONHASHSEED=0、OMP/OPENBLAS/MKL各1线程；没有新训练/环境seed，来源seed20261106/07/08仅是缓存provenance，环境seed=null、初态[]、交互步0。

| 阶段 | PID | 活跃秒 | 峰值MiB | 来源CSV遍 | 选中行 / 投影单元 | 来源SHA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 主分析 | 810609 | 0.115523 | 114.20 | 3 | 5808 / 65538 | 6 |
| 独立审计 | 810610 | 0.128694 | 114.91 | 3 | 5808 / 65538 | 6 |

每阶段24022物理行/380100词法单元，选中5808行/65538投影；合计6遍/48044物理行/11616选中行/131076投影/760200词法单元及12次来源SHA。四种保存向量每阶段22616坐标、5654坐标行，未投影数值转换0、parameter列数值转换0；数组/checkpoint/HDF5/RGB或封存读取/字节SHA、模型前向、梯度重算、优化更新及环境调用均0。父JSON完整解析只用于预检metadata，不从其中取向量作新分析。

四新CSV points22/trials132/heads11/strata12共177行/99936字节；全部空分层保留。审计在新进程独立读相同三表，自行重算向量几何、导数资格、余量分类、严格判断及试步贡献，不调用主分析或协议数值helpers作重建。重建内容、输出CSV/JSON与两个访问账本共13080项标量核对通过，且逐坐标恒等式/范数/分支和覆盖检查通过。新CSV与主JSON前后不可变，主pending状态保留，以audit/completion和两个exit0表示完成。

| 产物 | SHA-256 |
| --- | --- |
| [libero-task7-mse-gate-stop-direction-grid-20261109-v1.json](/tmp/libero-task7-mse-gate-stop-direction-grid-20261109-v1.json) | `54a32df9c6525e884628204b9816645783b997c4475a83ddbdf77ca67022e9e1` |
| [independent-audit.json](/tmp/libero-task7-mse-gate-stop-direction-grid-artifacts-20261109-v1/independent-audit.json) | `7ef00b35b5d53a37fbb1d44b03045a23d9b87634b9c8f0f219b52bce409399b2` |
| [completion.json](/tmp/libero-task7-mse-gate-stop-direction-grid-artifacts-20261109-v1/completion.json) | `d8681d689c5d8aabffeeb6825fbb89ea2c8b4fd842f3d012e2e0bf4f90d893d4` |
| [libero-task7-mse-gate-stop-direction-grid-20261109-v1-process.json](/tmp/libero-task7-mse-gate-stop-direction-grid-20261109-v1-process.json) | `696fd9f9dac486434b74753935bca9e15cf91ca23f2ded4bf78cd780a4596dfc` |

## 验证、文件改动与下一步

35项新增运行检查通过：32局部＋2补充ZIP/物理预算guard＋1完整规模合成主/审计。相关检查300 passed / 4 deselected；补充2项已单独通过且CPU覆盖。CPU回归3395 passed / 21 deselected（192个显式文件，69个可选Crafter关联排除，21项已验证完整合成/资源未重复），未运行完整Crafter套件。环境检查、py_compile及git diff --check通过；全规模合成包含非数值权重/未选字段、光滑/hinge/clip及未绑定/角色/坐标/网格/接受标志等反例。未绑定两个payload API在任何来源SHA/数值/目录创建前拒绝，元数据预检峰值约114.72MiB，详见绑定validation证明。

验证证据：`/tmp/libero-task7-mse-gate-direction-grid-{unit,extra-guard,synthetic,focused,cpu}-20261110-v1.{xml,log}`；同前缀cpu-selection/environment/prebinding-proof JSON保留。没有代码返修后重跑研究数据，真实主/审计仅各一遍。工作区留痕排除results及数据/归档，不重哈希无关历史实验。

新增分析入口、投影guard、独立审计、运行测试、执行契约/绑定和本报告，更新README/roadmap/500计划。保留AGENTS.md、.vscode、已有phase-gripper note和其它历史工作；不改已冻结科学方案/配置或实验原产物。

下一步建议**另冻全18头的原网格／只向更小α扩展网格的配对单变量对照**，保持同一辅助500起点、表示、loss、负总梯度方向和原Policy500设置；新网格范围、试步/更新总预算、停止及复现规则先冻结，不能按运行中结果自适应放宽。该对照检验5条有余量短路径的网格阻碍，按已有6零/5短/7完整分层报告；6个边界头的局部方向冲突仍需单独的方向假设，不能预称扩网格可解决。当前未定义或运行新网格，不把线性α尺度当新实验结果，也不同时修改loss/方向或开启正式评价。

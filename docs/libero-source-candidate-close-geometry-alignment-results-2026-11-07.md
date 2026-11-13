# LIBERO来源candidate close、接触代理与物体运动对齐（2026-11-07）

**已完成120条既有前缀、170个close段的只读对齐，独立审计通过。本轮决定保持500 updates、稳定化5步及现有loss，不统一提前/延后close或增加open偏好。** 证据显示至少存在两种不同的未完成过程：有些前缀未出现双侧接触代理，另一些已经有明显高度变化和目标距离缩短，仍未达成原生目标。尚不能由此指定视觉、动作头、覆盖或记忆为唯一根因，也没有证据证明某个新loss能修复它们。

本次属于已知source场景的diagnostic；不恢复[G的未通过准入](libero-formal-G-results-2026-11-06.md)，H未开展。validation、held-out、qualification、reserved及final evaluation无新增payload访问；无新训练、Policy前向、环境、reset、影子计算、checkpoint/初态/HDF5/RGB读取，不更新SPT/Knowledge或注册Module。

## 范围与对齐口径

依据[已审计采集](libero-source-candidate-geometry-capture-results-2026-11-07.md)和[本次只读口径](libero-source-candidate-close-geometry-alignment-plan-2026-11-07.md)，保留task0/1/3/4/7、seed20261106/07/08、A_original/B_fomaml、candidate0..3全部120条/60对。A/B为G现有30个500端点；每条原采集最多150个Policy动作，原生成功可提前停止，不换算成完整300步成功率或未见泛化。

原有359个命令段重算完全一致，170段close、189段open。close段[s,e]对应状态[s−1,e]，pre-state用于事件起点；直接后继负号动作才标记immediate open，不跨0合并。接触代理为保存边界两侧手指组各有dist≤0碰撞记录，不是力、抓稳、掉落或真实释放标签；位置均为grip_site/物体root body原点。

首次代理至close段末、首次代理至整个前缀末分开计算，后者可能跨多个命令段。run级与episode级分别汇总，不能把170个相关命令段当170个独立episode。五张表保存全部连续量、观察分母和删失，不定义“足够近/抓稳/抬起”等阶段阈值。两臂只按同一绝对状态比较；共同截止为min(nA,nB)，成功后的状态不填0或沿用终点。

## 总体事件核对

- 170个close段中，62段（来自60条不同前缀）在段前边界至段末出现双侧代理，108段没有。无代理段包含早期脉冲和后续重试，不能称108条失败。77个close段右删失，观察长度不是完整持续时长。
- 29段的首个代理块在close仍继续时结束；其中20段在close段末又有代理，9段末没有。代理消失/重现不足以标注掉落/重抓，也不支持直接加大夹爪保持惩罚。
- 120条中仍为67条出现过代理、16条没有close、19条在150步前缀内达成原生目标，配对四格不变。18/19条达成目标时末动作仍为close，只有1条末动作是open。因此终止命令的正号不足以识别失败，不能仅凭仍在close就归因为缺open；close命令也不保证两指仍与物体接触。该统计不排除更早出现过open，下面另列完整命令段反例。

## 接触前与物体运动的任务差异

下表各臂均12条。首次多步close距离仅对实际观察到该事件的轨迹计算；高度列为每条前缀的最高相对reset高度再取中位数，目标变化为每条保存终点XY距离减reset距离再取中位数。各自终点可能不同，这张表描述本臂过程，不作为两臂处理效应；同一时间的配对结果另列。

| task | arm | 多步close可见数 | 首多步close前距离中位（cm） | 有代理前缀数 | 前缀最高高度中位（cm） | 终点目标XY变化中位（cm） |
|---|---|---:|---:|---:|---:|---:|
| 0 | A | 11 | 7.986 | 7 | 9.268 | -6.909 |
| 0 | B | 12 | 37.874 | 1 | 0.017 | +0.019 |
| 1 | A | 10 | 4.893 | 12 | 3.605 | -11.504 |
| 1 | B | 12 | 5.112 | 11 | 10.377 | -22.114 |
| 3 | A | 12 | 37.344 | 5 | 1.463 | +0.533 |
| 3 | B | 12 | 5.572 | 11 | 9.103 | -12.933 |
| 4 | A | 8 | 7.951 | 4 | 0.000 | -0.183 |
| 4 | B | 7 | 9.346 | 3 | 0.000 | -0.528 |
| 7 | A | 10 | 7.867 | 8 | 1.914 | +0.261 |
| 7 | B | 9 | 9.413 | 5 | 0.947 | +0.167 |

task0的B臂多步close前距离仍大，12条仅1条有代理，最高高度中位约0.0175cm；task3的B臂则多步close前距离更小，11条有代理，并呈现较大的高度变化和目标距离缩短。task3 B的早期单步脉冲不能与后续多步段混作一次抓取；该反例继续反对统一按首close时刻改loss。

task4/7同样包含反例。task4 B seed20261106初态3最高高度变化16.493cm，保存终点的碗至目标盘XY距离从reset的35.734降到14.595cm，150步仍未达成目标；不能把该任务所有未完成都称为完全没有物体运动。另一方面，task4 seed20261108两臂全部4条均没有close，最高高度变化仅约4.38e−7米，不能将数值尺度的位移命名为有效抬升。

## 同一时间的全部15组配对

每组4个场景对。高度列对各场景在双方共同可见区间内分别取最高高度，再求B−A及组内中位数；XY列使用双方共同截止状态。正负号仅描述连续差值，不是预设方法接受门槛。所有组保留，不挑seed，不做显著性检验；表内近零舍入，原CSV保留完整数值。

| task | seed | 共同区间最高高度B−A中位（cm） | 共同截止目标XY距离B−A中位（cm） |
|---|---|---:|---:|
| 0 | 20261106 | -9.1938 | +6.9277 |
| 0 | 20261107 | -0.0325 | -0.4478 |
| 0 | 20261108 | -9.7038 | +9.9451 |
| 1 | 20261106 | -1.6072 | -1.1237 |
| 1 | 20261107 | +6.5878 | -9.5734 |
| 1 | 20261108 | +4.7395 | -7.8496 |
| 3 | 20261106 | +7.5321 | -15.1010 |
| 3 | 20261107 | +0.3139 | -5.4918 |
| 3 | 20261108 | +14.3557 | -10.3657 |
| 4 | 20261106 | +0.6464 | +0.1500 |
| 4 | 20261107 | +0.0000 | -0.1553 |
| 4 | 20261108 | +0.0000 | -0.3615 |
| 7 | 20261106 | +0.6514 | -0.0902 |
| 7 | 20261107 | -15.5290 | +42.1073 |
| 7 | 20261108 | -0.0250 | +0.0504 |

task0 B的高度中位差在三个seed均为负，但目标距离不是三seed同向；task3 B的两项配对中位差在三个seed均为更高/更近，不能推广为所有场景或正式方法优势。task1、task4和task7保留反向与近零组，当前没有跨全部任务的统一训练修改方向。

## 接触后仍未完成：task7的完整同seed反例

task7 A seed20261107的全部四个初态如下；展示这一完整组用于解释“有物体运动仍未完成”，不是挑选训练对象。高度为前缀最高值，距离为各自保存终点；初态不同且终止时刻不同，不能从四行推断代理消失或open的因果效应。

| 初态 | 保存步数 | 最高高度（cm） | 终点目标XY距离（cm） | 终点目标XYZ距离（cm） | 终点双侧代理 | 终点命令 | 原生目标 |
|---|---:|---:|---:|---:|---|---|---|
| 0 | 127 | 15.382 | 2.727 | 3.344 | 无 | close | 已达成 |
| 1 | 150 | 16.841 | 3.133 | 3.515 | 有 | close | 前缀内未达成 |
| 2 | 150 | 18.092 | 4.846 | 5.034 | 有 | close | 前缀内未达成 |
| 3 | 150 | 17.257 | 3.796 | 4.015 | 有 | close | 前缀内未达成 |

这组记录支持下一步检查目标附近的相对位姿、接触与原生成功谓词要求，不能仅凭“XY距离较小”断言已经完成放置。也不能把三条未达成直接叫“缺open”：同组成功轨迹从第41到127步为一个连续close段，未在close之后发open即达成目标。当前代理不测力或稳定支撑，不能从这四个不同初态认定释放机制。

## 实现、验证和两处执行记录

分析入口为[只读对齐脚本](../experiments/analyze_libero_source_candidate_close_geometry_alignment.py)，复用原保存字段验证器，标准库计算耗时3.166秒；独立[审计器](../experiments/audit_libero_source_candidate_close_geometry_alignment.py)不导入分析实现或数值runtime，完成435364项单元格/汇总核对，约1.71秒。审计有直接Path.open计数及导入/路径guard，确认240份JSON各读取一次、17247个几何边界，0被拒调用。新增28项合成测试，连同相关专项共110项通过（7.78秒），四个新Python文件编译通过；本次只涉及离线分析，没有重复此前1714项CPU回归，不声称全环境pytest通过。

保留两处技术记录，不把失败包装成成功：

1. v1预检误将旧G的`failed_keep_H_sealed`字符串按布尔True检查，在任何sidecar读取前拒绝。原8份绑定文件和失败记录保存在`/tmp/libero-source-candidate-close-geometry-alignment-v1-preflight-rejection-20261107/`。补4项精确G状态/H封存测试后，另建[执行绑定v2](../configs/libero-source-candidate-close-geometry-alignment-binding_v2.json)；[科学口径配置](../configs/libero-source-candidate-close-geometry-alignment-readonly_v1.json)和几何计算定义未变，旧v1绑定保留。
2. v2主计算已写出全部结果后，外围读取计数断言失败：Python3.10 Path.open使用缓存_accessor.open，修改io.open/builtins.open无法据此验证读取次数。因此不声称该外围guard通过，计数值未持久化；进程失败记录单独保留。没有重跑主计算、采集或训练；后续独立审计直接拦截Path.open，完成读取次数和数值复核。主结果保留原始pending-audit状态，通过状态另写审计与completion，不回写原始字节。

两处记录分别见`/tmp/libero-source-candidate-close-geometry-alignment-v1-preflight-rejection-20261107/failure.json`、`/tmp/libero-source-candidate-close-geometry-alignment-process-20261107-v1.json`；成功的独立审计计数见同前缀`audit-process-20261107-v1.json`。分析和审计各一遍原始JSON，完成后整理证据和画图只读新标量表；不包含新的Policy、仿真或模型数值读取。

## 产物与provenance

所有新产物均在新的/tmp路径，拒绝覆盖旧文件；原采集、源码执行绑定、checkpoint和历史结果不变。

| 产物 | SHA-256 |
|---|---|
| 科学口径配置：`configs/libero-source-candidate-close-geometry-alignment-readonly_v1.json` | `4326a8e12f1371d355a76c347d4ea218e4c5d357585f9499be0177c01214dd58` |
| 执行绑定v2：`configs/libero-source-candidate-close-geometry-alignment-binding_v2.json` | `2f30237058ce3c2369411f9dcb50c4dcf62467ea06522777f26b95f13c8df7d6` |
| 主分析：`/tmp/libero-source-candidate-close-geometry-alignment-20261107-v1.json` | `4ef42c89bd0fc7d4a161b6634e4680f49f9c27ffe7c57bfdc04ec4fabe1efc31` |
| 独立审计：`/tmp/libero-source-candidate-close-geometry-alignment-audit-20261107-v1.json` | `49bcd6bffc0108ae14b827ce48a2cf28424107d6b1ccd353aecd42c68d73d308` |
| 全部seed/120条证据汇总：`/tmp/libero-source-candidate-close-geometry-evidence-review-20261107-v1.json` | `dad4f731ce883da7d741155c15a5da475c17f1c70388451e164d977831d3bc38` |

五张表位于`/tmp/libero-source-candidate-close-geometry-alignment-tables-20261107-v1/`：

| 表 | 行数 | SHA-256 |
|---|---:|---|
| close_runs.csv | 170 | `3ecd23474ad05c7ad8e378df40792e3d73b0b8f1278b375d3a7588917aff36c5` |
| episodes.csv | 120 | `b3b54c44df1b83a398e8d1e82ab13f76625bfa5c54ed8acf62360b30e6d41b51` |
| pairs.csv | 60 | `15fd2f9065c88160a3cbdf02c9d04aada163d7758cb067a84db0dfa47db30978` |
| aligned_states.csv | 8967 | `44e70ac55eb48256d212b7ce752065916c36288c806d208fcc50ceb6e1dcd8c6` |
| event_windows.csv | 4675 | `06db60407f0952d1a0b1644415405fbe336ec698aeb496913285f2db82fdfa6e` |

全部15个task×seed格的五页[矢量图PDF](/tmp/libero-source-candidate-close-geometry-alignment-figures-20261107-v1/all_task_seed_alignment.pdf)及5份SVG位于同前缀figures目录；`figures.json`记录每份SHA。图中仅汇总同一绝对状态、共同可见场景，附n(t)页；状态k的命令产生该状态，起始pre-state见事件表。各面板自动纵轴，不同刻度不能直接比振幅，尤其task4后两seed的高度量级约1e−7米；逐时刻中位曲线也不等于逐条最高值的中位数。

最终完整性核对另存`/tmp/libero-source-candidate-close-geometry-alignment-completion-20261107-v1.json`，覆盖本轮开始时1308个保护文件、原355项执行引用、当前v2分析绑定、父结果谱系、表/图SHA及全部资源0新增约束。README、roadmap和500-update计划同步当前结论，原冻结说明与历史报告不回写。

## 训练决定与下一步

本轮不改loss、不增加updates、不统一close/open时机，不用这些已知来源诊断重新接受G。当前证据足以反对把问题归成单一夹爪命令时机，但尚未区分目标近邻的平移/姿态误差、物体支撑/接触条件以及原生成功谓词要求，也没有检验某种训练修改的修复效应。

下一步先只读核对原生目标判定代码与已存终点相对位姿字段，保留全部来源场景，重点列出task7这组及其他已有物体运动但未达成的反例。若现有字段不足，明确缺项，不补猜、不运行环境或读取封存数据。完成核对后，才决定是否另冻单变量训练诊断；继续500 updates、5步稳定化和现有loss。

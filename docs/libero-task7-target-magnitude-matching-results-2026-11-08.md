# task7目标幅值分层与同示教匹配结果（2026-11-08）

**只读匹配完成，独立审计通过13547871项核对；主结论受共同支持域不足限制，本轮不选择训练修改。** 仿射−原输出在all_common的精确匹配小子集中，两种焦点的原输出abs差仍为6/6模型负向，但正纠正每模型只匹配1–4条焦点记录、负新增2–7条，A/B两臂两种焦点都没有三seed共同精确格。不能据此排除总体目标尺度/示教构成影响，或认定原输出较小是失败根因。

## 冻结、实现和实际预算

[科学方案](libero-task7-target-magnitude-matching-plan-2026-11-08.md)、[配置](../configs/libero-task7-target-magnitude-matching-readonly_v1.json)和81项规则测试保持原字节，科学SHA `bb7ddb50c714332ac6e7518e028207f3c0ae0330a34efffd13e7b245cb1ae775`。[执行契约](libero-task7-target-magnitude-matching-execution-contract-2026-11-08.md)、[入口](../experiments/analyze_libero_task7_target_magnitude_matching.py)、[字段guard](../experiments/libero_task7_target_magnitude_matching_access.py)、[独立审计](../experiments/audit_libero_task7_target_magnitude_matching.py)和[51项运行测试](../tests/test_libero_task7_target_magnitude_matching.py)完成，七角色/650项源码依赖绑定后一次运行、真实重试0。

只读取上一轮派生paired-frames.csv的64764行×28列，每阶段一遍；主/审计合计2遍、129528词法行、3626784投影列值，字段、原始标量、精确float32预测/目标bits、三mode一致性、匹配/权重/覆盖、全部表与144份seed摘要独立复现。没有读取或哈希旧oof-frames/curves、NPZ/NPY/头/checkpoint/HDF5、观测/特征/公开本体/私有状态；研究训练、前向、优化器、环境、初态、近邻和封存payload调用均0。

固定task7、seed20261106/07/08、A_original/B_fomaml六个原Policy500与辅助500端点，三比较、两方向、三时间切片、全部四格和30 demo均保留。编码器已见全部source-support示教；这是暴露表示上的OOF辅助读出诊断，不能称未见泛化、正式SPI/SPT或Module结果。保持500/未来稳定化5步/现有loss、四个父joint cue=false、G停止/H未运行及source-query/validation/held-out/qualification/reserved/final封存。

实际总墙钟20.163秒，主阶段9.121秒/58.96MiB，审计10.609秒/92.99MiB。输出CSV 104247975字节；七表合计241521行，其中精确格114837行（上限129528）。所有180秒/512MiB/256MiB及总420秒限制满足，PID、软件、CPU环境、SHA和attempted/completed计数见产物。无环境seed或初态索引。

## 主匹配：关联仍在小子集中，覆盖率不足

主匹配固定同demo、目标方向和完全相同float32目标幅值；对corrected−stays_wrong和new_error−stays_correct分别计算。先在demo内对共同目标格等权，再在共同demo间等权。以下为仿射−原输出/all_common；差均为焦点−固定参照的原输出abs均值，负值表示焦点组较小。目标幅值在精确匹配标准化后差为0，但原始→匹配的变化同时受支持域和权重改变影响。

**负方向新增错误−仍正确**

| 来源/seed | 焦点匹配/全部 | 参照匹配/全部 | 共同demo/格 | 原始帧差 | 共同域帧差 | 格/demo等权差 |
| --- | --- | --- | --- | --- | --- | --- |
| A/20261106 | 7/34 | 8/2145 | 6/7 | -0.439679 | -0.250911 | -0.212062 |
| B/20261106 | 6/34 | 7/2135 | 5/6 | -0.471328 | -0.307095 | -0.287924 |
| A/20261107 | 2/7 | 2/2119 | 2/2 | -0.415320 | -0.195096 | -0.195096 |
| B/20261107 | 4/18 | 4/2131 | 4/4 | -0.409234 | -0.218014 | -0.218014 |
| A/20261108 | 4/26 | 5/2128 | 3/4 | -0.405160 | -0.190979 | -0.180203 |
| B/20261108 | 4/41 | 4/2119 | 4/4 | -0.397677 | -0.195039 | -0.195039 |

**正方向纠正−仍错误**

| 来源/seed | 焦点匹配/全部 | 参照匹配/全部 | 共同demo/格 | 原始帧差 | 共同域帧差 | 格/demo等权差 |
| --- | --- | --- | --- | --- | --- | --- |
| A/20261106 | 2/57 | 4/104 | 2/2 | -0.147392 | -0.134361 | -0.179541 |
| B/20261106 | 3/29 | 2/96 | 2/2 | -0.123955 | -0.144526 | -0.147516 |
| A/20261107 | 1/18 | 1/86 | 1/1 | -0.201359 | -0.130904 | -0.130904 |
| B/20261107 | 1/32 | 1/91 | 1/1 | -0.156786 | -0.145734 | -0.145734 |
| A/20261108 | 4/24 | 3/108 | 3/3 | -0.124447 | -0.049044 | -0.048548 |
| B/20261108 | 3/30 | 3/105 | 3/3 | -0.128080 | -0.066003 | -0.066003 |

负新增焦点覆盖9.76%–28.57%，仍对参照仅0.094%–0.373%；正纠正焦点3.13%–16.67%，仍错参照1.10%–3.85%。负新增共同demo仅2–6，正纠正1–3。表中记录为模型×帧，不当作独立环境样本；同一个专家帧可能在不同seed/比较中重复出现。

负新增原始abs差−0.471328至−0.397677，精确标准化后−0.287924至−0.180203。正纠正标准化后−0.179541至−0.048548，部分模型绝对差比原始更大，不能总结成统一衰减百分比。归一化原输出abs/目标abs的差同样6/6为负，但仍只适用于上述少量共同格。不能把它写成“已经在整个3598帧总体排除了目标尺度”。

三seed共同精确格在A/B两臂、两种all_common焦点均为0，公共支持域标准化值全null；各seed自身小子集的负差不等于在相同条件集合上跨seed复现。末20负新增六模型均无精确匹配；末20正纠正仅A/20261107和B/20261107各一个共同格，其余为null。补集也保留缺项，不用可用seed补成完整三seed均值。

## 辅助分层、其余对照和反例

七个固定层没有并层、改边界或放宽精确匹配。all_common粗层/同demo标准化中，仿射−原输出两种焦点仍是6/6负差；负新增焦点覆盖77.78%–97.06%，正纠正50.00%–72.41%。粗格保留的目标abs差并不为0：负新增各模型−0.035138至−0.006869；正纠正−0.006537至+0.014557。因此粗层不能代替精确控制目标尺度。

粗层三seed共同格只覆盖少量条件：all_common正纠正A3格/3demo、B2格/2demo；负新增A2格/1demo、B3格/3demo。B臂补集负新增的共同粗格上，三个seed原输出abs差分别−0.045600、−0.055425、+0.030826，保留反向seed。更大覆盖率不自动给出跨条件稳定机制。

下表完整保留三比较/两焦点在all_common的可估计模型数，括号为负差模型数。null不补0；三seed原值及时间切片见seed_summary和全部CSV。

| 比较 | 目标方向 | 精确匹配可估计（负差） | 粗层可估计（负差） |
| --- | --- | --- | --- |
| bias_only-minus-original | negative | 3/6（负3） | 4/6（负4） |
| bias_only-minus-original | positive | 2/6（负2） | 4/6（负4） |
| current_affine-minus-original | negative | 6/6（负6） | 6/6（负6） |
| current_affine-minus-original | positive | 6/6（负6） | 6/6（负6） |
| current_affine-minus-bias_only | negative | 6/6（负6） | 6/6（负6） |
| current_affine-minus-bias_only | positive | 6/6（负6） | 6/6（负6） |

## 符号正确不等于连续动作准确

冻结的12指标还显示：负新增精确匹配小子集中，原输出abs/目标abs的组均值为0.233–0.947，仍对参照为1.884–4.269；原Policy相对目标能量的误差在焦点组6/6小于参照组。这里的“仍对”只指符号始终正确，组均值包含较大的同号幅值误差；不能将它直接当作应模仿的动作幅值。该比较只指第3动作分量专家误差，不推断闭环成功。

仿射更新使匹配负新增组的连续误差在六个模型都上升；仍对参照组则3个改善、3个恶化，不能统一叙述成“牺牲小动作换取其它动作全部改善”。A/20261107的两组误差都升高，焦点−参照Δ误差反而为负；差值方向不代表焦点本身已改善。完整focus/control均值和原始/共同域/标准化三个层次都保留。

## 决定与下一步

本轮支持有限的条件关联：在少量完全相同目标/demo的可比格内，新增/纠正焦点的原输出幅值仍较小；更大的粗层集合也有同向关联，但有目标残余不平衡。主匹配的覆盖和三seed公共支持域不足，不能识别总体机制，也不能把“较小原输出”直接变成损失权重或认为它更不准确。

本轮不改训练、loss、预算或checkpoint规则，不放宽匹配，不恢复已停止的辅助候选/G/H。匹配路线先收束为覆盖受限结论；若继续新的受控训练方向，应另冻不同的可拒绝假设，例如“是否能保护原本专家误差较小的动作，同时保留必要纠正”，原Policy/输入/500预算固定，每次只改一个变量，仍需完整source护栏和三seed结果。该方向是下一项假设，不是本轮证明了loss根因或已经选择了新loss；本轮没有冻结或运行新训练。

## 验证和证据

科学冻结阶段81项规则测试、358项协议测试通过；新增51项运行、相关409项、排除可选Crafter后的2520项CPU回归通过，py_compile/git diff --check通过。密集及全部目标唯一的同规模合成流程分别由全新主/审计进程验证；稀疏例精确格恰达129528上限，主匹配输出完整null和覆盖率。没有新实现失败、实际研究运行失败或真实重试。可选Crafter测试依环境限制排除，未将其计为LIBERO失败。

保护执行开始时1426个文件；除README/roadmap/500计划的状态更新外，旧文件、科学配置/元数据/方案/规则测试和642/634/621/612/605/367旧绑定均保持原字节。AGENTS.md、.vscode及无关phase报告保留。最终核对见/tmp/libero-task7-target-magnitude-matching-final-verification-20261108-v1.json。报告只读本轮已审计model/common3派生表及结果JSON，未新增来源paired-frames数据遍历。

- [执行绑定](/home/zl202621/project/KECRL/configs/libero-task7-target-magnitude-matching-execution-binding_v1.json)，SHA `f4e11a6ec0fc5911bb9fbcdeb2db61562a726d6917242714bb070481ccb9ba29`。
- [主pending结果](/tmp/libero-task7-target-magnitude-matching-20261108-v1.json)，SHA `289c7e1844ba854706d3cb0b79b07c334489c0a5241e3f6f69142685f367e6dc`。
- [独立审计](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/independent-audit.json)，SHA `43ba1568e1eaa911ecbea1a9aa9fe68f5edf7e181bfdce7822133f03eda7c6b7`。
- [完成记录](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/completion.json)，SHA `18960562766387848cb35979e94d539784b39f25580a79e3da5a81bf5d71acee`。
- [进程与成本](/tmp/libero-task7-target-magnitude-matching-20261108-v1-process.json)，SHA `5575dfc412dc2ce51c8cee0567484daec410ad3716832e087fbc0d63fabcf452`。
- [派生报告摘要](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/report-summary.json)，SHA `bbd3c854a9cd20e990f64ad8dbbc5473d3aec5f47abfb3fcf3000bcbba8f97e4`。

全部7表固定路径与实际行数/SHA：

- [frame-strata.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/frame-strata.csv)：64764行，SHA `316401c39ca11ad7f932ac05569c6d7807715ad5401296bdaee2122809258390`。
- [bin-transitions.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/bin-transitions.csv)：3024行，SHA `987dafb2e6a4a5c79ee4bbe0ecd1e0afaed834d9d602f74386489b4f677eb672`。
- [coarse-bin-cells.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/coarse-bin-cells.csv)：45360行，SHA `e4ffe88c63dcc6f99b273a90a46fcc22f8308e12c615e73f0291457d03e86bfd`。
- [demo-matched.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/demo-matched.csv)：12960行，SHA `0b50cb277a1ed2e89dc11d80dd45fe9f28b1189a8b3fc75671068aeffe2f2777`。
- [model-matched.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/model-matched.csv)：432行，SHA `eb242197f969f7c7dee5f4157023dc79d3372c16c5cac6051eec5f45b76826e7`。
- [common3-summary.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/common3-summary.csv)：144行，SHA `be4eb4f844f15fa2fdabe64cdae56aee60976014fe893256d19232c977a56370`。
- [exact-target-cells.csv](/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1/exact-target-cells.csv)：114837行，SHA `c58faec8ceb2f9c27162020a84d7769d78bb51e26f777803be5472378a91aecc`。

# task7残差错号转移与幅值分解结果（2026-11-08）

**只读分解完成，独立审计通过6300492项核对；没有追加研究训练。** 当前仿射头在六个模型、正负两个方向都同时产生纠正与新增错误，净计数会掩盖相反转移。焦点错误的原输出幅值较小，但负方向新增错误的专家目标幅值也更小；具体纠正帧没有三seed稳定交集。原四个联合cue=false和候选停止决定保持，不能据此恢复G/H或确定视觉、历史、表示或loss中的唯一根因。

## 设置、实现与预算

[冻结方案](libero-task7-residual-sign-transition-plan-2026-11-08.md)、[科学配置](../configs/libero-task7-residual-sign-transition-readonly_v1.json)、[执行契约](libero-task7-residual-sign-transition-execution-contract-2026-11-08.md)保持原字节和科学SHA `7de0233952c2913c50b59071d7af79706dc26477f4db635d1bd6606a7aac0c0a`。实现[主入口](../experiments/analyze_libero_task7_residual_sign_transition.py)、[字段guard](../experiments/libero_task7_residual_sign_transition_access.py)、[独立审计](../experiments/audit_libero_task7_residual_sign_transition.py)及[72项运行测试](../tests/test_libero_task7_residual_sign_transition.py)，七角色/642项依赖绑定后真实执行一次、重试0。

task7、seed20261106/07/08、A_original/B_fomaml全部六个原Policy500与辅助500端点；A/B是特征来源，不是早期baseline/jitter。本次原输出、bias_only、current_affine三个读出均保留。编码器已见全部30条source-support示教，OOF只针对辅助头的整demo三折，不是未见编码器泛化或正式结果。action index2不解释为世界z、夹爪或物理阶段，末20是示教时间切片。

每阶段frames75384行仅17列；curves4608行仅9个ID，500/probe576行再读14指标。0/100/250点及fit指标均未数值投影。主/审计合计4次来源CSV遍历、159984词法行、2662128投影列值；各8064项probe摘要复现。3598共享material帧：负2273/正1325、末20负402/正75；三比较产生64764配对记录。12张输出表共115728行、72份三seed汇总全部核对，空格和30个示教均保留。审计读取新生成12表用于核对，来源4遍与输出核对分别记账。

实际总墙钟32.186秒，主活动7.899秒/峰值72.73MiB，审计23.901秒/85.45MiB；输出CSV 63112272字节。各阶段≤180秒/512MiB、总≤420秒/256MiB，CPU/Python3.10.22/NumPy1.22.4、CUDA隐藏及单线程环境匹配。PID、时间、attempted/completed行/投影和输入前后SHA见结果/审计/process。没有环境seed或初态索引。

原NPZ/NPY/头/checkpoint/HDF5/RGB/特征/公开本体或私有状态读取及字节哈希均0，研究训练、原Policy/新读出前向、优化器、环境/近邻调用均0。source-query、validation、held-out、qualification、reserved和final payload未访问；500/未来稳定化5步/现有loss保持。

## 全部seed的净变化与相反转移

以下每格为“纠正数 / 新增数 / 净错误变化”；净变化=新增−纠正。负/正分母分别2273/1325，不把seed×帧或A/B来源重复当作独立样本。

**bias_only-minus-original，共同t≥5评分集**

| 来源/seed | 负方向：纠正 / 新增 / 净变化 | 正方向：纠正 / 新增 / 净变化 |
| --- | --- | --- |
| A/20261106 | 0 / 10 / +10 | 40 / 0 / -40 |
| B/20261106 | 0 / 21 / +21 | 29 / 0 / -29 |
| A/20261107 | 11 / 0 / -11 | 0 / 15 / +15 |
| B/20261107 | 0 / 13 / +13 | 5 / 0 / -5 |
| A/20261108 | 6 / 0 / -6 | 0 / 7 / +7 |
| B/20261108 | 1 / 2 / +1 | 6 / 2 / -4 |

**current_affine-minus-original，共同t≥5评分集**

| 来源/seed | 负方向：纠正 / 新增 / 净变化 | 正方向：纠正 / 新增 / 净变化 |
| --- | --- | --- |
| A/20261106 | 2 / 34 / +32 | 57 / 7 / -50 |
| B/20261106 | 13 / 34 / +21 | 29 / 19 / -10 |
| A/20261107 | 37 / 7 / -30 | 18 / 35 / +17 |
| B/20261107 | 16 / 18 / +2 | 32 / 20 / -12 |
| A/20261108 | 25 / 26 / +1 | 24 / 22 / -2 |
| B/20261108 | 25 / 41 / +16 | 30 / 23 / -7 |

**current_affine-minus-bias_only，共同t≥5评分集**

| 来源/seed | 负方向：纠正 / 新增 / 净变化 | 正方向：纠正 / 新增 / 净变化 |
| --- | --- | --- |
| A/20261106 | 6 / 28 / +22 | 26 / 16 / -10 |
| B/20261106 | 26 / 26 / +0 | 15 / 34 / +19 |
| A/20261107 | 26 / 7 / -19 | 28 / 30 / +2 |
| B/20261107 | 26 / 15 / -11 | 32 / 25 / -7 |
| A/20261108 | 21 / 28 / +7 | 30 / 21 / -9 |
| B/20261108 | 26 / 41 / +15 | 24 / 21 / -3 |

例如B/20261106的仿射−偏置在负方向净变化0，内部却是26纠正、26新增；不能把净不变解释成动作或错误集合没变。仿射−原输出正方向5/6净改善、负方向5/6净恶化，同时保留A/20261107正方向104→121、负方向147→117的相反seed。

**仿射−原输出，末20帧**（负/正分母402/75）

| 来源/seed | 负方向：纠正 / 新增 / 净变化 | 正方向：纠正 / 新增 / 净变化 |
| --- | --- | --- |
| A/20261106 | 0 / 2 / +2 | 8 / 3 / -5 |
| B/20261106 | 12 / 0 / -12 | 0 / 10 / +10 |
| A/20261107 | 0 / 6 / +6 | 18 / 0 / -18 |
| B/20261107 | 0 / 15 / +15 | 32 / 0 / -32 |
| A/20261108 | 0 / 6 / +6 | 4 / 0 / -4 |
| B/20261108 | 0 / 13 / +13 | 8 / 0 / -8 |

B/20261106末20负方向20→8、正方向51→61；B/20261107负方向0→15、正方向75→43。这些反例保持，不用更好看的seed、方向或末窗结果代替完整护栏。

## 输出距0、残差与目标尺度

下表固定仿射−原输出、all_common，分别检查正方向corrected与负方向new_error。距0阈值0.025沿用冻结方案，不扫描阈值；“较小”是相对固定正确性参照组的描述，不能将所有帧称为近零。

| 来源/seed | 正纠正n | 原输出abs中位数 | abs<.025 | 负新增n | 原输出abs中位数 | abs<.025 |
| --- | --- | --- | --- | --- | --- | --- |
| A/20261106 | 57 | 0.024681 | 29/57 | 34 | 0.056452 | 6/34 |
| B/20261106 | 29 | 0.030473 | 12/29 | 34 | 0.043559 | 7/34 |
| A/20261107 | 18 | 0.009036 | 14/18 | 7 | 0.032576 | 3/7 |
| B/20261107 | 32 | 0.091034 | 0/32 | 18 | 0.051457 | 5/18 |
| A/20261108 | 24 | 0.007891 | 22/24 | 26 | 0.031137 | 9/26 |
| B/20261108 | 30 | 0.025914 | 14/30 | 41 | 0.055584 | 9/41 |

正纠正原输出abs均值0.011664–0.088199，仍错参照组0.136111–0.244985；负新增0.026774–0.069228，仍对参照组0.440770–0.517147。共同评分集两种原输出abs组间差在六模型全部为负，同示教两组均非空后等权的差也6/6为负；正纠正对应共同demo数6–15、负新增5–17。该组间关联不是因果效应。

不能把末窗缺项填补成共同模式：A/20261108末窗正纠正与仍错没有共同demo，B/20261106末窗没有正纠正或负新增，相关标准化差为null。补集中A/B的20261107均无正纠正，保留null；所有30个demo和三个seed仍在输出中，完整均值仅三个seed均可用才定义。

本批正纠正共190条模型×帧记录，其中91条原输出abs<0.025；负新增共160条，其中39条<0.025、121条≥0.025，均无恰0原输出或候选输出。因此“全部只是临界翻号”不受支持；这两个计数集合不是350个独立环境样本。

两类焦点记录的平均raw_shift均为正：正纠正各模型0.055953–0.176297、负新增0.055428–0.158868；其它帧的残差方向仍随seed变化。有效位移/原输出abs的中位数分别2.004–6.757与1.582–3.114。越过0与该比值有关联约束，不能把“比值过1”另当独立机制证据。两类焦点均无clip标记，raw/effective gap绝对值最大6.8271e−9，与保存float32舍入一致；本批焦点翻号不是clip标记或舍入伪差所解释。所有原预测/残差/clip精确重建已通过。

**负方向目标尺度仍是重要混杂。** 负新增的专家abs均值0.082653–0.155515，仍对参照组0.507350–0.511327；同demo等权目标幅值差也6/6为负。原输出较小同时伴随专家要求的动作较小，本轮没有在相同目标幅值内匹配，不能确定原输出距0的关联是否独立于目标尺度。正纠正相对仍错的目标幅值差和同demo差方向混合，保留六个模型原值；不能用一条全局“小动作”解释两种焦点。

## MSE、错号和跨seed稳定性

正纠正的190条模型×帧中189条连续误差下降、1条上升；负新增160条中158条误差上升、2条下降。符号变化和MSE方向没有逻辑等价性，合成测试也覆盖这两类反例。

A/20261108末窗正方向纠正4条、新增0条，错号71→67，但MSE 0.075946→0.085451。四格ΔSSE依次为：仍对−0.020675、纠正−0.007816、新增0、仍错+0.741387；主要误差增长在67条仍错记录，不能把它全部归给那4条纠正。

下表为同帧三seed票数；any/all3只统计指定transition，另列共同参考/候选错误，不把共同新增少等同于没有共同错误。

| 来源 | 焦点 | 至少一seed | 三seed全共同 | 原输出三seed共同错 | 候选三seed共同错 |
| --- | --- | --- | --- | --- | --- |
| A_original | positive/corrected | 84 | 0 | 76 | 90 |
| A_original | negative/new_error | 54 | 0 | 73 | 98 |
| B_fomaml | positive/corrected | 78 | 0 | 73 | 81 |
| B_fomaml | negative/new_error | 64 | 1 | 69 | 96 |

两臂没有三seed共同纠正的正方向帧；原输出已有76/73条三seed共同正错，候选仍有90/81条共同正错。负新增的共同三seed帧为A0/B1，原输出共同负错为73/69。分组尺度关联可在三个seed同向，同时具体修正帧并不一致；净总数与共同错误集合描述不同，二者都保留。

常数偏置与仿射的交集也不一致：A/20261106正纠正两者共同31条、仿射独有26条、偏置独有9条；B/20261106为14/15/15；A/20261107和08的偏置没有正纠正，仿射各18/24条。负新增存在仿射独有7–40条/模型。完整4×4交集在head-overlap.csv；两个独立训练头的差不能拆成同一仿射头的Wz和自身bias。

## 支持、反证与下一步

支持的有限结论：当前500端点的残差能改善部分专家动作，但同时带来方向性副作用；净变化掩盖大量相反转移。焦点转移与较小原输出幅值相关，该关系在all_common的同demo参照下仍存在。实现/clip/float32复现通过，本批焦点没有数值重建故障。

反证与未解决问题：多数负新增原输出超过0.025；目标幅值在负新增组也更小；具体正纠正帧无三seed共同交集，原共同错误仍保留；末窗和MSE存在相反seed与不变符号误差增长。此处“更小”均为描述值比较，不作显著性或因果结论。不能据此唯一归因视觉/缺少历史/动作头容量/loss或闭环分布偏移，也不能推广到闭环成功、source-query或正式评价。

保持四个父joint cue=false、候选停止及500/5步/现有loss，不部署辅助头、不追加训练或选择250点。**下一步建议另冻仅用本轮派生CSV的目标幅值分层与同demo匹配对照**：在同一目标方向、目标幅值层和原本正确性内，检查原输出距0与新增/纠正的关联是否仍存在；先分离目标尺度混杂，再决定是否值得提出单变量训练候选。阈值、匹配/缺项、字段与预算须先冻结，本轮不执行该新增分层。

## 验证与证据

72项新增运行、349项相关测试和2388项CPU回归通过，py_compile及git diff --check通过。完整同规模合成主/审计用例在专项、相关和两次CPU集合中重复验证；本轮实际研究分解一次、无真实失败或重试。首次扩大回归遗漏test_spatial_gather_spt.py的Crafter依赖，保留2399通过/18缺crafter失败/4跳过日志；补全排除后2388全部通过。未安装可选Crafter或修改其测试，不将环境依赖限制称为LIBERO代码失败。

旧1415工作区文件中除README/roadmap/500计划状态更新外保持原字节；原AGENTS.md、.vscode及无关phase报告保留，旧634/621/612/605/367代码绑定未改。最终文件/产物完整性核对见/tmp/libero-task7-residual-sign-transition-final-verification-20261108-v1.json。源CSV在真实两阶段之外只做字节SHA校验；报告只读本轮已审计派生表，未新增来源CSV数据遍历或数组/模型访问。

- [执行绑定](/home/zl202621/project/KECRL/configs/libero-task7-residual-sign-transition-execution-binding_v1.json)，SHA `7804869d9df1c025e7ae1ff1a3da15f968468431141f51d16c1dbef885ceaa95`。
- [主pending结果](/tmp/libero-task7-residual-sign-transition-20261108-v1.json)，SHA `f702c18e6a1f61a66d88d53805dfcba95c5dacf1267b05c8d0b40974f4a11184`。
- [独立审计](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/independent-audit.json)，SHA `6fa7968780569c827a09f15dfd4c1db1c20fa30f6a0919761195ad652fe91bff`。
- [完成记录](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/completion.json)，SHA `56b83176cc33656611312f0a34f352ea793e9b2a96863f4567be6f8b528d4845`。
- [进程与墙钟](/tmp/libero-task7-residual-sign-transition-20261108-v1-process.json)，SHA `c338cb6e2010e2817b4226f51f6c4aff08bccba81e13c8db2609691756a56592`。
- [派生报告摘要](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/report-summary.json)，SHA `83f9795ceadf030bca8250fe01a8726f9ecc5394bcce16512dbed7c7d64185de`。

全部12表固定路径与SHA如下；不将/tmp产物复制到仓库：

- [paired-frames.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/paired-frames.csv)：64764行，SHA `7003a9cbb404abbfd4be1212b603419a4e1751b4c4614f3cf70c0015a95a5ef0`。
- [model-transitions.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/model-transitions.csv)：432行，SHA `238c25ddcd7333b9e0a2d142cd88271248460f5f2e327ee5c046f62efd4c389f`。
- [demo-transitions.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/demo-transitions.csv)：12960行，SHA `5f6af91935a98a1d512f28822315b5cbfd4c3c891a5f6340544798eb88420968`。
- [margin-cells.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/margin-cells.csv)：3888行，SHA `470797bd435bac4ca5a37cb14a69d85d6b627c16cdf25384aef90dc434a813cb`。
- [error-cells.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/error-cells.csv)：3888行，SHA `a0c6948910ed187ade344c8b817e6140112366b1135027653fc610377d8d6847`。
- [contrasts.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/contrasts.csv)：216行，SHA `f4e19b39abebcd7449953945c97eacdda23492098d7971519fe3d5a5331ea230`。
- [demo-contrasts.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/demo-contrasts.csv)：6480行，SHA `7bd89b7680c4008093c6e0df004a1837ddb89a601c67f5a25561d6309a40b97e`。
- [demo-standardized.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/demo-standardized.csv)：216行，SHA `6ab93832b83ea2158c366a8e7ebe1b72b2d9c0e2c1bf04804e57acc1a8422a94`。
- [crossseed-frames.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/crossseed-frames.csv)：21588行，SHA `8e66a73b84ae40ee93257ac668147f542781b2f33675fb2f424331d95890db1e`。
- [crossseed-summary.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/crossseed-summary.csv)：144行，SHA `4969a7955a951f6081c3653e4567dbedb727d9ab75acd7012d5697a1ea04f3e3`。
- [head-overlap.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/head-overlap.csv)：576行，SHA `40deb8a8264f8be6c7185522011af8cad56aafe8b37d6f7293413b3e44a920d3`。
- [probe-parity.csv](/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/probe-parity.csv)：576行，SHA `2a5150eeda899075e527d2d228543a14554c8befa6b9664939a446dc93c85784`。

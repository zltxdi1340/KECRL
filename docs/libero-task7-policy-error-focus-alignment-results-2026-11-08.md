# task7原Policy连续动作误差与固定焦点对齐结果（2026-11-08）

**已按冻结方案执行，独立审计通过。** 共同评分集上，负方向NN“新增错误”焦点的原Policy第3分量MSE与错号率高于固定参照，六模型在原始帧加权及共同demo等权下均同向；正方向NN“纠正”焦点则低于其固定参照。但末20步的负焦点有5/6模型不含Policy错号，正焦点仍有大量错号。说明存在离线局部关联，不能把NN标签变化当作Policy动作改善或损害，也不能直接决定加历史、改loss或认定闭环根因。

保持六个500-update端点、未来稳定化5步和现有均匀七维masked MSE；G失败/H未运行不变。source meta_query、validation、held-out、qualification、reserved和final evaluation payload未访问；无新训练、CNN/Policy前向、环境、checkpoint或近邻扫描，不更新SPT/Knowledge或注册Module。

## 范围与执行证据

[科学协议](libero-task7-policy-error-focus-alignment-plan-2026-11-08.md)、[科学配置](../configs/libero-task7-policy-error-focus-alignment-readonly_v1.json)、[执行契约](libero-task7-policy-error-focus-alignment-execution-contract-2026-11-08.md)固定task7、seed20261106/07/08、A_original/B_fomaml六个现有端点。A/B为G原始/候选初始化适应臂，不是历史baseline/jitter。编码器已见全部30条source meta_support示教，本项是training-fit diagnostic，不是未见测试或正式迁移结果。

新增实现：[入口](../experiments/analyze_libero_task7_policy_error_focus_alignment.py)、[字段/CSV投影guard](../experiments/libero_task7_policy_error_focus_alignment_access.py)、[独立审计](../experiments/audit_libero_task7_policy_error_focus_alignment.py)及[运行测试](../tests/test_libero_task7_policy_error_focus_alignment.py)。

每模型4188个t≥1原始动作、共同t≥5且abs(target[2])≥0.025的3598帧（负2273/正1325），末20步负402/正75。第3分量为action index2，不作物理世界z解释；tail20只表示时间窗。每帧只有同一份原Policy七维输出与专家目标，CC/HC/HH仅为已保存分组。

| 项目 | 实际值 |
|---|---:|
| 主分析＋独立审计NPZ加载 | 7＋7＝14 |
| 字段解码 | 11＋11＝22 |
| 标签CSV数据遍历 | 2＋2＝4 |
| 每阶段投影行／列值 | 43176／518112 |
| 每阶段原始预测行／平方误差坐标 | 25128／175896 |
| 每阶段共享目标能量坐标 | 29316 |
| 每阶段legacy标量复现 | 216 |
| 九CSV总行数 | 38990 |
| 独立逐字段核对 | 2244820 |
| 总wall时间 | 8.995秒 |
| 主／审计active时间 | 3.179／4.614秒 |
| 主／审计峰值RSS | 104.91／103.86MiB |

CPU Python3.10.22/NumPy1.22.4、CUDA隐藏、单线程；运行命令、PID、UTC、退出码及无技术重试见[process](/tmp/libero-task7-policy-error-focus-alignment-20261108-v1-process.json)。科学配置、七角色、621项依赖先绑定后解码。输入7个NPZ/2个标签CSV的SHA阶段前后不变；主pending-audit JSON原字节保留，audit/completion另存。没有复用旧读取额度。

## 共同评分集：确有误差关联，方向不同

负焦点=NN new_error，参照=stays_correct；正焦点=corrected，参照=stays_wrong，匹配目标符号及CC原本对错。Δ均为“焦点−固定参照”的同一原Policy误差差，不是训练前后变化。下表错号为wrong_sign或预测零；实际material帧预测零为0。demo Δ只在同demo两组均非空时定义，不替代全帧Δ。

| seed / arm | 负焦/参错号 | 负ΔMSE | 负demo ΔMSE | 正焦/参错号 | 正ΔMSE | 正demo ΔMSE |
|---|---|---:|---:|---|---:|---:|
| 20261106 / A_original | 1/41；6/2092 | 0.055344 | 0.024085 | 32/58；68/89 | -0.039025 | -0.011568 |
| 20261106 / B_fomaml | 5/37；10/2092 | 0.058074 | 0.017318 | 29/57；66/88 | -0.031968 | -0.019050 |
| 20261107 / A_original | 5/38；38/2104 | 0.053190 | 0.014654 | 18/45；54/83 | -0.035044 | -0.037050 |
| 20261107 / B_fomaml | 4/38；21/2106 | 0.077466 | 0.056014 | 18/38；54/74 | -0.041404 | -0.022937 |
| 20261108 / A_original | 4/28；22/2100 | 0.035293 | 0.006207 | 17/31；61/79 | -0.053919 | -0.028177 |
| 20261108 / B_fomaml | 2/28；23/2108 | 0.078386 | 0.032360 | 16/40；54/82 | -0.060184 | -0.025463 |

六模型的负焦点目标平均绝对幅值均小于参照，MSE、SSE/目标能量、错号率却更高；共同demo等权后六模型上述误差差仍为正。这支持同帧原Policy动作问题的局部关联，不能归为“只是目标幅值大”或仅用不同demo构成解释。负焦点同时同帧其它平移/夹爪误差也有差异；这没有单独分离第3分量、视觉、动作头或历史机制。正焦点在共同评分集六模型的MSE与错号率较参照低，仍非Policy更新后的改善。

三个时间层完整三seed均值如下；高/低为ΔMSE方向数/3，不是显著性或准入。括号为三seed实际共同demo数，不能将同帧重复模型当独立样本。

| arm | 时间层 | 方向 | 原始ΔMSE均值（高/低） | demo等权ΔMSE均值（高/低） | 共同demo数 |
|---|---|---|---:|---:|---|
| A_original | all_common | negative | 0.047943（3/0） | 0.014982（3/0） | 17/15/10 |
| A_original | all_common | positive | -0.042663（0/3） | -0.025598（0/3） | 17/13/10 |
| A_original | tail20 | negative | -0.001623（2/1） | -0.010254（1/2） | 11/10/6 |
| A_original | tail20 | positive | 0.001388（1/2） | -0.031893（0/3） | 4/4/4 |
| A_original | complement | negative | 0.099451（3/0） | 0.040692（3/0） | 10/11/6 |
| A_original | complement | positive | -0.068430（0/3） | -0.016726（1/2） | 15/7/7 |
| B_fomaml | all_common | negative | 0.071309（3/0） | 0.035231（3/0） | 15/12/12 |
| B_fomaml | all_common | positive | -0.044519（0/3） | -0.022483（0/3） | 15/10/11 |
| B_fomaml | tail20 | negative | 0.009449（2/1） | 0.009302（2/1） | 10/7/6 |
| B_fomaml | tail20 | positive | 0.011194（2/1） | -0.050349（0/3） | 5/2/1 |
| B_fomaml | complement | negative | 0.121469（3/0） | 0.075524（3/0） | 8/8/7 |
| B_fomaml | complement | positive | -0.059953（0/3） | 0.005136（2/1） | 10/5/6 |

补集的负焦点六模型原始及demo等权MSE、错号率差均为正；末20不能照搬该判断。补集正焦点原始MSE六模型较低，但demo等权仅A2/3、B1/3较低，保留构成反例，不能给全部时段同一解释。

## 末20步：错号、幅值及示教构成分开

| seed / arm | 正纠正组Policy错号/帧 | 正固定参照错号/帧 | 负新增组Policy错号/帧 | 负新增组同号不足/过大 | 正原始ΔMSE | 正demo ΔMSE（共同demo数） |
|---|---|---|---|---|---:|---:|
| 20261106 / A_original | 11/12 | 34/37 | 0/23 | 18/5 | -0.049516 | -0.017656（4） |
| 20261106 / B_fomaml | 8/16 | 28/32 | 3/19 | 10/6 | -0.056435 | -0.029008（5） |
| 20261107 / A_original | 16/17 | 27/27 | 0/21 | 6/15 | 0.057728 | -0.067324（4） |
| 20261107 / B_fomaml | 11/11 | 30/30 | 0/22 | 14/8 | 0.084785 | -0.091391（2） |
| 20261108 / A_original | 10/12 | 28/28 | 0/13 | 12/1 | -0.004049 | -0.010699（4） |
| 20261108 / B_fomaml | 9/9 | 24/24 | 0/11 | 5/6 | 0.005231 | -0.030647（1） |

负焦点5/6无Policy错号，只有B/20261106为3/19，参照7/361；其余负焦点仍有同号幅值误差，不能写成“没有动作问题”。六模型负焦点ΔMSE原始4/6较高、demo等权3/6较高，不支持末窗统一恶化。正焦点原始ΔMSE的方向不一致；共同demo等权六模型均较低，且仅1–5个共同demo，不能据少量配对把较低误差泛化成稳定收益。

整个75个正目标末帧（不是仅焦点）情况如下。wrong_sign的预测绝对幅值≥0.025分开计数，避免把全部问题解释成临界翻号；本轮没有阈值扫描。

| seed / arm | 正目标错号/75 | 错号输出幅值<0.025 / ≥0.025 | 错号占第3分量SSE | 第3分量bias | MSE/目标能量均值 |
|---|---|---|---:|---:|---:|
| 20261106 / A_original | 62/75 | 8/54 | 98.72% | -0.258584 | 3.874 |
| 20261106 / B_fomaml | 51/75 | 10/41 | 95.47% | -0.216318 | 2.985 |
| 20261107 / A_original | 66/75 | 14/52 | 98.73% | -0.292587 | 4.837 |
| 20261107 / B_fomaml | 75/75 | 0/75 | 100.00% | -0.361279 | 6.621 |
| 20261108 / A_original | 71/75 | 7/64 | 99.00% | -0.248359 | 3.222 |
| 20261108 / B_fomaml | 75/75 | 0/75 | 100.00% | -0.282312 | 3.812 |

正目标末窗有51–75/75帧错号，错号占第3分量SSE的95.47%–100%，多数反向输出已有≥0.025幅度；原Policy负偏置是实际离线输出问题。固定参照同样大量错号，不能把这些错误全部归为NN纠正焦点特异问题。专家动作非唯一成功控制，此结论仍不证明其造成闭环失败。

## 中间抵消组与跨seed

次焦点CC=HH=−1、HC=+1，与CC=HH=HC=−1参照比较：负目标端点都对，正目标端点都错。末窗负次焦点第3分量原始MSE差只有3/6为正、demo等权1/6为正；同帧夹爪MSE差在原始和demo等权下却都是6/6为正（原始+0.425505至+0.675257）。这提示分组还伴随其它控制分量/上下文差异，不能按NN方向标签单独调第3分量。正次焦点原始MSE差2/6为正，demo等权1/6为正；其夹爪原始差6/6为正、demo等权4/6为正，全部反例保留在CSV。

| arm | 时间层/方向 | 三seed共同Policy错号帧 | 三seed共同焦点成员帧 | 三seed成员且错号帧 |
|---|---|---:|---:|---:|
| A_original | all_common/negative | 73/2273 | 5 | 0 |
| A_original | all_common/positive | 76/1325 | 7 | 1 |
| B_fomaml | all_common/negative | 69/2273 | 4 | 0 |
| B_fomaml | all_common/positive | 73/1325 | 8 | 1 |
| A_original | tail20/negative | 1/402 | 1 | 0 |
| A_original | tail20/positive | 60/75 | 1 | 1 |
| B_fomaml | tail20/negative | 0/402 | 0 | 0 |
| B_fomaml | tail20/positive | 51/75 | 0 | 0 |

末窗正目标共同Policy错号A60/B51帧，NN共同正纠正成员A1/B0；不能因为“共同成员且错号少”就宣称Policy缺乏跨seed共同错误。反过来，共同Policy错误多也不证明NN成员一致或历史输入必需。

## 验证与修正记录

新增运行测试47项；连同元数据及父协议共137项重点通过，排除可选Crafter项的CPU回归2073通过，py_compile和git diff --check通过。CPU环境检查见绑定中的日志。未运行依赖可选Crafter的全量测试；不是LIBERO代码失败。完整合成流水线使用与实际相同的资源限制，并在写出主结果后禁用主数值函数，独立审计仍通过。

绑定前保留两次测试失败及修正：首次76通过/1失败，原因是等权平均在零附近的求和顺序改变精确方向标签，统一每轴float64列和与math.fsum汇总后通过；首次CPU回归2065通过/8失败，为访问单测共享Torch进程RSS超过512MiB，隔离单测资源桩后通过。完整流水线始终在新进程中执行真实上限。两项均发生在真实payload之前，不修改科学方案、512MiB预算或旧缓存；本轮实际研究执行一次，重试0。

旧612/605/367代码绑定均不变。执行开始保护1393文件；除README、roadmap及500计划的状态同步外，已有文件、无关phase报告、AGENTS.md/.vscode和历史实验均保持。科学冻结四文件不回写。[最终完整性核验](/tmp/libero-task7-policy-error-focus-alignment-final-verification-20261108-v1.json)记录文件/来源/绑定及实际预算。

## 文件与provenance

- [执行绑定](/home/zl202621/project/KECRL/configs/libero-task7-policy-error-focus-alignment-execution-binding_v1.json)，SHA-256 `df3f7dc79676b47d4c73e76dab60ab17962f4c8f6c915171566db21c6bd4c8e2`。
- [主结果（pending-audit原字节）](/tmp/libero-task7-policy-error-focus-alignment-20261108-v1.json)，SHA-256 `186b49379f9c9d2aa7b5ee5a903a095e8ccaa7a1e3f93ae4faa0476447cc78a1`。
- [独立审计](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/independent-audit.json)，SHA-256 `ce94a8d1d45a647b53c15e82a07ca1bfbdc922ce223fedfc67713d8275886a14`。
- [完成记录](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/completion.json)，SHA-256 `f9208f5b5aae55b7f27d7577b9060e694c486a4f1ec5bc86e1d451a8df9c5ccb`。
- [派生报告汇总（0原始读取）](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/report-summary.json)，SHA-256 `01a7e19a1bf7ab69c1cf65162439b8b8b58fac7a1a594cd890e8c7019ea070b0`。

九表均在同一artifacts目录，供完整seed/demo/空组反例复核：

- [frames.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/frames.csv)：21588行；SHA `f4713f6d8416f744e4c79089356731652c7a5140ad9a6aefdd4041ae03d45308`。
- [model-groups.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/model-groups.csv)：252行；SHA `3d3dc66a21a25a04bcef88584186a0e1686b60d2f46491c4fc07738b95647e91`。
- [demo-groups.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/demo-groups.csv)：7560行；SHA `c935299459319aca6a94363c07dcd149e6d53aefb422925ad3746410169c0d59`。
- [contrasts.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/contrasts.csv)：72行；SHA `c99ba42f68d43fe6306d4ae6283f81f4758120bef24ea2a9083116a139449b04`。
- [demo-contrasts.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/demo-contrasts.csv)：2160行；SHA `ebfb967f23c836283d6715aa2c56f2ccd28d308e54e0ef3af186f642d4332eeb`。
- [demo-standardized.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/demo-standardized.csv)：72行；SHA `aa77b0c3b8cda81028ebc9f6a69d6fc9c33d3852acfcd06fcf321af6b6999a68`。
- [crossseed-frames.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/crossseed-frames.csv)：7196行；SHA `bace9186db978fa6dd69336719d7a9b4df323f9d1505a6463276561a791e520a`。
- [crossseed-summary.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/crossseed-summary.csv)：72行；SHA `1197688a3613cd0c876742e08e927fa5a375c8aa6229c8f5cf9fa37b86a207a4`。
- [legacy-parity.csv](/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1/legacy-parity.csv)：18行；SHA `083002771b57404280883b7d211a16dd785861e0efdd6378296a1bc5e86e9b28`。

## 决定与下一步

收束近邻扫描，继续保持500/5步/现有loss及当前G停止，不立即统一加历史或做方向加权。已确认的线索是原Policy正方向末窗负偏置，以及共同评分集的焦点/参照误差关联；尚未分离误差是否能被现有表示的连续动作读出修正，也未证明闭环因果。

下一步另冻一个有界的“固定当前表示→连续第3分量/残差读出”可纠正性对照：固定六端点，source-support内部按完整demo隔离fit/probe，原Policy作固定参照，同时检查正方向误差和负方向退化，保留末窗及补集。先明确新读出训练预算、参数/初始化对照、数据和停止规则，再实现/绑定执行；不沿用已消费14次额度，不访问source-query或正式评价，不在本轮追加训练。该对照检验可读出性，不单独认定action head根因或接受新Policy。

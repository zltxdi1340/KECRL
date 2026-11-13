# task7固定当前表示连续动作可纠正性结果（2026-11-08）

**训练、字段/角色guard和独立审计均完成；四个最终联合cue全部false。** 36个辅助头各500，共18000次更新和backward，完整固定预算一次执行，无真实重试。相对原输出，current_affine在共同评分集正方向MSE与错号数5/6模型改善，但负方向错号数5/6模型增加；末窗负错号也5/6增加。常数偏置及当前仿射残差均未满足三seed、双方向及时间层护栏，按冻结规则结束本候选，不追加训练、改loss、选择其它曲线点或推进闭环。

这只评价已暴露source-support表示上的辅助读出。不能推出当前表示充分/不足、原action head根因、历史必要或闭环收益；未改变原Policy500端点、未来稳定化5步、原loss，G失败/H未运行及source-query/正式评价payload封存保持。未更新SPT/Knowledge或注册Module。

## 冻结设置与实现

[科学协议](libero-task7-current-residual-correctability-plan-2026-11-08.md)、[配置](../configs/libero-task7-current-residual-correctability-diagnostic_v1.json)、[执行契约](libero-task7-current-residual-correctability-execution-contract-2026-11-08.md)固定task7、三seed20261106/07/08、A_original/B_fomaml六个现有500端点。A/B是原G特征来源，不是本次三读出或早期baseline/jitter。原输出作固定参照；bias_only仅学1个bias，current_affine学256权重+bias，零残差初始化，只修正action index2，其余六维逐位复制原输出。参数自由度不同，不能据差值作独立模块因果归因。

新增实现：[训练入口](../experiments/run_libero_task7_current_residual_correctability.py)、[字段/角色guard](../experiments/libero_task7_current_residual_correctability_access.py)、[独立审计](../experiments/audit_libero_task7_current_residual_correctability.py)、[运行测试](../tests/test_libero_task7_current_residual_correctability.py)。科学冻结四文件原字节不变，七角色/634项源码与本地Torch依赖先绑定后读取。

source meta_support demos0..29按完整demo三折0..9/10..19/20..29轮作probe，另20条fit；每帧只作一次新读出probe。编码器和原Policy已见全部30条support，因此不是编码器未见测试。fit-only scaler、9张配对PCG64采样表、500步Adam/均匀七维masked MSE均冻结；只允许新头梯度，全部36头完成500后才回溯0/100/250/500快照评分。读取当前表示/原动作缓存，不载入checkpoint/HDF5/RGB/公开proprio/私有状态或旧NN标签，不调用原CNN/Policy或环境。

OOF共同t≥5共4068共享帧，其中负material2273/正1325，末20负402/正75；neutral470共同帧保留在总指标。第3分量不是物理世界z，末20不是物理接近/释放标签。所有aux输出均为clip(p0[2]+残差,−1,1)，没有方向/phase/gripper加权。

## 实际预算、数值及资源

| 项目 | 实际值 |
|---|---:|
| 辅助头／每头updates | 36／500 |
| optimizer／backward／训练前向 | 各18000 |
| 训练行访问 | 1152000 |
| fit-only scaler／fit行 | 18／50256 |
| 唯一采样表 | 9（A/B及两头共用） |
| 主／审计快照读出 | 144／144批，各603072行 |
| 源NPZ加载／成员 | 14／34 |
| 新头审计加载／成员 | 36／684 |
| 总NPZ加载／成员 | 50／718 |
| 输入标签CSV遍历 | 0 |
| 每阶段legacy复现 | 216标量 |
| 七CSV行数／新头归档 | 151434／36份19字段 |
| 独立核对 | 13769914 |
| 总wall时间 | 72.211秒 |
| 主／审计active | 28.673／41.966秒 |
| 主／审计峰值RSS | 496.48／116.16MiB |
| CSV／头NPZ总字节 | 127419942／23153688 |
| 真实技术重试 | 0 |

CPU Python3.10.22/NumPy1.22.4/Torch2.4.1+cu124，CUDA隐藏、单线程、确定性；180秒/512MiB阶段上限、420秒总上限及磁盘额度均满足。命令/PID/UTC/退出码见process记录，环境seed/初态不适用。七源文件前后SHA不变；主pending-audit JSON保持原字节，audit/completion另存。

审计独立重建144批，保存float32输出的最大差0、全部快照的符号分歧0，六个非目标分量逐位相同；18组最终数值稳定性门均通过。最终cue失败来自动作误差/退化条件，而非数值重建。主文件的provisional数值门固定pending false，不能用来判断结果；本报告只采用audit/completion中的最终cue。审计核对状态/角色及预测，不独立重跑18000步优化。

## 500端点：正方向改善不能补救负方向退化

下表错号为material目标上的严格反号或预测零。顺序固定为original → bias_only → current_affine；共同正分母1325、负2273，每seed相同，不跨seed或来源臂合并为独立样本。

| seed / 来源 | 共同正MSE（原→偏置→仿射） | 共同正错号数 | 共同负MSE（原→偏置→仿射） | 共同负错号数 |
|---|---|---|---|---|
| 20261106 / A_original | 0.039293 → 0.035219 → 0.034117 | 161 → 121 → 111 | 0.033649 → 0.035404 → 0.035677 | 94 → 104 → 126 |
| 20261106 / B_fomaml | 0.035531 → 0.032812 → 0.034396 | 125 → 96 → 115 | 0.036453 → 0.037466 → 0.036571 | 104 → 125 → 125 |
| 20261107 / A_original | 0.032652 → 0.034793 → 0.033803 | 104 → 119 → 121 | 0.046680 → 0.043446 → 0.038727 | 147 → 136 → 117 |
| 20261107 / B_fomaml | 0.036724 → 0.034865 → 0.034083 | 123 → 118 → 111 | 0.037494 → 0.039072 → 0.035515 | 124 → 137 → 126 |
| 20261108 / A_original | 0.036130 → 0.036879 → 0.032093 | 132 → 139 → 130 | 0.041568 → 0.040958 → 0.039879 | 119 → 113 → 120 |
| 20261108 / B_fomaml | 0.038491 → 0.038296 → 0.032679 | 135 → 131 → 128 | 0.039332 → 0.039596 → 0.039668 | 113 → 114 → 129 |

current_affine相对original：共同正方向MSE/错号数均5/6改善；共同负MSE3/6改善，但负错号数仅1/6改善、其余5/6增加。负方向均值MSE下降也不能补救逐seed错号护栏。bias_only的共同正方向两项4/6改善，另2模型更差；常数偏置不能统一修正本问题。

| seed / 来源 | 末窗正MSE（原→偏置→仿射） | 正错号/75 | 末窗负MSE（原→偏置→仿射） | 负错号/402 |
|---|---|---|---|---|
| 20261106 / A_original | 0.091329 → 0.078873 → 0.081253 | 62 → 54 → 57 | 0.035223 → 0.038629 → 0.037103 | 10 → 13 → 12 |
| 20261106 / B_fomaml | 0.070371 → 0.060052 → 0.079948 | 51 → 42 → 61 | 0.040989 → 0.046230 → 0.038918 | 20 → 31 → 8 |
| 20261107 / A_original | 0.114026 → 0.128328 → 0.097554 | 66 → 73 → 48 | 0.036268 → 0.036030 → 0.039980 | 7 → 7 → 13 |
| 20261107 / B_fomaml | 0.156064 → 0.146418 → 0.092520 | 75 → 75 → 43 | 0.037366 → 0.036909 → 0.038255 | 0 → 0 → 15 |
| 20261108 / A_original | 0.075946 → 0.078412 → 0.085451 | 71 → 72 → 67 | 0.044856 → 0.044196 → 0.040852 | 2 → 2 → 8 |
| 20261108 / B_fomaml | 0.089864 → 0.089308 → 0.080123 | 75 → 75 → 67 | 0.041802 → 0.042259 → 0.044201 | 0 → 0 → 13 |

末窗仿射相对原输出：正MSE4/6、正错号5/6改善；负MSE仅2/6改善，负错号5/6增加。B/seed06保留明显反例：负错号20→8变好，正错号51→61变差；B/seed07正错号75→43变好，负错号0→15新增。A/seed08还保留“正错号71→67减少但MSE0.075946→0.085451升高”的错号/幅值取舍，不能只看一个指标。

三个时间层的完整三seed均值另列；这些均值不覆盖上方seed反例，也不替换冻结cue。每格原→偏置→仿射：

| 来源 | 时间层 | 方向 | 第3分量MSE均值 | 三seed各自错号数（原；偏置；仿射） |
|---|---|---|---|---|
| A_original | all_common | positive | 0.036025 → 0.035630 → 0.033338 | 161/104/132；121/119/139；111/121/130 |
| A_original | all_common | negative | 0.040632 → 0.039936 → 0.038095 | 94/147/119；104/136/113；126/117/120 |
| A_original | tail20 | positive | 0.093767 → 0.095204 → 0.088086 | 62/66/71；54/73/72；57/48/67 |
| A_original | tail20 | negative | 0.038782 → 0.039618 → 0.039312 | 10/7/2；13/7/2；12/13/8 |
| A_original | complement | positive | 0.032560 → 0.032056 → 0.030053 | 99/38/61；67/46/67；54/73/63 |
| A_original | complement | negative | 0.041030 → 0.040004 → 0.037833 | 84/140/117；91/129/111；114/104/112 |
| B_fomaml | all_common | positive | 0.036915 → 0.035325 → 0.033719 | 125/123/135；96/118/131；115/111/128 |
| B_fomaml | all_common | negative | 0.037760 → 0.038712 → 0.037251 | 104/124/113；125/137/114；125/126/129 |
| B_fomaml | tail20 | positive | 0.105433 → 0.098593 → 0.084197 | 51/75/75；42/75/75；61/43/67 |
| B_fomaml | tail20 | negative | 0.040052 → 0.041799 → 0.040458 | 20/0/0；31/0/0；8/15/13 |
| B_fomaml | complement | positive | 0.032804 → 0.031529 → 0.030691 | 74/48/60；54/43/56；54/68/61 |
| B_fomaml | complement | negative | 0.037267 → 0.038048 → 0.036562 | 84/124/113；94/137/114；117/111/116 |

## 曲线、示教构成和全部停止条件

四个曲线点均在所有fit完成后回溯评分，未按probe曲线选预算。下表all_source七维MSE在同来源内按实际帧数汇总三seed×三折；fit行每源帧出现两次，probe每帧一次，所以这些是不同角色的描述量，不是独立重复。

| 来源 | 读出 | 角色 | 0 | 100 | 250 | 500 |
|---|---|---|---:|---:|---:|---:|
| A_original | bias_only | fit | 0.039804 | 0.039748 | 0.039739 | 0.039738 |
| A_original | bias_only | probe | 0.039804 | 0.039750 | 0.039745 | 0.039746 |
| A_original | current_affine | fit | 0.039804 | 0.039411 | 0.039208 | 0.039190 |
| A_original | current_affine | probe | 0.039804 | 0.039530 | 0.039384 | 0.039341 |
| B_fomaml | bias_only | fit | 0.038981 | 0.038938 | 0.038931 | 0.038930 |
| B_fomaml | bias_only | probe | 0.038981 | 0.038943 | 0.038940 | 0.038940 |
| B_fomaml | current_affine | fit | 0.038981 | 0.038330 | 0.038211 | 0.038229 |
| B_fomaml | current_affine | probe | 0.038981 | 0.038461 | 0.038390 | 0.038429 |

偏置头18/18个fit折的全源MSE较初始低，但probe只有13/18低；仿射头fit与probe均16/18低。学习目标下降不等于每折/每方向均改进，也不证明预算普遍不足或已收敛。B仿射250步均值低于500仍保留为曲线事实，主端点始终500，不选择250。其它六维保持不变，七维MSE差=第3分量MSE差/7，平移差=该差/3，不能称旋转、夹爪或完整控制能力改善。

每组全部30个demo及空组、demo等权差和方向计数见oof-demo-groups/demo-contrasts/contrasts；主判据固定合并OOF帧权重，不因等权demo或均值更好而替换护栏。全部18个固定比较最终均不满足完整条件：

| seed / 来源 | 比较 | 未通过条件 |
|---|---|---|
| 20261106 / A_original | bias_only − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/negative/component3_mse，complement/negative/material_sign_error_rate |
| 20261106 / A_original | current_affine − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/negative/component3_mse，complement/negative/material_sign_error_rate，all_common/all/all7_mse |
| 20261106 / A_original | current_affine − bias_only | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/positive/component3_mse，tail20/positive/material_sign_error_rate，complement/negative/component3_mse，complement/negative/material_sign_error_rate，all_common/all/all7_mse |
| 20261106 / B_fomaml | bias_only − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/negative/component3_mse，complement/negative/material_sign_error_rate |
| 20261106 / B_fomaml | current_affine − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/positive/component3_mse，tail20/positive/material_sign_error_rate，complement/negative/component3_mse，complement/negative/material_sign_error_rate，all_common/all/all7_mse |
| 20261106 / B_fomaml | current_affine − bias_only | all_common/positive/component3_mse，all_common/positive/material_sign_error_rate，tail20/positive/component3_mse，tail20/positive/material_sign_error_rate，complement/positive/component3_mse，complement/negative/component3_mse，complement/negative/material_sign_error_rate，all_common/all/all7_mse |
| 20261107 / A_original | bias_only − original | all_common/positive/component3_mse，all_common/positive/material_sign_error_rate，tail20/positive/component3_mse，tail20/positive/material_sign_error_rate，complement/positive/component3_mse，complement/positive/material_sign_error_rate |
| 20261107 / A_original | current_affine − original | all_common/positive/component3_mse，all_common/positive/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/component3_mse，complement/positive/material_sign_error_rate |
| 20261107 / A_original | current_affine − bias_only | all_common/positive/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/component3_mse，complement/positive/material_sign_error_rate |
| 20261107 / B_fomaml | bias_only − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/positive/material_sign_error_rate，complement/negative/component3_mse，complement/negative/material_sign_error_rate，all_common/all/all7_mse |
| 20261107 / B_fomaml | current_affine − original | all_common/negative/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/component3_mse，complement/positive/material_sign_error_rate |
| 20261107 / B_fomaml | current_affine − bias_only | tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/component3_mse，complement/positive/material_sign_error_rate |
| 20261108 / A_original | bias_only − original | all_common/positive/component3_mse，all_common/positive/material_sign_error_rate，tail20/positive/component3_mse，tail20/positive/material_sign_error_rate，complement/positive/component3_mse，complement/positive/material_sign_error_rate |
| 20261108 / A_original | current_affine − original | all_common/negative/material_sign_error_rate，tail20/positive/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/material_sign_error_rate |
| 20261108 / A_original | current_affine − bias_only | all_common/negative/material_sign_error_rate，tail20/positive/component3_mse，tail20/negative/material_sign_error_rate，complement/negative/material_sign_error_rate |
| 20261108 / B_fomaml | bias_only − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/positive/material_sign_error_rate，tail20/negative/component3_mse，complement/negative/component3_mse，complement/negative/material_sign_error_rate，all_common/all/all7_mse |
| 20261108 / B_fomaml | current_affine − original | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/material_sign_error_rate，complement/negative/material_sign_error_rate |
| 20261108 / B_fomaml | current_affine − bias_only | all_common/negative/component3_mse，all_common/negative/material_sign_error_rate，tail20/negative/component3_mse，tail20/negative/material_sign_error_rate，complement/positive/material_sign_error_rate，complement/negative/material_sign_error_rate |

A/B来源各自bias_only与current_affine_extra三seed联合cue全部false，每组均0/3个seed满足完整条件。这里的extra要求同时胜过原输出和偏置参照，不能挑某个seed/某折/某方向推进。

![共同评分集OOF曲线：线为三seed均值，带为实际范围，不是置信区间](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/oof-diagnostic-curves.png)

该图只展示共同集合、不是末窗或闭环成功率。源码和完整逐seed/时间层指标以绑定与CSV为准；[PDF](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/oof-diagnostic-curves.pdf)可独立导出。

## 代码验证、绑定前修正和证据

新增运行测试39项；与元数据/父协议共212项重点、排除可选Crafter项的CPU回归2195项通过，py_compile和git diff --check通过。未执行依赖可选Crafter的完整集合，不将依赖限制记作LIBERO失败。合成端到端与实际一样采用真实180秒/512MiB限制，并以全新、无Torch/无主训练数值模块的审计进程复算。

三次绑定前合成内存失败均保留：默认Dynamo包装导入过大；大矩阵临时副本/CPU heap；完整指标字典缓存导致一次峰值537841664字节，超过536870912。修正为原Torch Adam eager数值内核、原地/流式数学读出、紧凑对照缓存并归还未使用heap页，VmHWM不重置。500步标准Adam与eager路径参数及m/v逐位一致。科学配置/预算/旧缓存未改，三次仅合成数据，真实执行一次、重试0。具体日志/源代码SHA见[执行契约](libero-task7-current-residual-correctability-execution-contract-2026-11-08.md)及绑定。

保护原1404文件；除README/roadmap/500计划的状态更新外，旧文件及621/612/605/367代码绑定均保持，AGENTS.md、.vscode和原无关phase报告保留。完整性及成本核验见[/tmp final-verification](/tmp/libero-task7-current-residual-correctability-final-verification-20261108-v1.json)。

- [执行绑定](/home/zl202621/project/KECRL/configs/libero-task7-current-residual-correctability-execution-binding_v1.json)，SHA-256 `634889184ab14dcea00fbf7f104b7e0026e90dc9b5db3c13c6ea51130fd3c18c`。
- [主pending结果](/tmp/libero-task7-current-residual-correctability-20261108-v1.json)，SHA-256 `3dede2f27f67135c2ac6a54e1402b79869b330c562f7bc10b35aab0e7ffc5973`。
- [独立审计及最终cue](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/independent-audit.json)，SHA-256 `c932f93555f518b194732e3ac2052456454a120a5334007aa1d51841b13b09c1`。
- [完成记录](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/completion.json)，SHA-256 `042a426495aebc53496bf9ebcb2d4e3500222c650a1caa5eb0f247f7eb9affd0`。
- [派生CSV报告摘要](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/report-summary.json)，SHA-256 `c680fd1abc6c2a309bac4e33005ee66aeae35d31d81364f6bfd9951bccb9dba0`。

七表完整保存，所有产物留在/tmp，新checkpoint或数据不入仓库：

- [curves.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/curves.csv)：4608行，SHA `42175f71dc314f95ad9b1642c535bf39e60e803c8094c0680b272a91f5e8203e`。
- [oof-groups.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/oof-groups.csv)：1152行，SHA `d61a398eac6043ed157f4b73b5b61115f7eefc594b668010e549198831107b7c`。
- [oof-demo-groups.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/oof-demo-groups.csv)：34560行，SHA `7344b0a8e3fb7bec90ef341802119574f206955652c08f898f213a351014e07e`。
- [contrasts.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/contrasts.csv)：1152行，SHA `ee7bcb5593e52f2234b1b1044c2c4aea7753999cb06515d41b758e93f0589c05`。
- [demo-contrasts.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/demo-contrasts.csv)：34560行，SHA `8da9c8b869e07504da269237450a8469cd2cf6dfcd957dc017921ee5a43518d3`。
- [oof-frames.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/oof-frames.csv)：75384行，SHA `5c4cb891cc24c90f3edba7b781b4389f8cc63f73ae61bffa060ec1b87f2eec00`。
- [legacy-parity.csv](/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1/legacy-parity.csv)：18行，SHA `083002771b57404280883b7d211a16dd785861e0efdd6378296a1bc5e86e9b28`。

## 决定与下一步

结束这项500-update辅助读出候选，不追加训练、延长预算、选250点、改loss或启动闭环。当前结果支持“可降低部分专家动作误差，但未形成跨seed/方向稳定纠正”的有限判断；不能把失败唯一归因于CNN、动作头或缺少历史，也不自动更改G/H正式协议。

下一步建议仅对本轮已审计oof-frames/curves CSV另冻一次只读错号转移和残差幅值分解，核对正方向纠正与负方向新增错误是否集中于相同源帧、是否跨seed重复、是否接近0；保留全部示教和末窗/补集。只使用本次输出，不重读原NPZ或增加拟合，再据证据决定是否值得提出单变量训练候选。source-query及正式评价继续封存。

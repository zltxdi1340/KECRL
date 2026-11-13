# task7 五个 fine20 新停止点：只读方向与末轮20次拒绝核对（2026-11-10）

**五个头均仍有严格正的MSE余量，但当前d=−G在每个新停止点都降低T、提高专家误差MSE；末轮100次全部违反固定起点MSE上界，其中73次仅被MSE门否决，27次同时不满足Armijo。** 最小α=2^-19的五个真实试步均满足Armijo，却超出MSE上界。这把本轮有限网格停止定位到已很小的误差余量和当前方向的取舍；仍有正余量，不能称为“边界上完全无可行步”或全局不可优化。

原Policy500、未来稳定化5步、固定表示和现有loss保持。此项只分析上轮选定的short5全部新停止点及各自辅助500参照，属于fit机制诊断。0新训练/Policy或CNN前向/梯度计算/方向搜索/环境调用；无NPZ/CSV源文件、头/checkpoint、HDF5/RGB或正式评价访问/SHA，无SPT/Knowledge更新或Module注册。没有使用这些结果选择checkpoint。

## 来源、冻结与口径

[方案兼执行契约](libero-task7-fine-stop-direction-review-plan-2026-11-10.md)、[科学配置](../configs/libero-task7-fine-stop-direction-review-readonly_v1.json)、[执行绑定](../configs/libero-task7-fine-stop-direction-review-execution-binding_v1.json)。承接[coarse12/fine20已审计结果](libero-task7-mse-guard-finer-grid-results-2026-11-10.md)，固定A/06/f1 k7、B/06/f0 k7、A/07/f1 k8、A/08/f0 k16、B/08/f0 k10；seed完整为20261106/07/08，A=A_original，B=B_fomaml。

只使用上轮结果JSON（SHA `006090f077ea86a55bb254b3d86f760ac2e2433b4eaa3777eddc1fae13d78c26`）的白名单投影。每阶段10点/2570梯度坐标行/100个末轮试步，分别260/28270/2100列值，合计30630；主/审计共61260投影列值。10280个已存梯度/方向坐标每阶段各复核一次，没有新梯度求值。

元数据发现为核对字段名完整解析过父JSON一次，发生于冻结前，未计算新方向/试步分解；执行另按冻结预算主/审计各完整解析一次并前后SHA，共2解析/4SHA。JSON解析物理解码了未选记录和parameters字段；这些值被排除在数值投影外，不能宣称它们未解码。报告不再次读取或哈希父JSON/源NPZ/CSV，以下数字仅来自新派生JSON。

科学SHA `15939ffb2abe778dd7ad834059252d0f669d264d12656e57ec22b656fc78dff7`；774项依赖在本轮方向/试步分析前绑定。Python3.10.22/CPU/stdlib-only，CUDA隐藏、PYTHONHASHSEED0、BLAS三项单线程；没有新随机采样，环境seed为null、初态为空。13个其余头仅作为18头分母和角色元数据核对，不是本轮梯度对照。

## 误差余量与梯度方向

S=M500−Mstop，M500是本头辅助500的专家MSE上界，区别于惩罚中的原Policy逐帧误差e0。Gb为MSE3/7的梯度，Gp为P梯度，G=Gb+Gp；d=−G。十个起止点均无clip/hinge精确tie，因此本轮sM=7Gb·d、sT=缓存精确右导数具有可解释性；复核sT≈G·d、范数与坐标守恒。cosine表示Gb与Gp的方向夹角。

| seed末两位 | 来源 | fold | 停止k | S余量 | 500起点sM | 停止sM | 停止sT | Gb/Gp cosine | S/sM线性尺度 | 尺度/2^-19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06 | A_original | 1 | 7 | 3.016235278e-08 | -0.04394975785 | 0.01868437077 | -0.01247067028 | -0.7915836974 | 1.614309261e-06 | 0.8463629741 |
| 06 | B_fomaml | 0 | 7 | 7.571023343e-09 | -0.003750095763 | 0.005670238771 | -0.0005186516893 | -0.9710612295 | 1.335221258e-06 | 0.7000404827 |
| 07 | A_original | 1 | 8 | 9.693181396e-10 | -0.01263900328 | 0.001101429808 | -0.0001089896009 | -0.9752142614 | 8.800543916e-07 | 0.4614019569 |
| 08 | A_original | 0 | 16 | 5.188646179e-10 | -0.02096337155 | 0.000479910387 | -2.141005644e-05 | -0.9946740317 | 1.081169802e-06 | 0.566844353 |
| 08 | B_fomaml | 0 | 10 | 1.75349308e-09 | -0.0331926808 | 0.003205976324 | -0.0003301317334 | -0.9643460019 | 5.469451121e-07 | 0.2867567589 |

五条路径的起点sM均负，停止点均正；停止点sT仍严格负。原方向可以降低复合目标，但此时其MSE分量在局部上升，下降由惩罚项贡献支撑。Gb/Gp cosine约−0.792至−0.995，与方向取舍一致；这不是原Policy闭环失败的唯一归因，也不说明改变loss必然有效。

S约5.19e-10至3.02e-8，均严格为正且高于描述性1e-10带。因此五个点的冻结推论都是`positive_slack_local_feasible_by_continuity`：连续数学目标上存在足够小α保留余量并下降T。S/sM约5.47e-7至1.61e-6，是fine最小步的0.287–0.846倍；仅为一阶尺度，不能据此指定安全α，更没有实际评价更小的新步长。有限20级全部拒绝与连续小步存在性可以同时成立。

## 末轮20次拒绝：逐头保留全部原因

| seed末两位 | 来源 | fold | 试步数 | Armijo通过 | MSE通过 | 仅MSE否决 | 两条件失败 | 仅Armijo失败 | 最小MSE超界 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06 | A_original | 1 | 20 | 12 | 0 | 12 | 8 | 0 | 5.483993024e-09 |
| 06 | B_fomaml | 0 | 20 | 15 | 0 | 15 | 5 | 0 | 3.244177413e-09 |
| 07 | A_original | 1 | 20 | 15 | 0 | 15 | 5 | 0 | 1.131504695e-09 |
| 08 | A_original | 0 | 20 | 15 | 0 | 15 | 5 | 0 | 3.96495288e-10 |
| 08 | B_fomaml | 0 | 20 | 16 | 0 | 16 | 4 | 0 | 4.36146199e-09 |

100/100的MSE门都失败，73/100的Armijo条件成立；不能把全部拒绝都描述成没有目标下降。27条较大试步同时不满足Armijo，保留这些反证。各头最小超界均来自j19，仍为严格正数（最小约3.96e-10），没有把数值容差加到接受上界。

下表是已执行过的最小试步j19，α=1/524288≈1.90735e-6；不是本轮新试步。Δ均相对新停止点，超界则相对辅助500。ΔM/7+ΔP=ΔT，Armijo margin为bound−Ttrial。

| seed末两位 | 来源 | fold | ΔM | 超上界 | ΔP | ΔT | Armijo margin | ΔM−αsM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06 | A_original | 1 | 3.564634581e-08 | 5.483993024e-09 | -2.88764298e-08 | -2.378409468e-08 | 2.378171609e-08 | 8.736758591e-12 |
| 06 | B_fomaml | 0 | 1.081520076e-08 | 3.244177413e-09 | -2.534262279e-09 | -9.892335995e-10 | 9.891346743e-10 | 7.85883542e-14 |
| 07 | A_original | 1 | 2.100822835e-09 | 1.131504695e-09 | -5.079963228e-10 | -2.07878775e-10 | 2.078579869e-10 | 1.219655517e-14 |
| 08 | A_original | 0 | 9.153599059e-10 | 3.96495288e-10 | -1.716014794e-10 | -4.083577834e-11 | 4.08316948e-11 | 3.48549139e-15 |
| 08 | B_fomaml | 0 | 6.11495507e-09 | 4.36146199e-09 | -1.503233675e-09 | -6.296686644e-10 | 6.296056966e-10 | 4.051133428e-14 |

j19五条均有正Armijo margin、负ΔT和正超界；其惩罚下降超过MSE3/7上升。误差一阶残差约3.49e-15至8.74e-12，在这个已观测小步上与局部正sM相符；不能把该一致性外推成未试α的安全保证。全部j0..19的幅值、割线、预测/残差和原因在新trials.csv与结果JSON逐条保留，未只挑j19作为总体证据。

## 独立核对、预算与访问记录

| 项目 | 主分析 | 独立审计 | 冻结上限/合计 |
| --- | --- | --- | --- |
| 父JSON解析/SHA | 1 / 2 | 1 / 2 | 合计2 / 4 |
| 端点/坐标行/试步 | 10 / 2570 / 100 | 10 / 2570 / 100 | 各10 / 2570 / 100 |
| 投影列值 | 30630 | 30630 | 合计61260 |
| 新四表行数 | 121 | 121 | points10/trials100/heads5/strata6 |
| 源NPZ/CSV/模型/封存读取及SHA | 0 | 0 | 0 |
| 新训练/前向/梯度/环境调用 | 0 | 0 | 0 |
| 活跃秒 | 0.07097316626 | 0.07653153781 | 120 |
| 峰值RSS MiB | 116.0976562 | 118.015625 | 256 |

独立审计完成8745项核对，未导入主数值实现。它从相同字段白名单独立重建点积/范数、clip/hinge条件、推论、20级严格条件与损失分账，数值绝对1e-10、相对0，身份/计数/接受和原因精确。源JSON前后SHA保持；新四表/结果在审计前后保持。两进程exit0，总watchdog墙时13.306485秒（上限300），包含元数据/代码绑定校验与进程启动；纯数值活跃时间另列，不能把墙时全部当训练。

41项新增检查通过，覆盖完整两进程合成、clip/hinge/临界数值、严格门小于1e-10的超界拒绝、20级/字段/方向篡改及无关parameters毒值隔离；163项相邻协议/执行检查通过，1个已通过的大合成不重复。共204项通过，无共享训练代码变化，未重复完整Crafter/重型优化套件。CUDA隐藏环境检查、Python3.10编译和git diff --check通过。

开发时首轮测试脚本生成有一处嵌套引号语法错误，文件未生成导致测试未收集；未开始真实分析。修正后新增41项和相邻163项通过，首轮日志保留 `/tmp/libero-task7-fine-stop-review-unit-20261110-v1.log`，通过版为v2。真实分析无失败、重试或覆盖。绑定前1418个原有非results文件全部未变；最终只新增本项9文件并在三处进度文档前置结果，用户AGENTS/.vscode和既有phase文档保持。

## provenance与文件

主结果保留pending状态，完成由独立audit/completion及process两进程exit0共同证明；派生数据留/tmp，没有改写上轮产物或复制checkpoint。

| 项目 | 路径 | SHA-256 |
| --- | --- | --- |
| 科学配置 | /home/zl202621/project/KECRL/configs/libero-task7-fine-stop-direction-review-readonly_v1.json | 15939ffb2abe778dd7ad834059252d0f669d264d12656e57ec22b656fc78dff7 |
| 执行binding | /home/zl202621/project/KECRL/configs/libero-task7-fine-stop-direction-review-execution-binding_v1.json | e0479dae36f82e4a20fe478ba14a2466a4d574d19908160cabae99686b6124a0 |
| 结果 | /tmp/libero-task7-fine-stop-direction-review-20261110-v1.json | e5c5b3fcfed0051c9a55dbfd1d53adad958ff15c93ce1e0c037dfaf35007431a |
| 独立审计 | /tmp/libero-task7-fine-stop-direction-review-artifacts-20261110-v1/independent-audit.json | 01bbf04dc0be4f6d42f6eb41d998c0604dc1dba9fe6b45f1c6c6aa5e36eda5fe |
| completion | /tmp/libero-task7-fine-stop-direction-review-artifacts-20261110-v1/completion.json | d87c7017ff38e3a806251a8b6147c310afff58fb27c95c2fe8f7d0d3e25c5b18 |
| process | /tmp/libero-task7-fine-stop-direction-review-20261110-v1-process.json | 1745282b6d9edfa7278e6e5f461c059974829f3b74e00661b975665549841eb9 |

代码：[分析入口](../experiments/analyze_libero_task7_fine_stop_direction_review.py)、[投影guard](../experiments/libero_task7_fine_stop_direction_review_access.py)、[独立审计](../experiments/audit_libero_task7_fine_stop_direction_review.py)、[元数据/判据](../experiments/libero_task7_fine_stop_direction_review_protocol.py)、[测试](../tests/test_libero_task7_fine_stop_direction_review.py)。测试、环境、绑定前证明及初始快照引用见binding.validation。

## 结论边界与下一步

支持证据：五条都是正余量/正sM/负sT，100次MSE全否决，73次仍有Armijo下降；最小实际步长的正超界和小一阶残差相互对齐。反证与限制：27次同时没有Armijo下降；余量并非精确0；尚未评估其它方向或新α，只有已选择的5个fit头，无query/闭环证据，不能推全局不可行或单一根因。

据此收束本轮网格检查。下一项建议先另冻“同一表示、同一loss与固定MSE上界下，改变下降方向能否兼顾目标下降和MSE不增”的单变量假设，写明候选构造、零方向/夹点处理、配对预算、严格接受和停止规则，再决定是否执行辅助优化对照。当前不继续试新α、不重训原Policy或自动放行probe/闭环；原Policy500、稳定化5步、现有loss与正式角色封存保持。

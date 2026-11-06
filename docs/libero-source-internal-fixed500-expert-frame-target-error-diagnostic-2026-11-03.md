# 固定500端点逐帧专家动作误差诊断（2026-11-03）

**预算已冻结并实际完成819个主前向＋18个重复批次。** source专家输入上的错误集中在close/open切换附近，open附近最突出；早期t=1..20没有出现夹爪符号错误。该模式在A/B/C都存在，不能据此解释为FOMAML特有缺陷，也不能把专家输入上的热点直接当闭环失败根因或恢复已停止的局部loss分支。

## 冻结、执行与数据

先核对当前工作区、[前序0/500汇总分解](libero-source-internal-initialization-endpoint-target-error-diagnostic-2026-11-03.md)和原数据/代码；只读取60个source-support组的shape/dtype并核对两个文件SHA。随后冻结[科学配置](../configs/libero-source-internal-fixed500-expert-frame-target-error-diagnostic_v1.json)、[协议](libero-source-internal-fixed500-expert-frame-target-error-protocol-2026-11-03.md)，通过61项专项后绑定[执行代码](../configs/libero-source-internal-fixed500-expert-frame-execution-binding_v1.json)。冻结时Policy/数值payload调用均0，记录保存在`/tmp/libero-source-internal-fixed500-expert-frame-freeze-verification-20261103-v1.json`，SHA `f54ec5a89788ec654ec1e80408cc38ff50aa3a7a6c260dfbe48ab8067d8736d1`。

[执行入口](../experiments/run_libero_source_internal_fixed500_expert_frame_diagnostic.py)随后在空闲物理GPU1运行，GPU0的其他作业未动。固定task0/1、seed20261022/23/24、A_original/B_fomaml/C_pooled_bc全部18个500端点，无模型选择或新增适应。输入是source-support demos0..29、`obs[t-1] -> action[t]`、t>=1，CNN-only、sequence10/stride10/batch8、无增强。task0为2895有效步/304窗口/每模型38批，task1为4059步/422窗口/53批。

| 预算口径 | 声明 | 实际 |
|---|---:|---:|
| 主前向批次 | 819 | 819 |
| 首批重复批次 | 18 | 18 |
| 主有效预测行 | 62586 | 62586 |
| 重复有效预测行 | 1440 | 1440 |
| 主计算槽（含padding） | 65340 | 65340 |
| 重复计算槽 | 1440 | 1440 |
| 固定checkpoint加载 | 18 | 18 |
| 每模态共享source观察步 | 6954 | 6954 |
| source动作读取行（含60个label-only t0） | 7014 | 7014 |
| 训练更新 / backward / 环境交互 / 初态或private state | 0 | 0 |

18/18首批重复逐值精确，包含padding；18/18全support汇总匹配历史500点（有效n、符号准确率精确，MSE abs1e-10/rel0）。所有输入批次摘要共享且前后不变，模型文件/tensor hash验证、读出后权重和grad检查均通过。source-query、qualification、reserved、validation、held-out、final evaluation保持封存。闭环稳定化5步不变，离线专家观察额外稳定化0。

## 窗口和指标

早期为每demo动作t=1..20；close/open事件为专家夹爪符号从非正到正/非负到负，取前后各5步并集。动作0仅参与t=1事件标签，不评分、不重建初态。它们是命令标签，不是物理抓取/运输/释放阶段。

主指标为平移、旋转、夹爪MSE和符号错误率。每窗口同时报告帧占比、误差占比、集中比、窗口内外均值及差；MSE按分量维数3/3/1归一化。8个互斥原子桶保留所有交叉及空桶：task0的close/open窗口重叠21帧，task1重叠0帧；不能相加边际窗口的错误占比。

## B臂：错误在哪里

以下是三个seed算术均值，逐seed和A/C全部保留于JSON/NPZ。错误占比指每seed窗口内错误数占其全部错误数的比例，再对三seed平均，不把三模型重复帧当新样本。

| task / 窗口 | 帧数 / 占比 | 平移MSE：内 / 外 | 夹爪符号错误率：内 / 外 | 符号错误占比 |
|---|---|---|---|---:|
| 0 / 早期 | 600 / 20.73% | 0.033019 / 0.078284 | 0.00% / 8.76% | 0.00% |
| 0 / close附近 | 370 / 12.78% | 0.073742 / 0.068193 | 20.90% / 4.90% | 38.75% |
| 0 / open附近 | 339 / 11.71% | 0.164138 / 0.056272 | 33.43% / 3.43% | 58.28% |
| 1 / 早期 | 600 / 14.78% | 0.011815 / 0.063419 | 0.00% / 8.84% | 0.00% |
| 1 / close附近 | 374 / 9.21% | 0.034472 / 0.057955 | 24.78% / 5.78% | 30.32% |
| 1 / open附近 | 361 / 8.89% | 0.098976 / 0.051575 | 43.86% / 3.98% | 54.09% |

task0/1 B的open窗口只占11.71%/8.89%的专家帧，却分别包含平均58.28%/54.09%的符号错误；集中比分别4.98/6.08。夹爪幅值MSE也提高：open窗口内/外为task0 `0.890294/0.103480`、task1 `1.051461/0.122802`。平移和旋转的open窗口误差同样在两task、三seed均高于补集；不能只描述成夹爪head问题。

close附近的夹爪MSE/符号错误也在全部18模型中高于补集，但平移没有统一升高：task1三臂三seed均低于补集，task0方向混合。不能把close和open归成同一个几何错误模式。

## 全臂一致性及早期反证

下表只列四项主指标中满足“三seed窗口内均值均高于补集”的项目；全部72个flag和每seed差值都保留，flag不是显著性或训练门槛。

| task / arm | 早期 | close附近 | open附近 |
|---|---|---|---|
| 0 / A | 无 | 旋转、夹爪MSE、符号错误 | 四项全部 |
| 0 / B | 无 | 夹爪MSE、符号错误 | 四项全部 |
| 0 / C | 无 | 旋转、夹爪MSE、符号错误 | 四项全部 |
| 1 / A | 无 | 夹爪MSE、符号错误 | 四项全部 |
| 1 / B | 无 | 夹爪MSE、符号错误 | 四项全部 |
| 1 / C | 无 | 夹爪MSE、符号错误 | 四项全部 |

**18模型在早期600帧/任务上符号错误全部为0，但这些早期专家目标也全部为open。** 这说明模型能拟合该专家输入条件下的open读出，没有验证早期close判断、接触或自由运行首动作。专家早期输入与candidate reset观察不同，没有runtime专家标签，不能据此认定闭环首动作正确或排除初态观察分布问题。

open热点也不是统一“该open时不open”：B臂near-open符号混淆如下，均为实际帧数，两个方向全部报告。

| task / seed末两位 | 专家open、预测close | 专家close、预测open |
|---|---:|---:|
| 0 / 22 | 16 | 102 |
| 0 / 23 | 61 | 50 |
| 0 / 24 | 78 | 33 |
| 1 / 22 | 153 | 19 |
| 1 / 23 | 24 | 111 |
| 1 / 24 | 153 | 15 |

同一task的seed间错误方向会反转；不能统一增加open偏好、延后close或将幅值MSE当实际握力。窗口横跨事件前后，符号混淆也不是已确认的实际动作延迟。A/C均有同类热点，B的SPT候选身份不能由这些source-support误差获得接受。

## 结论和下一步

本轮完成了“错误是否集中在早期/切换附近”的定位：**专家输入上早期不集中，close/open附近集中，open附近同时有机械臂分量误差和夹爪命令错误。** 这补充了前序整体MSE，不证明这些帧导致闭环失败。

保持500 updates、现有loss/观测、闭环5步，维持此前局部2×gripper loss和增强分支的停止决定。下一项可只读本轮NPZ，把near-open按事件前后、两种符号错误方向、demo及同帧平移误差分开，检查是否有跨seed共同模式；该步骤无需新Policy前向，也不先调权重或访问qualification。当前尚未据此进行训练修改，不更新活动SPT/Knowledge或注册Module。

## provenance、审计与验证

入口总耗时14.0731s，UTC 2026-11-03 09:02:54.998942请求、09:02:55.739916开始、09:03:09.072014结束。cache0.9974s、推理与逐批检查11.6431s、数值分析0.6224s。耗时短是固定权重推理，无训练backward/optimizer。Python3.10.22、Torch2.4.1+cu124、CUDA12.4、NumPy1.22.4、RTX4090物理GPU1；deterministic开启、TF32关闭，host cache峰值415117180 bytes、CUDA allocated峰值149237760 bytes。

| 产物 | SHA-256 |
|---|---|
| 科学配置 | `7d3447ab1b736a3a123a3dd68dcfa7d22f6e6e54f1aaaae9cedd55a331968e75` |
| 执行绑定 | `636780603866708614782165712211d3a011b715bbc6c77365dfb0a9d1a9f47a` |
| `/tmp/libero-source-internal-fixed500-expert-frame-20261103-v1.json` | `0b7b12554ce7df86ad1d5e1731417376e8bf72ab72d2d6b7357cf9071f797489` |
| `/tmp/libero-source-internal-fixed500-expert-frame-independent-audit-20261103-v1.json` | `5b0e206a294bee121be4557f573860c672e9022341c773aefb7ada993081bb4e` |
| 独立审计 `/tmp/audit_libero_source_internal_fixed500_expert_frames_20261103_v1.py` | `4113487fed9aeac11704ee9d8d5c484b3b78aeeb7487862794a33777cf4f3207` |
| task0完整NPZ | `eae6a721620819458b5e81b0738f23ea95f1de6b043cf4322507a61fc53768bd` |
| task1完整NPZ | `20394b4fb6da65a1d6b6b5e51003c000563935ff16c5420557b53eca265d2ac5` |

逐批events、两个完整NPZ、两个提前保存的labels NPZ及18份模型prediction NPY均在`/tmp/libero-source-internal-fixed500-expert-frame-artifacts-20261103-v1/`；没有RGB、checkpoint副本或simulator state。

独立审计不导入runner/算子，从完整NPZ重算18模型×18分层、9720个demo分层、72个flag和12个配对；294770项递归检查通过，预设abs/rel1e-12，最大差0。核对事件窗口t>=7的定义、全部早期/命令/原子分区和原始预测文件。action0数值未归档，t=1事件标签没有再读source独立核验；前6帧只检查与某一合法action0符号一致，主入口的标签逻辑另有合成测试和绑定。审计新增Policy/HDF5/checkpoint/环境/训练调用均0。

新增25项测试，与前序汇总、causal数据集和确定性池化合计 **61 passed in0.93s**；编译、绑定preflight和CPU环境检查通过。未运行缺少可选Crafter依赖的全量pytest；没有修改共享训练或环境代码。原产物/保护文件、最终数字和hash核对见`/tmp/libero-source-internal-fixed500-expert-frame-final-verification-20261103-v1.json`。所有结果为已暴露source内部diagnostic，不是独立泛化、闭环因果、SPT接受或Module qualification。

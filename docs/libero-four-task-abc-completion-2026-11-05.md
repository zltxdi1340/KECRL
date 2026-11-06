# LIBERO四任务A/B/C本轮收尾结果（2026-11-05）

**task0/1/4/7的三seed A/B/C训练、固定500端点闭环和独立审计现已补齐。** 本轮新增task4/7的18个Policy、9000次适应更新和360条闭环；与历史task0/1的360条分别保留provenance，不计算四任务混合成功率。四任务沿用相同学习流程和评价预算，但来源几何覆盖不同，全部属于已暴露source内部diagnostic。

## 固定500端点闭环

每个已测格为三训练seed×20个已用candidate初态；稳定化5步、环境seed20261022、最多300个Policy动作。A是原始生成器初始化，B是上下文FOMAML候选，C是普通pooled BC预训练；每个Policy都在探针support上适应500次。

| task | 黑碗起始位置 | A | B | C | 必须保留的分层 |
|---|---|---:|---:|---:|---|
| 0 | 盘子与小烤碗之间 | 8/60（13.33%） | 12/60（20.00%） | 12/60（20.00%） | 来源覆盖探针几何 |
| 1 | 小烤碗旁边 | 0/60（0.00%） | 11/60（18.33%） | 1/60（1.67%） | 来源覆盖探针几何 |
| 4 | 已打开的上层抽屉内 | 0/60（0.00%） | 1/60（1.67%） | 1/60（1.67%） | 来源缺探针几何 |
| 7 | 炉灶上 | 0/60（0.00%） | 0/60（0.00%） | 0/60（0.00%） | 来源缺探针几何 |

下表每格分母20，seed顺序固定20261022/23/24：

| task | 臂 | seed22 | seed23 | seed24 |
|---|---|---:|---:|---:|
| 0 | A_original | 0 | 6 | 2 |
| 0 | B_fomaml | 11 | 0 | 1 |
| 0 | C_pooled_bc | 4 | 5 | 3 |
| 1 | A_original | 0 | 0 | 0 |
| 1 | B_fomaml | 0 | 0 | 11 |
| 1 | C_pooled_bc | 0 | 1 | 0 |
| 4 | A_original | 0 | 0 | 0 |
| 4 | B_fomaml | 1 | 0 | 0 |
| 4 | C_pooled_bc | 0 | 1 | 0 |
| 7 | A_original | 0 | 0 | 0 |
| 7 | B_fomaml | 0 | 0 | 0 |
| 7 | C_pooled_bc | 0 | 0 | 0 |

task0的B/C总数相同；task1的B高于C但11次成功集中于seed24。新task4/7也按全部seed保留，不挑有利结果。现有四任务结果不构成覆盖全部任务和三个seed的稳定B>C证据；不能把单seed较高成功率当可靠迁移。

配对四格顺序为“共同成功 / 仅B成功 / 仅参照成功 / 共同失败”，每行60对；逐seed差值以seed22/23/24排列。只作描述，不进行事后显著性检验：

| task | 比较 | 逐seed成功数差 | 配对四格 |
|---|---|---|---|
| 0 | B−A | +11/-6/-1 | 0 / 12 / 8 / 40 |
| 0 | B−C | +7/-5/-2 | 4 / 8 / 8 / 40 |
| 1 | B−A | +0/+0/+11 | 0 / 11 / 0 / 49 |
| 1 | B−C | +0/-1/+11 | 0 / 11 / 1 / 48 |
| 4 | B−A | +1/+0/+0 | 0 / 1 / 0 / 59 |
| 4 | B−C | +1/-1/+0 | 0 / 1 / 1 / 58 |
| 7 | B−A | +0/+0/+0 | 0 / 0 / 0 / 60 |
| 7 | B−C | +0/+0/+0 | 0 / 0 / 0 / 60 |

## 离线适应与闭环分开解释

以下为三seed的query七维动作MSE均值、样本标准差及固定0/50/150/300/500五点曲线的0..500归一化面积。标准差描述seed差异，不是成功率置信区间，曲线不用于选checkpoint。

| task | 臂 | 500 query MSE均值 | seed样本标准差 | query曲线面积 |
|---|---|---:|---:|---:|
| 0 | A_original | 0.085372 | 0.008339 | 0.128576 |
| 0 | B_fomaml | 0.081612 | 0.010240 | 0.112937 |
| 0 | C_pooled_bc | 0.079437 | 0.007875 | 0.111308 |
| 1 | A_original | 0.052681 | 0.004409 | 0.095564 |
| 1 | B_fomaml | 0.052996 | 0.006405 | 0.081101 |
| 1 | C_pooled_bc | 0.053334 | 0.003915 | 0.080563 |
| 4 | A_original | 0.043222 | 0.004297 | 0.082471 |
| 4 | B_fomaml | 0.040961 | 0.004849 | 0.067416 |
| 4 | C_pooled_bc | 0.040361 | 0.005486 | 0.064733 |
| 7 | A_original | 0.043415 | 0.001442 | 0.081610 |
| 7 | B_fomaml | 0.042704 | 0.000191 | 0.076386 |
| 7 | C_pooled_bc | 0.042145 | 0.000333 | 0.071556 |

来源反馈改善部分早期离线学习，不等于FOMAML独有优势，也不保证闭环。task4/7的B-A曲线面积均各3/3 seed下降，但B-C面积各3/3更高；500端点B-C仅task4一seed更低、task7没有。task0/1与新结果采用相同t>=1 causal输入口径；原task4/7的单任务BC含t0重建、初始化也不同，不能把其3/60、4/60填作本表A。原BC/jitter结果仍见[历史横向整理](libero-task0-task1-task4-task7-results-comparison-2026-11-05.md)。

训练曲线图：[/tmp中的四任务support/query均值图](/tmp/libero-four-task-abc-completion-20261105-v1-curves.png)。实测逐seed数据保存在同前缀`-curves.csv`（360行）、`-seed-endpoints.csv`（36行）、`-paired-outcomes.csv`（24行）和`-overview.csv`（12行）；无需重训即可复查。

## 设计、复现和解释边界

task0探针的来源为1/3/4/7，task1为0/3/4/7，均覆盖四类来源几何。task4来源0/1/3/7，缺open_container；task7来源0/1/3/4，缺fixture_surface。因此后两者是几何外推诊断，不能把四任务当同分布或给固有难度排序。task4的抽屉初始已开，没有学习开抽屉。

source演示support0..29/query30..49角色固定；来源query是元更新/普通预训练的训练反馈，探针query只评价。每fold全部九个support fit完成后才读取探针query。B/C各50次生成器更新且数据梯度/批次/容量匹配；B另有临时Policy内循环，不声称全部计算开销相等。500仅指探针适应预算。两次闭环使用同GPU1/EGL1、相同runtime与确定性池化，但分别执行，旧结果不改写为本次fresh测量。

本轮修复的是扩展入口适用性：旧task0/1专用的query门槛不能按`probe+1`外推到非连续编号4/7；新版按fold序号、完整seed/arm/端点检查。旧几何覆盖门槛不放松，新外推设计独立冻结。新增汇总端到端单测发现Path对象JSON导出错误，已在真实汇总前修复；冻结训练/闭环及研究预算不受影响。旧文件、历史失败与其他分支的负结果保留，不声称这些修正提高了控制成功率。

## 新增实验成本、审计和provenance

训练详见[task4/7执行报告](libero-source-internal-geometry-extension-results-2026-11-05.md)：18模型、132.023815秒，9000适应更新、600生成器更新、1200临时SGD；108份权重、180组指标和9000更新事件独立审计通过，初始化重建差0。

新增闭环实际360条、107776个Policy动作、1800个固定零动作、109576个native step；elapsed1832.079486秒。全部attempted/completed计数相同；18个adapter、40个reset reference及320次精确比较通过。独立审计复核360份episode及全部公开offset、18份端点权重和四格，额外Policy/环境/optimizer/初态bank调用0。

训练相关48项、闭环相关133项、CPU回归1351项和最终汇总12项测试通过。CPU回归排除可选Crafter、spatial_gather和smoke组，不声称全量pytest。最终文件保全、代码/输入hash、语法/文档检查和HEAD记录见`/tmp/libero-source-internal-geometry-extension-final-verification-20261105-v1.json`。

汇总JSON：`/tmp/libero-four-task-abc-completion-20261105-v1.json`，SHA-256 `60ff204b060976fade649f26e2fe37d5352d8a65b0008d517d21c0579a41b4bd`。以下八份原始结果/独立审计分别固定hash，不覆盖历史文件：


- 分层`geometry_covered`：
  - `/tmp/libero-source-internal-spt-spi-20261103-v2.json`，SHA-256 `66a4fb7fac6288a710072271e440f6cea42167f4fb782b9a774c7554af6d0169`。
  - `/tmp/libero-source-internal-spt-spi-independent-audit-20261103-v2.json`，SHA-256 `c9b8b0d5174374cdd0f9c7493bc9211e0f30ee14a7f0be4c5224527a3f7ca68b`。
  - `/tmp/libero-source-internal-spt-spi-closed-loop-20261103-v1.json`，SHA-256 `4cba11fd4e5cb22d471ee681ae0e18a7b77c4698f904439a36e37ef9f4ba3ad0`。
  - `/tmp/libero-source-internal-spt-spi-closed-loop-independent-audit-20261103-v1.json`，SHA-256 `bec300a1adeaedc324dbc6f89325ce53bb13993a5378e08c88ce26cc7a2314e2`。

- 分层`probe_geometry_absent_from_source_updates`：
  - `/tmp/libero-source-internal-geometry-extension-20261105-v1.json`，SHA-256 `cbd97afc6f87a5c064aa55a4dd8f5b088afcd8d3ad9b13b0f9d596b9d8da1419`。
  - `/tmp/libero-source-internal-geometry-extension-independent-audit-20261105-v1.json`，SHA-256 `d07c4e8b7da0b075bb7eb718dcfe1eac3eb84bb9d254266f0f0f112e0f861f4c`。
  - `/tmp/libero-source-internal-geometry-extension-closed-loop-20261105-v1.json`，SHA-256 `edd4c5ac9263044ceebf61d3aca27d183823fe573ebfdac578702ac3d2f7b8df`。
  - `/tmp/libero-source-internal-geometry-extension-closed-loop-independent-audit-20261105-v1.json`，SHA-256 `cb1689536c2341ddba46ddb9c7e69972b361bf1611b3fb4b19c7182a87640f77`。

## 本周完成范围与下一步

本周优先项“source内部四任务方法对照、真实闭环、独立审计及可复现结果交付”已完成，时间与数据证据齐全，不以B必须获胜作为完成条件。当前配置下没有充分证据接受FOMAML候选；保持活动SPT不变，不注册Module、不更新Knowledge。

整个LIBERO正式验证仍未完成：validation2/6、held-out5/8/9、qualification20..39、reserved40..49及final evaluation继续封存。下一步先冻结正式实验的主假设、任务角色、学习总预算、baseline、checkpoint规则、评价统计与qualification阈值，并明确闭环分辨力不足时的停止决策；在此之前不读取封存数据，不因本轮成绩追加无止境调参。原始观测敏感性仅留作未决问题，本轮不继续展开小读出头分支。

给老师可概括为：“LIBERO已完成四个来源任务、三种初始化、三个seed的固定预算适应和配对闭环。来源经验能改善部分离线学习，但尚未证明FOMAML稳定优于普通预训练，控制成功率与seed稳定性仍有限；几何覆盖与外推任务分开报告。当前是source内部可行性结论，正式留出迁移和Module资格评价尚未开启。”

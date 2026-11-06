# LIBERO task4/7 A/B/C几何外推诊断执行记录（2026-11-05）

**训练、360条配对闭环及两份独立审计均已完成。四任务最终分层结果见[收尾报告](libero-four-task-abc-completion-2026-11-05.md)。** 本项按[执行前冻结协议](libero-source-internal-geometry-extension-protocol-2026-11-05.md)补齐四任务对照，不改变500 updates、5步稳定化或均匀七维loss。全部seed、实际成本及独立审计已补记如下；没有使用中途结果选择模型或续训。

## 已完成设置和离线结果

探针4的source更新任务为0/1/3/7，探针7为0/1/3/4。两者来源分别缺open_container、fixture_surface，属于几何外推分层；与task0/1来源覆盖探针几何的结果分别解释。任务都已暴露，评价candidate0..19已多轮使用，不能称未见泛化。task4抽屉初始已经打开。

A不做来源更新；B为support一步适应后的query一阶反馈；C为相同source批次与容量的pooled BC。每fold、seed的B/C各50次生成器Adam，每个探针Policy各500次Adam。三seed20261022/23/24，双RGB+joint/gripper的CNN-only Policy、t>=1 causal输入、batch8、sequence10、stride10，统一loss和初始化/批次配对。probe query在该fold全部九个support fit后才读取，仅评价。详细优化器与预算以冻结配置为准。

500端点query七维动作MSE，三seed均值：

| task | A原始初始化 | B FOMAML | C普通预训练 |
|---|---:|---:|---:|
| 4 | 0.043222 | 0.040961 | 0.040361 |
| 7 | 0.043415 | 0.042704 | 0.042145 |

B-A的500端点MSE在task4改善3/3 seed、task7改善2/3；B-C分别仅1/3、0/3。固定0/50/150/300/500点的0..500归一化query曲线面积，B低于A在两任务各3/3，但B高于C亦各3/3。支持来源反馈改善早期离线适应的线索，尚不支持FOMAML相对普通预训练优势。不得将离线动作误差换算为闭环成功率。

训练首次完整运行，新增18个Policy、9000次适应更新、600次生成器Adam、1200次临时SGD、4800次source梯度、2400次VJP及90份曲线checkpoint全部符合预算。运行UTC03:48:03.769192请求、03:48:07.304536开始至03:50:19.328344结束；核心elapsed132.023815秒，含输入校验和设置总入口135.559159秒。分段dataset/cache4.588516秒、source更新39.114632秒、probe拟合65.862988秒、离线评价21.852599秒；计时器范围不同，不作精确互斥相加。

物理GPU1 RTX4090、进程cuda:0、PyTorch2.4.1+cu124/CUDA12.4、Python3.10.22、NumPy1.22.4。host tensor cache峰值3,174,367,630字节，CUDA allocated峰值486,916,096字节。GPU启动前空闲，GPU0未占用。等价池化CPU/CUDA输出与梯度差0，重复位级一致，关闭TF32；生成器初始化重建误差0。训练阶段0环境、0官方初态读取。

独立离线审计复核108份文件/tensor hash、180组指标、9000条适应更新及source窗口角色/预算；额外Policy前向/optimizer/环境0。审计1.945966秒，通过。

## 执行修正与验证边界

历史v2入口只支持task0/1，查询开启计数使用`(probe+1)*9`；扩展到非连续编号4/7时不能直接复用。本次新版本按已完成fold序号及完整seed/arm/五点端点检查，训练前测试覆盖缺seed、重复seed、缺臂、缺500及错误task编号算术。旧几何覆盖门槛也在新协议中明确改为三类来源几何的外推要求，不放宽历史对照的四类覆盖。旧执行代码、冻结文件及既有结论保留不变；没有发现需要重跑旧task0/1的证据。

训练相关48项、闭环及候选前缀相关133项、汇总角色/完整性12项测试分别通过；排除`tests/*crafter*.py`、`test_spatial_gather_spt.py`和`test_smoke.py`的CPU回归1351项通过（18.22秒，发生在新增汇总测试之前）。不声称包含可选Crafter的全量pytest。新增端到端汇总测试曾发现合成Path对象未转字符串的导出错误，已在真实汇总前修复，扩展测试最终12项通过；不影响冻结训练/闭环或研究调用预算。原测试失败与修正记录在`/tmp/libero-geometry-extension-software-validation-20261105-v1.json`。四个执行/审计入口和汇总入口编译、运行环境检查通过。

## 绑定与产物

- [训练配置](../configs/libero-source-internal-geometry-extension-diagnostic_v1.json)，SHA-256 `9cd4fbdf7a9ea2110c1716c3db134f6b0da340e7659983cba12e54fa72405fa8`；[训练绑定](../configs/libero-source-internal-geometry-extension-execution-binding_v1.json)，`bbd9eb8cf9df39e516a71f9df05566f4953e7d02dad8f92b54d1e39a439d274c`。
- 训练结果`/tmp/libero-source-internal-geometry-extension-20261105-v1.json`，SHA-256 `cbd97afc6f87a5c064aa55a4dd8f5b088afcd8d3ad9b13b0f9d596b9d8da1419`；权重保持在`/tmp/libero-source-internal-geometry-extension-artifacts-20261105-v1`，不复制进仓库。
- 训练审计`/tmp/libero-source-internal-geometry-extension-independent-audit-20261105-v1.json`，结果状态passed、初始化重建差0，完整SHA另见最终核验记录。
- [闭环配置](../configs/libero-source-internal-geometry-extension-closed-loop-diagnostic_v1.json)，SHA-256 `0d835fa1d74f5696515123f1e828a6b1301cb59813a86a6bf87b306422dc31e3`；[闭环绑定](../configs/libero-source-internal-geometry-extension-closed-loop-binding_v1.json)，`243c7e97cbe2fff61a2b10f6e7b68efe077ef4dc553ca56ddbb24b8c22c4db93`。全部18份500端点文件/tensor预检通过，预检0初态解码/环境。
- CPU回归原始日志`/tmp/libero-geometry-extension-cpu-regression-20261105-v1.log`。

闭环按task→seed→A/B/C→candidate0..19执行，最多108000个Policy动作及1800个稳定化零动作。40个reset reference、320次精确比较，独立审计后才形成主结果。技术失败保留partial，不补样本或放松门槛。



## 闭环完成结果与独立审计

| task | A | B | C |
|---|---:|---:|---:|
| 4 | 0/60 | 1/60 | 1/60 |
| 7 | 0/60 | 0/60 | 0/60 |

- task4 A_original逐seed成功数：[0, 0, 0]（各20条）。

- task4 B_fomaml逐seed成功数：[1, 0, 0]（各20条）。

- task4 C_pooled_bc逐seed成功数：[0, 1, 0]（各20条）。

- task7 A_original逐seed成功数：[0, 0, 0]（各20条）。

- task7 B_fomaml逐seed成功数：[0, 0, 0]（各20条）。

- task7 C_pooled_bc逐seed成功数：[0, 0, 0]（各20条）。

实际107776个Policy动作+1800个稳定化零动作，native step总109576，elapsed1832.079486秒。独立审计3.880684秒，360份episode/108136行公开轨迹及18份权重通过，40个reset reference和320次精确比较通过。研究执行首次完整成功，无技术中止/重试。只数值解码两个bank的candidate前20行（共40），后30行未解码；容器SHA校验字节读取与数值访问分别记账。

闭环结果`/tmp/libero-source-internal-geometry-extension-closed-loop-20261105-v1.json`，SHA-256 `edd4c5ac9263044ceebf61d3aca27d183823fe573ebfdac578702ac3d2f7b8df`；独立审计`/tmp/libero-source-internal-geometry-extension-closed-loop-independent-audit-20261105-v1.json`，SHA-256 `cb1689536c2341ddba46ddb9c7e69972b361bf1611b3fb4b19c7182a87640f77`。全部端点、配置、代码绑定保持不变。

## 下一步与本周收尾

360条闭环及独立审计已完成，四任务逐seed、配对四格、曲线CSV与总表均已交付。下一步按收尾报告冻结正式方案，当前不自动开启封存角色或继续小头调参。本周范围见[收尾计划](libero-weekly-completion-plan-2026-11-05.md)。不根据效果加updates、改loss或挑seed；validation/held-out/qualification/reserved/final evaluation继续封存，不接受活动SPT、不更新Knowledge、不注册Module。

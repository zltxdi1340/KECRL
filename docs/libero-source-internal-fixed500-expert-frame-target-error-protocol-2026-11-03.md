# 固定500端点的逐帧专家动作读出协议（2026-11-03）

本轮为source内部diagnostic。[0/500点汇总分解](libero-source-internal-initialization-endpoint-target-error-diagnostic-2026-11-03.md)已说明平均目标误差不能单独解释六组闭环差异；原汇总没有逐帧预测。本协议在读取本轮数值payload之前冻结，用同一批18个500端点定位早期与专家close/open切换附近的误差分布，不改训练。

## 数据、端点与精确预算

[科学配置](../configs/libero-source-internal-fixed500-expert-frame-target-error-diagnostic_v1.json)固定task0/1、seed20261022/23/24、A_original/B_fomaml/C_pooled_bc全部18个500-update checkpoint的文件SHA和tensor hash。无0点或其他checkpoint选择。

仅source-support demos0..29的`obs[t-1] -> action[t]`、t>=1。先完成角色授权和文件SHA核验，再用原CausalT1数据集读取；官方初态、query、validation、held-out、qualification和private state均无访问路径。动作t=0只为t=1切换标签读取，不评分、不重建初始观察。保留闭环稳定化5步，离线输入不执行额外稳定化。

| task | 唯一专家步 | sequence10/stride10窗口 | batch8批次/模型 | 模型数 | 主前向批次 |
|---|---:|---:|---:|---:|---:|
| 0 | 2895 | 304 | 38 | 9 | 342 |
| 1 | 4059 | 422 | 53 | 9 | 477 |
| 合计 | 6954 | 726 | 91 | 18 | **819** |

各模型仅将原首批8窗口额外读出一次，共**18个重复批次**；总计**837批次**。主有效预测62586行，重复1440行；含sequence padding的计算槽分别65340/1440，padding不评分。重复检查不增加独立样本。每任务所有9模型共享缓存、窗口顺序和输入摘要；最后不足8个窗口的批次不补窗口。

source只读取一次共享的6954个观察步/模态，动作有效目标6954行加60个label-only action0，共7014行；不对每个模型重新读取示教。2GiB host cache上限。训练updates、backward、环境reset/step、Policy控制步数均0。

预算依据为已核对的60个support组shape/dtype元数据：`/tmp/libero-source-internal-expert-frame-metadata-20261103-v1.json`，SHA `916321c4bf539b0bf9d9ced84873122a6f2760b285be5c9f7276278e31717a65`。该次核对未解码动作/RGB/观察数组；两个完整source文件只作SHA校验，没有读取query组或初态。

## 窗口与误差口径

- 早期固定为每demo专家动作`t=1..20`；其余`t>=21`为补集。这是专家时间轴，不与自由运行Policy的第20步强行匹配。
- close事件：专家夹爪符号从非正到正；open事件：从非负到负，事件t>=1。事件前后各5步（含端点），截到同demo的合法评分区间；同类窗口取并集，帧不重复。
- 保留早期、near-close、near-open的全部交叉：三布尔标记组成8个互斥原子桶，包含空桶。边际窗口可重叠，不把它们相加当总体贡献；原子桶必须完整分割全部6954帧。
- 同时按expert open/close/neutral分层。以上都是命令标签，不等于物理接触、抓取、运输或释放阶段。

主指标为平移、旋转、夹爪MSE及夹爪符号错误率，分别报告，不挑有利分量。平移/旋转/夹爪按3/3/1维归一化，保留七维MSE、夹爪有符号bias及3×3符号混淆矩阵作次级读出。严格按正/负/零分类，无epsilon调节；连续夹爪幅值误差不能直接解释为握力或实际闭合不足。

每窗口/指标同时输出帧数、补集帧数、帧占比、误差总量占比、二者之比、窗口内/外均值及差值。MSE误差总量为逐帧分量均方误差之和；符号误差为错误帧数。这样区分“帧多”与“单位帧误差高”。分母为0或分层为空时用null和计数，不补0。

保留全部六组与三臂，逐seed报告；主汇总按帧加权，次级给每demo均值/中位数及有效/缺失demo数。B-A/B-C在相同帧和分层上配对，不按闭环成败筛输入。每task/arm/指标/窗口的三seed均满足非空且窗口内均值高于补集时，才标记描述性一致集中；有缺失则null，其余false。该标记不是显著性、根因或训练修改门槛，不合并任务，不把重叠帧和重复预测当独立样本。

## 执行、复核与停止

物理GPU1、进程cuda:0、CUDA_VISIBLE_DEVICES=1；idle门槛为显存<=512MiB、利用率<=5%、无compute进程，失败时在payload/model加载前停止。不使用GPU0或自动切换CPU。Python3.10.22、Torch2.4.1+cu124、CUDA12.4、NumPy1.22.4；deterministic algorithms/cudnn开启、benchmark/TF32关闭，复用原确定性平均池化。推理seed20261022、threads1，模型eval、inference_mode，权重hash前后不变、grad保持None。

执行代码与测试通过后单独绑定runner/validator/tests/本文及原数据/Policy代码；科学配置不因结果改变。各模型首批重复必须逐值精确一致（含padding）。18模型的全support结果与历史500点核对：有效步数和符号准确率精确，四项MSE abs1e-10、rel0；这只是汇总一致性，不证明历史逐帧位级重放。

输入/权重/shape/dtype/角色/计数/有限性/动作界/重复/汇总参照/分区恒等式任一不符立即停止；保留失败JSON、事件和已产生的partial文件，不放宽容差、不换checkpoint、不追加前向或调参重试。执行产物另存`/tmp`：JSON、逐批输入/输出摘要、每task一个targets/9模型predictions/帧ID/masks原始NPZ，无RGB、权重或simulator state。审计从NPZ独立复算，新增Policy调用0。

本协议的冻结与元数据验证不是实验结果。实际前向、耗时、设备、所有计数、artifact hash、支持证据和反证另写dated执行报告。即使发现集中，也不自动调整loss/预算/观测，不更新活动SPT或Knowledge，不注册Module。

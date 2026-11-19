# LIBERO受限视觉微调对照：真实训练与闭环授权记录（2026-11-14）

**12个新Policy各500更新已完成，共6,000次真实更新，独立训练核对通过。** 训练阶段内部计时72.918719秒，所有attempted/completed一致。后续240条配对闭环已获本轮明确授权并建立新评估绑定；本页记录训练阶段，不将训练loss当闭环收益。

科学范围见[冻结协议](libero-restricted-vision-finetune-protocol-2026-11-14.md)及[准备记录](libero-restricted-vision-finetune-preparation-2026-11-14.md)。task4/7 × seeds20261106/07/08 × `frozen_resnet18`/`layer4_conv_finetune`，每个从相同公共初始化独立训练，只保存500终点；未使用旧u500续训。两臂完整初始tensor与500批次身份配对通过。

## 本轮授权和执行绑定

用户明确指令：“建立本轮执行绑定后训练。授权后续240条配对闭环”。据此新建[本轮初态授权](../configs/libero-restricted-vision-finetune-source-initial-authorization_v1.json)，限task4/7已用candidate_nonregression初态0..19、12模型合240条fresh episode、每条5步稳定化后最多300个Policy动作。授权绑定本轮科学配置，未转用上一轮授权，没有开放source meta_query、其他任务或正式数据角色。

- [训练执行绑定](../configs/libero-restricted-vision-finetune-training-binding_v1.json)：研究权重/HDF5读取前建立；573项依赖及143项历史合成验证证据核验通过，训练阶段numeric_rows/episode预算均为0。
- [评估执行绑定](../configs/libero-restricted-vision-finetune-evaluation-binding_v1.json)：独立训练核对通过后、初态读取前建立，绑定本轮授权、训练输出及核对收据。闭环阶段训练更新/HDF5预算均为0。
- 科学配置、冻结协议/绑定、旧试验文件和执行源码均保持原样；冻结中的“未授权/未绑定”字段是准备时的历史状态，本轮新记录承载后续授权与执行状态。

## 实际训练范围与预算

| 工作 | 完成 / 上限 |
| --- | ---: |
| 训练前向 / backward / Adam更新 | 各6000 / 6000 |
| 保存500终点 / 模型构造 | 12 / 12；14 / 14 |
| source窗口 / 源文件SHA | 890 / 1100；2 / 2 |
| HDF5打开 / 数组切片 | 4 / 4；4450 / 5500 |
| layer3 prefix前向 | 248 / 302 |
| layer4前向 / 像素缓存Policy核对 | 12024 / 12024；12 / 12 |
| 初态数值行 / adapter / reset / episode | 均0 |

task4/7各读取meta_support demos0..29，causal obs[t−1]→action[t]、t≥1，分别458/432窗口。不解码action0、query、attrs、私有simulator state或正式数据。完整SHA会读取容器所有字节，不声称query字节物理隔离。

Adam lr3e−4、betas(.9,.999)、eps1e−8、weight_decay0、amsgradFalse、batch8、sequence/stride10及原uniform masked七维MSE保持。所有实际optimizer state恰好500，全部更新轨迹保留；没有按loss选择checkpoint。

候选仅训练layer4五个卷积权重，新增8,388,608个可训练参数；总参数相同12,689,864，控制/候选可训练参数分别1,513,352/9,901,960。训练前后不可训练tensor、全部BN/归一化与layer3缓存摘要不变；独立核对再次对照公共权重检查冻结范围。每个模型第一批原像素/缓存动作差通过预设1e−5实现核对。

| task / seed | 冻结臂裁剪次数 | 微调臂裁剪次数 |
| --- | ---: | ---: |
| 4 / 20261106 | 69 | 23 |
| 4 / 20261107 | 53 | 14 |
| 4 / 20261108 | 55 | 18 |
| 7 / 20261106 | 37 | 17 |
| 7 / 20261107 | 46 | 20 |
| 7 / 20261108 | 46 | 16 |
| 合计 | 306 | 108 |

所有trainable参数共同参与global clip norm1。裁剪前范数超过1记一次，两臂容量及梯度范围不同，因此不把相同超参数解读成head每步有效更新相同，也不把裁剪次数差异当成功原因。

按task依次保留cache：task4 raw450,530,020/prefix600,309,760字节；task7 raw424,954,080/prefix566,231,040字节，均在各2GiB上限内。训练Torch allocator峰值389,347,840字节，不代表完整host RSS/渲染VRAM。物理GPU1/CUDA逻辑0、单线程、确定性、TF32关闭；Python3.10.22、Torch2.4.1+cu124及其余版本与配置一致。内部计时不含公共预训练、准备、独立核对与未来闭环成本。

## 证据与下一阶段

[紧凑训练摘要](evidence/libero-restricted-vision-finetune-training-summary-2026-11-14.json)保留全部模型pin、初始化/采样/不可训练状态摘要、实际计数、裁剪次数和训练loss记录。真实输出位于`results/libero-restricted-vision-finetune-20261114-v1/`：`training.json`、`training-audit.json`、12个500终点及完整`training-events.jsonl`。日志在`logs/libero-restricted-vision-finetune-execution-20261114-v1/`，含执行前快照、环境、训练/审计控制台。

本轮未修改执行代码或重跑pytest，沿用已绑定143项软件验证；真实训练与独立核对为本轮新增验证。后续闭环必须保持全部240身份、严格reset配对、原生成功判据及原10个百分点/各task与seed不退化门；不追加训练、换seed、补样本或按中途成功率调整设置。开发试验不构成正式SPT迁移或Module准入，LIBERO继续作为主环境。

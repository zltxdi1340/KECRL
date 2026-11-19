# LIBERO冻结视觉与受限视觉微调对照：科学协议（2026-11-14）

本轮准备同架构、同初始化的完整控制开发对照，检验在固定500更新内开放最后视觉阶段是否带来闭环收益。前一轮[冻结ResNet结果](libero-pretrained-vision-control-results-2026-11-14.md)为16/120、较CNN增加6.67个百分点，未过原10个百分点门；该结论不改写。本轮控制臂重新训练、重新评估，不能把旧16/120当新控制，不能从旧u500追加500步。

本协议科学冻结与角色范围声明均不授予执行权限。当前任务只准备代码、协议和合成验证，不读取或SHA研究HDF5、真实权重、旧checkpoint或初态，不训练或运行物理环境。后续须另建本版本训练执行绑定；闭环还须本轮新的source初态授权及评估执行绑定。上一轮用户对task4/7初态0..19的授权只适用于上一轮，不能转用。

## 两臂及唯一主动变化

| 项目 | `frozen_resnet18` | `layer4_conv_finetune` |
| --- | --- | --- |
| 初始卷积及BN | 相同官方ImageNet-1K ResNet18 V1 | 同左，逐位一致 |
| stem、layer1..3 | 全部冻结 | 全部冻结 |
| layer4五个卷积权重 | 冻结 | 开放梯度并训练 |
| 全部BN affine、running mean/var/counter | 冻结、eval | 冻结、eval |
| projection/fusion/MLP/head初始状态 | 同seed逐位一致 | 同左 |
| 总参数 | 12,689,864 | 12,689,864 |
| 可训练参数 | 1,513,352 | 9,901,960 |

候选额外开放8,388,608个参数，仅以下五个tensor：`visual_encoder.backbone.7.0.conv1.weight`、`7.0.conv2.weight`、`7.0.downsample.0.weight`、`7.1.conv1.weight`、`7.1.conv2.weight`（后四项同一前缀）。这是训练范围受限，不能称参数高效微调或轻量适配。两臂模型结构、全部初始张量、输入、目标、Adam超参数及批次固定，主动变化为这五个tensor的可训练掩码；训练容量、Adam状态和实际计算成本随之增加，不能宣称两臂FLOPs、耗时或显存相同。

完整Policy仍为双128×128 RGB、joint7/gripper2、512×4×4展平投影128、fusion256、原CNN-only逐帧MLP256、七维tanh动作。无GRU、语言分支、增强、resize/crop、新loss或动作后处理。两臂使用同一公共文件`resnet18-f37072fd.pth`，SHA和来源沿用已记录的公开pin；公共文件缺失的BN计数按Torch兼容加载规则置int64零并保持。构造器不隐式联网。

BN的eval状态与autograd是两种机制，eval不会自动关闭卷积梯度，见[Torch 2.4.1官方autograd说明](https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/autograd.rst)。实现通过requires_grad限定权重、通过eval固定BN；仅冻结前缀在no_grad中计算。缓存输出之后要进入训练图，因此构造缓存不使用inference_mode。

## 数据、训练与缓存

task4/7 × seeds20261106/07/08 × 两臂，共12个新Policy。每个从相同初始化独立训练恰好500次，只保存500终点，不选checkpoint。沿用source各自meta_support demos0..29，causal obs[t−1]→action[t]、t≥1。source meta_query、其他source任务和validation/H/qualification/reserved/final及其初态均封存；不做元训练、辅助头搜索、SPT/Knowledge更新或Module注册。

两臂共用原训练内核：Adam lr3e−4、betas(.9,.999)、eps1e−8、weight_decay0、amsgradFalse；batch8、sequence/stride10，uniform masked七维MSE，global gradient clip norm1。采样种子seed+task×10000+777，同task/seed的500批身份完全相同。候选卷积也进入同一个全局梯度范数，可能改变裁剪频率与head的有效步长；逐步保留裁剪前范数和loss，并报告各模型裁剪次数，不临时改学习率或裁剪规则。

两臂均只缓存冻结到layer3的`float32 [256,8,8]`特征，禁止旧`[512,4,4]`最终卷积缓存。每步对双视图重算layer4，候选保留其autograd；head/projection正常训练。缓存逐项复制、detached，交付batch另行collate，训练前后摘要相同。每个模型第一批以原像素输入核对缓存动作，绝对差≤1e−5，仅作实现核对，不是成功判据容差。合成测试另检查卷积梯度逐位一致。

HDF5逻辑guard沿用两文件、demos0..29、四个公开观测数组/actions白名单，拒绝attrs、simulator state、action0及query decode。整文件SHA会读全容器字节，不声称物理字节隔离。CPU缓存最多每task原始2GiB+prefix2GiB；按task依次处理，不同时保留两个task。记录实际cache字节、Torch allocator峰值和耗时；不将allocator峰值说成完整host RSS/渲染VRAM。

## 未来闭环与预算

若取得本轮授权，才开放两task已用candidate_nonregression初态0..19，每模型20条fresh episode，共240条、120对；不读取其余数值行，不使用pickle/torch整体初态反序列化。每模型新adapter，环境seed20261022；恰好5个零动作稳定化，然后最多300个Policy动作，native horizon305。同task/初态在两臂和三个seed的reset公开摘要严格一致。仅使用原生success/time_limit，不改成功条件。

| 工作 | 训练阶段硬上限 | 闭环阶段硬上限 |
| --- | ---: | ---: |
| Policy训练前向 / backward / Adam更新 | 各6000 | 各0 |
| 模型构造 / 保存终点 | 14 / 12 | 12 / 0 |
| source窗口 / 完整源文件SHA | 1100 / 2 | 各0 |
| HDF5 opens / slices | 4 / 5500 | 各0 |
| prefix前向 | 302（含24个像素核对分支） | 144000 |
| layer4前向 | 12024（含24个核对分支） | 144000 |
| 像素/缓存Policy核对 | 12 | 0 |
| 初态数值行 | 0 | 新授权后40，否则0 |
| adapters / resets / episodes | 各0 | 12 / 240 / 240 |
| Policy前向 / 动作 | 各0 | 各72000 |
| 零动作 / 受控native step | 各0 | 1200 / 73200 |
| 内部执行计时 | 1800秒 | 7200秒 |

prefix上限：1100窗口按两个task、每8窗口分批，双视图最多278次；12模型各两个原像素分支再加24。layer4为6000×2+12×2。每个阶段账本将另一阶段操作cap置零，先准入再执行、attempted/completed分记。物理GPU1、CUDA逻辑0/EGL1、单线程、确定性、TF32关闭，版本固定在新配置中。时间为内部阶段计时，不包括独立审计、安装或公共预训练成本；可能在阶段开始的输入核对上另花时间。

任何预算超限、非有限数、梯度掩码错误、缓存/BN变化、配对失败或技术错误立即停止，保留failure和partial，不自动重试、补样本、续训、扩域或换设备。新results目录只写一次，拒绝覆盖。完成训练后须先过独立训练核对，再可读获准初态；独立核对仍会再次读取新checkpoint和已准入公共权重，但不重训、不重跑物理环境。

## 判断与后续分流

事前继续门保持：新微调臂减新冻结臂的两task/三seed等权成功率增益≥1/10；每task三seed均值、每seed两task均值均≥0。全部240身份齐全才能用整数/Fraction原样判门，零容差。展示每个task×seed、全部120配对四格以及冻结成功保留率（分母为零记null）；保留率为描述指标，无事后添加/删减门，不把总体提高描述为所有旧场景无损。

通过才准备全部5个source的后续验证，不能自动宣称已经通过全source或启动正式迁移；未通过只结束本受限微调版本，LIBERO仍是主环境，不转Continual World。新臂若有收益，支持在该配置/预算下开放视觉权重的实用效果，不能独立证明视觉瓶颈、预训练因果或未见泛化。开发初态重复暴露，不作盲测、显著性或正式SPT迁移主张。

独立审计检查所有12模型/240身份、每模型500个实际optimizer state、完整批次和裁剪记录、不可训练tensor及BN逐位不变、候选仅五卷积可变化、输入角色和缓存、严格reset配对、动作/原生终止、实际计数及继续门。不从公开轨迹反推全部接触谓词，不将合成流程当研究结果。FOMAML仍为Skill Evolution唯一主方法；未来A/B/C需要另冻三臂一致的后端和数据角色协议。

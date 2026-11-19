LIBERO首轮新视觉后端配对闭环已完成（2026-11-14）：240条fresh episode及独立审计通过，[完整结果与范围](libero-pretrained-vision-control-results-2026-11-14.md)。CNN为8/120（6.67%），冻结ResNet为16/120（13.33%），提升6.67个百分点；task4为0→7/60，task7为8→9/60。两个task及三个seed均值未退化，但未达到原10个百分点扩大门，不自动扩大到全source或SPT迁移。仅解码本轮获准的task4/7初态0..19共40行，其余数据继续封存；500更新/5步稳定化/原loss保持。LIBERO继续为主环境，不开展Continual World；下一候选准备围绕受限视觉适配，尚未执行。下方保留此前阶段记录。

# 继续LIBERO：控制后端改进与闭环开发对照（2026-11-13）

最新执行进展：[12个Policy真实训练与独立核对](libero-pretrained-vision-control-training-2026-11-13.md)已经完成，共6000次实际更新，60项定向检查通过。科学协议和556项执行依赖已绑定；240条真实闭环尚待source开发初态范围确认，当前没有闭环收益结论。以下保留候选选择与执行前范围说明，实际科学冻结以[固定协议](libero-pretrained-vision-control-protocol-2026-11-13.md)及新配置为准。

**用户明确继续LIBERO，不开展Continual World。此前由“现有候选未选出”推到切换环境的安排撤回。** 旧G未通过和各局部候选的历史结果保留；它们没有检验全部可选后端，不能作为排除LIBERO的依据。本页取代此前去留评审与阶段计划中的环境切换和停止LIBERO研发安排。

当前目标是先取得可复核的完整Policy闭环改善，再用同一个后端公平比较SPT更新。基础控制候选失败时，结束该具体版本并在LIBERO内根据证据选择下一项实质改进；不自动将环境退出作为后续动作。当前500 updates、稳定化5步、原均匀七维masked MSE及正式数据封存保持。

## 第一项新候选

比较当前从头训练的双视角CNN-only控制器，与**冻结ImageNet-1K ResNet18视觉骨干＋可训练空间投影＋原fusion/逐帧MLP/七维tanh动作头**。每视角保持128×128，保留ResNet的512×4×4空间特征，再投影到原128维；共享同一个骨干。使用ImageNet通道mean/std，不执行官方分类默认的224中心裁剪，避免改变机器人视野。

假设是：在500次适应更新内，复用预训练视觉特征可能比从头学习当前视觉骨干更有效。它是新的完整控制后端对照，不是旧GRU/history、loss或257维辅助头的重复，也不把视觉指定为已经证实的唯一根因。

官方依据与实际采用范围：

- [Torchvision 0.19 ResNet18](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet18.html)提供`IMAGENET1K_V1`权重及归一化定义。本实验采用其卷积权重与通道归一化，128分辨率与空间投影是本项目的设计，并非照搬官方分类预处理。
- [robomimic预训练表征文档](https://robomimic.github.io/docs/tutorials/pretrained_representations.html)提供冻结视觉骨干的集成方式；不以其文档或其他任务成绩推断本候选在LIBERO必然有效。
- 本地LIBERO的`libero/lifelong/models/modules/rgb_modules.py`已有ResNet与空间投影模块，说明这类视觉编码与机器人策略接口可组合；本候选不称为官方LIBERO基线的完整复现。

外部预训练、总参数量和可训练容量均与旧CNN不同，必须明确报告。500指本轮适应更新，不能宣称总学习数据或总算力相同；若进入A/B/C，三个臂都必须使用同一个选定后端。基础控制改善不等于FOMAML优于普通预训练。

## 首轮具体开发范围（待执行协议，不是结果）

- 固定source task4/7，seeds20261106/07/08，两臂，共12个完整Policy，各500更新，合6000更新。两任务是已知困难source探针，不是未见任务；不据结果换任务或seed。
- 训练只用各自meta_support demos0..29的causal obs[t−1]→action[t]、t≥1；meta_query保持封存。固定Adam lr3e−4、batch8、sequence/stride10、clip norm1及原loss；全部非视觉参数初始化和批次序列配对。
- 若获准使用source开发初态，每模型candidate_nonregression 0..19，共240条fresh闭环；每条最多300个Policy动作、先5步零动作。最多72000个Policy动作、1200个稳定化动作，native step上限73200；每次使用同一环境seed20261022。
- 两臂都在同一当前栈执行；不拿旧成功率替代新对照。全部模型、场景及配对结果保存。候选骨干权重和BN统计冻结，空间投影、fusion/MLP/head可训练。
- 拟定开发继续门：两个task等权、三个seed等权的成功率提升至少10个百分点；每个task的三seed均值及每个seed的两task均值均不退化。属于已用source场景上的描述性开发门，不作正式泛化或Module资格判断。必须在研究执行前冻结，不能运行后修改。
- 通过后扩大到全部5个source任务核对护栏，再设计同后端A/B/C迁移比较；未通过则记录该后端版本的结果，并在LIBERO内评估下一项实质改进。

训练权重、公共预训练缓存与全部新证据进入个人持久盘。使用新协议、代码绑定和输出路径；旧科学配置、旧SHA和旧结果不改写。尚未实现或尚未核验的执行门不写成已通过。

## 已完成的具体工作与当前状态

新增可选模型模块`src/algorithms/libero_frozen_resnet18_policy.py`，不修改旧Policy默认值。仅从显式SHA绑定的本地文件加载视觉权重，构造模型不联网；冻结卷积/BN，保持原七维接口。专项测试使用发明的ResNet状态和RGB，检查非视觉参数配对、step/sequence一致性、RGB缩放、真实梯度更新及骨干/BN不变、错误SHA在反序列化前拒绝。

公共权重已按官方固定URL下载至`cache/torch/hub/checkpoints/resnet18-f37072fd.pth`，46830571字节，完整SHA256为`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。这是公开文件传输核验，不是研究训练或控制结果；未读取封存的研究参数/数据以选择权重。

source开发初态的访问范围需与此前“初态封存”要求明确对应。闭环若未开放，不能用离线loss冒充本页的成功率门。本轮尚无新闭环成绩，正式validation/H/qualification/reserved/final和source meta_query保持封存。

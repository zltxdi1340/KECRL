# LIBERO新视觉后端：source配对闭环执行记录（2026-11-13）

**240条配对source闭环已经完成并通过独立审计；跨日结果见[2026-11-14完整报告](libero-pretrained-vision-control-results-2026-11-14.md)。** CNN为8/120，ResNet为16/120，提升6.67个百分点，未达到原10个百分点扩大门。下面保留本次授权与启动时范围；LIBERO继续为主环境。

用户本轮确认：“确认本轮仅开放task4/7已用过的source开发初态0..19用于这次评估”。[授权记录](../configs/libero-pretrained-vision-control-source-initial-authorization_v1.json)仅覆盖两个task的candidate_nonregression前20行；[新的评估绑定](../configs/libero-pretrained-vision-control-evaluation-binding_v1.json)引用原训练绑定、同一科学配置、556项依赖以及该授权。原训练绑定和旧科学文件保持原字节。

按[固定科学协议](libero-pretrained-vision-control-protocol-2026-11-13.md)评估[已完成独立训练核对的12个500终点](libero-pretrained-vision-control-training-2026-11-13.md)：task4/7×三个seed×scratch_cnn/frozen_imagenet_resnet18，每模型20条fresh episode。每集先5个零动作，最多300个Policy动作；整个评估最多72000个Policy动作、1200个稳定化动作、73200个受控native step。

两个初态容器各只数值解码0..19，合40个source初态；完整文件SHA覆盖容器全部字节，不宣称物理隔离其余行。其余source初态行、source meta_query、validation、H、qualification、reserved、final不开放。本次不重新训练、不挑checkpoint、不把训练loss或旧成功率当作当前闭环结果。

全部240条完成后，由已绑定的独立核对器重算模型/场景身份、配对初始观察、成功数、原生终止记录和实际预算，并按原Fraction门判断是否扩大source开发。无论本候选结果如何，LIBERO保持主环境，不转Continual World。

本轮新日志在`logs/libero-pretrained-vision-evaluation-20261113-v1/`；逐条新轨迹和结果进入`results/libero-pretrained-vision-control-20261113-v1/`。模型、数据和详细轨迹保留个人数据盘。

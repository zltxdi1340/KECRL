# LIBERO预训练视觉后端：首轮source控制开发协议（2026-11-13）

科学范围固定为source task4/7、seeds20261106/07/08、`scratch_cnn`与`frozen_imagenet_resnet18`两臂。每臂每task/seed独立适应500次，共12个Policy/6000更新。所有12个终点完成并经训练核对后才允许闭环；只保存500终点，不选checkpoint，不做元训练、辅助头或活动SPT更新。本协议是开发对照，不是正式迁移或Module资格试验。

控制采用原CNN-only配置、双128×128 RGB、joint7/gripper2、fusion256、逐帧MLP256及七维tanh动作。候选仅替换视觉编码器为官方ImageNet-1K ResNet18卷积骨干（冻结参数与BN统计），512×4×4网格展开后可训练投影到128；其余初始参数在同seed两臂逐位一致。候选可训练参数1513352、冻结参数11176512，控制可训练参数821704；因此比较完整后端的实用效果，不单独归因预训练或宣称总学习资源相同。

公共权重固定`https://download.pytorch.org/models/resnet18-f37072fd.pth`，46830571字节，SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。保留原128视野，无resize/crop/增强；RGB映射到[0,1]后按mean[.485,.456,.406]/std[.229,.224,.225]归一化。公共文件没有序列化BN计数，按PyTorch兼容加载规则置int64零并全程保持零。模型构造只读绑定本地文件，不隐式联网或更换权重。

训练仅各task的meta_support demos0..29，causal obs[t−1]→action[t]、t≥1；source meta_query不解码。Adam lr3e−4、betas(.9,.999)、eps1e−8、weight_decay0、amsgradFalse，batch8、sequence/stride10，gradient clip norm1，原uniform masked七维MSE。采样种子seed+task×10000+777，两臂的500个批次身份相同，全部轨迹保留。冻结骨干的特征可按同一source顺序缓存；每个候选首次批次做原像素/缓存动作绝对1e−5核对，非主成功门容差。

HDF5只允许两个绑定文件、demos0..29及四个公开观察数组/actions；不读attrs、init_state、action0或其他任务。整文件SHA仍会物理读取未投影字节，不声称物理隔离query字节。原数据、配置和历史绑定不改写。

闭环要求单独的用户source初态授权记录并绑定；授权缺失则在初态文件读取前拒绝。请求范围仅两task的已暴露candidate_nonregression 0..19，每模型20条fresh episode，共240条。源码/数据角色和初态容器SHA/header保持原值；使用选择性行reader，不反序列化其余30行。不能以本协议的范围声明冒充用户同意。正式validation/H/qualification/reserved/final及其初态始终封存。

两臂同一当前CPU/CUDA/EGL栈，物理GPU1、单线程、确定性、TF32关闭；Python3.10.22、Torch2.4.1+cu124、Torchvision0.19.1+cu124、NumPy1.22.4、h5py3.11.0、robosuite1.4.0、MuJoCo3.2.3。每模型新adapter，环境seed20261022，初态顺序0..19，稳定化恰好5个零动作，再至多300个Policy动作；native horizon305。相同task/state的reset公开观察摘要在全部seed/两臂严格一致。只用原生success/time_limit结果，不以影子几何或代理误差改判成功。

| 工作 | 首轮硬上限 |
| --- | ---: |
| source窗口 / HDF5打开 / 数组切片 | 1100 / 4 / 5500 |
| 源文件完整SHA / 模型构造（训练阶段） | 2 / 14 |
| 冻结骨干调用 / 像素缓存核对 | 278 / 6 |
| 训练前向 / backward / Adam更新 | 各6000 |
| 保存终点 | 12 |
| source初态数值行 | 获授权后40；之前0 |
| adapters / resets / episodes | 12 / 240 / 240 |
| Policy前向/动作 | 各72000 |
| 稳定化零动作 / 受控native step | 1200 / 73200 |
| 每task原始cache / 特征cache | 2GiB / 1GiB |
| 训练阶段 / 闭环阶段时间 | 1800s / 7200s |

各操作先检查计数再执行，尝试/完成分别记账；提前成功记录实际动作数。技术问题保留partial和failure，不自动重试、补样本或换设备。全部文件只写新的个人盘results目录，拒绝覆盖；不保存私有初态向量和live simulator state。

开发继续门预先固定：两个task等权、三个seed等权的候选−控制成功率≥1/10；每个task的三seed均值、每个seed的两task均值均≥0。以整数成功数/Fraction判门，不加epsilon、不作未见泛化或显著性主张。通过才扩到全部5个source任务检验；未通过只结束该后端版本，在LIBERO内据证据设计下一候选，不自动退出环境。

独立核对读取新checkpoint、训练轨迹及新episode文件，检查500更新、配对批次、完整12模型/240身份、冻结卷积与BN、原始公共权重映射、reset配对、动作边界、终止和实际计数，独立重算成功率与继续门。它不重训或重新执行物理环境，也不声称由公开轨迹重算全部原生接触谓词。训练与评估各自的记录及核对结果分别保存。

实际执行还须有本协议对应的新科学配置、代码哈希绑定、完整合成验证和所需角色授权；本页单独不授予数据访问。未来A/B/C若采用该后端必须三臂一致，基础控制改善不能代替FOMAML迁移证据。

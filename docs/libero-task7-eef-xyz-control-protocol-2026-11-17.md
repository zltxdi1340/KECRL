# task7缓存EEF XYZ配对控制：模型v2执行协议（2026-11-17）

承接[唯一候选选择](libero-task7-eef-xyz-model-v2-selection-2026-11-16.md)及[完整36条几何读出](libero-postgrasp-singlepass-geometry-v2-results-2026-11-16.md)。用户已批准本轮训练、22状态交接，及过继续门后的120条完整source开发评估。本协议在新研究载荷读取前冻结；科学配置为`configs/libero-task7-eef-xyz-control_v2.json`，训练/交接/full分别新绑，历史产物不改写。此文是方案，不预告实验有效。

## 假设与唯一变量

检验显式笛卡尔自身位置能否降低500更新内学习关节到末端映射的负担，改善搬运后水平对准与下降的协调。关节角原则上已包含运动学信息，几何归因并未证明表示是唯一根因。

两臂均为相同268维融合输入的frame MLP：`zero_xyz_mlp`新增三通道恒0，`cached_xyz_mlp`输入当前公开缓存EEF XYZ。两臂同参数量、完整初始state_dict、seed、逐更新batch、视觉缓存及损失。保留原265列权重/偏置和其他网络；新增3列由seed+9100001固定初始化。原265维MLP的3/22与GRU的4/22只作历史参照。本轮需训练6个新模型、评估44条，不能把旧控制臂冒充新匹配控制。

ResNet18/BN冻结；同公开ImageNet权重、双128×128 RGB、512×4×4卷积网格、共享128维投影、joint7/gripper2、原frame MLP及七维动作解释。无视觉解冻、GRU、动作分块、新夹爪头、物体/盘坐标输入或几何监督。

## 通道、尺度与采样时刻

新增XYZ以世界坐标米、float32原值拼接，与joint/gripper使用同一“不额外标准化”的规则。不做示教统计归一化、32维投影或比例扫描；这是预先选定的最小干预，若失败不能排除其他尺度方案。

本地官方`third_party/LIBERO/scripts/create_dataset.py`把缓存`obs["robot0_eef_pos"]`写入`obs/ee_pos`。训练仅从这个字段读取；闭环适配器仅复制同一次原生step/reset返回的缓存`robot0_eef_pos`，不读`site_xpos`、不强制刷新观测。数据生成代码只支持通道来源判断，不冒充逐示教的高精度原生时刻证明。

实际训练先逐项校验demos0..29的字段shape/dtype，缺失或非有限立即拒绝；禁止用`ee_states`、私有原生位姿或补造重放静默替代。所有字段保持既有因果t≥1窗口：观察[t−1]→动作[t]，seq10/stride10，尾部零填充且valid_mask排除填充损失。观测XYZ与图像、关节的存储索引相同。

## 固定训练与读入预算

仅task7 meta_support demos0..29；432既有窗口。3seed=20261106/07/08，每臂每seed恰好500次实际Adam更新；batch8、uniform masked七维MSE、Adam lr3e-4/betas(.9,.999)/eps1e-8/weight_decay0/amsgrad=False，梯度总范数裁剪1。CPU单线程、CUDA隐藏、固定确定性配置。保留完整500步batch ID/loss/梯度范数和optimizer状态；仅保存last500，不用闭环结果选择checkpoint。

训练上限3000前向/反向/optimizer、6checkpoint、7模型构造（含1缓存模型）、6像素/缓存前向核对、150冻结编码器调用；HDF5仅1源文件SHA、2打开、550窗口、3300切片，其中EEF≤550切片/16500坐标；公开权重各1次SHA/load。实际432窗口是2592总切片，EEF坐标按有效行精确计数。原始缓存≤2GiB、卷积缓存≤512MiB、wall≤1800秒。参数、BN/冻结状态、Adam500、零XYZ三列无更新、完整初始权重与batch相等均由独立进程审计。逐步另存XYZ/关节输入梯度范数；XYZ小缓存单独持久化一次供独立审计重算哈希/填充/有效行，不再读取HDF5。模型权重保存在`checkpoints/libero-task7-eef-xyz-control-20261117-v2/`；事件、XYZ小缓存及审计在`results/`，执行日志在`logs/`。

访问guard限制具体文件、demo和6个字段（两RGB、joint、gripper、ee_pos、actions），拒绝属性、私有状态、其他分区和动作t0；在工作前计预算。这是研究访问契约，不是操作系统安全沙箱。

## 交接与条件性完整开发评估

严格沿用20261115-v1的22个交接来源，含unchanged和confirm5两种旧前缀，各对应两臂。每次新环境按reset0..target、每次稳定化5步，重放原applied动作至冻结anchor；之后执行新臂动作，不再施加confirm5规则。两臂都是逐帧，仍保留各9次被丢弃的前缀预热预测以维持旧执行口径。300是整条episode上限，非交接后再加300。

旧归档已存joint/gripper及reset摘要必须精确匹配；当前两臂的完整前缀RGB/joint/gripper与新增缓存EEF必须逐位一致。旧记录无EEF，不能宣称已证明旧EEF逐位相等。RGB出现分歧按本次严格门技术停止，保留失败/部分轨迹，不自动近似放行或重试。

交接上限：44环境/episode、556reset、2780零动作、4700前缀动作、8500新控制动作、15980物理步；396预热+8500预测，冻结编码17792次、时间行8896；初态容器一次只解码0..18这19行；13756公开观测/41268 EEF坐标，原始RGB≤1352269824字节，wall≤1800秒。交接不得访问HDF5/新训练。

继续门以新匹配控制为参照：候选净增≥4/22，且每seed成功数不退化。完整44条齐全并独立审计通过才计算准入，不能只评18个共同失败。记录全部逐状态成败、新增成功和丢失成功，不将此小样本门当显著性证明。

仅过门才新建full授权/绑定并评估120条（每臂3seed×已用source初态0..19）；6环境、120reset、600稳定化、36000模型动作、36600物理步上限，wall≤3600秒。下一继续门为净增≥6/60且每seed不退化。未过交接门则full为0条，不建立full绑定，不把本轮旧授权扩张到其他任务或封存初态。

所有阶段独立审计原始观察归档、前缀/动作/原生终止、账本和整数门，不额外启动环境/模型前向。技术失败留现场，研究阶段无自动重试；资源等待最多3600秒，不占用其他用户的繁忙GPU。实际计数、运行时间和参数量以审计产物为准。

## 结论与封存边界

这是已用task7 source开发状态上的表示干预诊断；不足以称正式Skill Evolution迁移收益、完整KECRL或Module资格。Policy500、稳定化5、现loss保持；meta_query、validation/held-out/qualification/reserved/final和其他任务继续封存。无Knowledge/SPT更新、Module注册、部署或checkpoint择优。结果不得把单臂偶然增益写成已验证根因。

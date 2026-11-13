# LIBERO来源candidate有限几何采集结果（2026-11-07）

**冻结范围内的120条新短轨迹已完成，独立审计通过。** 本次补齐了动作、末端/物体几何及碰撞接触代理的同边界记录。观察到的模式依赖任务：task0的B臂在首次多步close时仍常离物体较远；task3的B臂则常先发单步close脉冲，随后在更近处形成多步close。现有证据不足以统一提前/延后close，也不足以确定视觉、动作头或数据覆盖中的单一根因。

本次是diagnostic，不改变[G未通过的准入与停止决定](libero-formal-G-results-2026-11-06.md)。保持500 updates端点、5步稳定化和原loss，0新训练；H未开展，validation、held-out、qualification、reserved和final evaluation均无新增payload访问。不更新SPT或Knowledge，不注册Module。

## 执行设置与实际预算

科学设计见[冻结协议](libero-source-candidate-geometry-capture-protocol-2026-11-06.md)及[科学绑定](../configs/libero-source-candidate-geometry-capture-freeze-binding_v1.json)；实现隔离、有界reset和独立审计见[执行契约](libero-source-candidate-geometry-capture-execution-contract-2026-11-07.md)及[执行绑定](../configs/libero-source-candidate-geometry-capture-execution-binding_v1.json)。两层绑定在真实payload读取前完成，执行层固定355项源码/输入引用。

固定task0/1/3/4/7、seed20261106/20261107/20261108、A_original/B_fomaml现有30个500-update端点和candidate_nonregression全局初态0..3，共60对/120条。A为G阶段原始初始化适应臂，B为G阶段FOMAML候选初始化适应臂；本次只执行其已有权重，不再训练生成器或Policy。C没有本协议所需的source端点，不临时补训。每条使用全新adapter、一次底层reset、5个零动作稳定化，最多150个Policy动作，原生任务成功可提前终止。环境seed和每条构建前Python/NumPy/Torch seed均为20261022；原生300步Policy horizon保留。

本次使用已用过的source candidate初态，不能称为未见泛化。新鲜adapter生命周期也不同于旧G和validation的复用环境；不能给旧validation36对补上“几何真值”，也不能合并成功率。

| 资源 | 冻结上限 | 实际完成 |
|---|---:|---:|
| 固定checkpoint读取/Policy对象 | 各30 | 各30 |
| 初态容器/数值行 | 5/20 | 5/20 |
| 新adapter/底层reset/短轨迹 | 各120 | 各120 |
| Policy前向/动作 | 各18000 | 各17127 |
| 稳定化零动作 | 600 | 600 |
| 原生控制步 | 18600 | 17727 |
| 物理积分步 | 465000 | 443175 |
| 独立影子MjData分配 | 120 | 120 |
| 影子位置计算/几何行/公开状态行 | 各18120 | 各17247 |
| bowl相关碰撞记录，含正margin | 4638720 | 301138 |
| 活动时间 | 3600秒 | 473.614秒 |

全部资源attempted=completed，无技术失败、重试、替换或额外真实smoke。5个初态容器仅将0..3行数值解码；整容器字节校验与数值解码分开记录，不用pickle/torch整库反序列化。HDF5读取、旧原轨迹读取、新优化更新和封存角色payload读取均为0。

运行于物理GPU1，UUID `GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff`，逻辑设备cuda:0，float32、deterministic、TF32关闭；Python3.10.22、PyTorch2.4.1+cu124、MuJoCo3.2.3、robosuite1.4.0。进程起止为北京时间2026-11-07 12:56:04–13:03:59，进程总墙钟475.431秒，约7分55秒；活动计时不包含全部启动/退出开销。实际命令、环境变量、占用快照和版本均有provenance记录，无设备回退。

## 几何隔离与独立核对

[observer](../src/environments/libero_geometry_observer.py)在独立MjData中复制冻结字段，每个保存边界只做一次mj_fwdPosition；不额外推进或forward原生环境，不刷新原Policy观测。几何只进入评估sidecar，不进入Policy，不归档原始全场景私有状态或RGB，不读取/输出接触力数值。末端为grip_site，bowl/plate为root body原点，世界坐标单位米、旋转为3×3矩阵；距离不是指尖到物体表面的净间隙。

[有界adapter](../src/environments/libero_geometry_capture_adapter.py)绕过第三方无限reset重试循环，每条只调用一次底层reset；三个索引角色分别记录：global和selected-bank-local为0..3，注入单行后的adapter-local恒为0。observer期间原生step/forward调用被拒绝，实际拒绝计数0；17247个边界的原生存储字节指纹和Policy观测摘要前后相同。30份checkpoint在首个环境创建前核验文件与张量hash，使用后再次核对权重不变且无梯度。

每条reset后、首个Policy动作前保存真实编译XML hash、site/body/geom名称和ID、坐标/维度。相同task/global初态跨六个模型比较，20个参照、100次公开reset、几何reset和编译映射精确比较全部通过。

[独立审计器](../experiments/audit_libero_source_candidate_geometry_capture.py)以标准库运行，禁止导入Torch、NumPy、MuJoCo、LIBERO、robosuite或采集实现。它独立核对120条身份、60对完整性、原始文件hash、距离/旋转/时钟、碰撞分组、删失、计数及隔离凭据；3.250秒完成，0新Policy/环境/初态/checkpoint调用。它没有重新求解物理接触，结论限于源码、保存字段及隔离凭据的审计。

原Policy使用Observable采样缓存，当前native qpos和缓存输入不必相等。已保留原输入，并标注差值：所有保存行的joint L2中位数0.000882（关节角坐标），gripper L2中位数0.00000161米。差异包含采样与数值表示因素，不能据此指定精确延迟或闭环根因。执行前合成测试发现的MuJoCo非原生NULL视图问题及缓存口径修正已记于执行契约；修正均在绑定前完成，没有因此增加真实采集次数。

## 固定事件定义下的结果

close指第七维动作>0，open指<0，0单独记录。first multi-close指回顾性识别出的首个连续至少两个正号动作段，不等于抓稳。action k的pre-state是k−1；close起点时刻取state[k−1].sim_time，首次双侧接触代理取state[j].sim_time，不能直接以(j−k)换算。接触代理仅表示保存边界上两侧手指碰撞组都与bowl有dist≤0记录，不代表夹持力或稳定抓取；正margin记录不算touch。

120条中，104条观察到close、16条未观察到；67条出现过双侧接触代理，reset时均没有该代理。104条close轨迹中，43条在close起点及之后的保存边界没有该代理；这不等于43条从未接触，也不能排除控制边界之间的瞬态接触。67与43不是互补计数。

首次close前的末端至bowl三维距离，在104条可观察轨迹上均值0.1657米、中位数0.0834米、范围0.0220–0.4208米；首次多步close前的距离在103条上均值0.1270米、中位数0.0768米。20条首close是完整单步脉冲；47条首close段右删失，不能从观察长度推断实际保持时长。没有close或对应事件的数据保持缺失，不补0、不向前填充。

首次close和首次双侧接触代理都可见的66条轨迹中，时间差中位数0.525秒，范围−5.15–2.95秒；负值表示首次代理早于首次close，不改写成“close后接触”。该时间量跨任务混合，仅描述已观察事件，不代表统一反应时间。逐条XYZ/XY/有符号Z、close前距离变化、事件±5邻域、物体高度和bowl至plate距离均保留在审计表和sidecar中。

下表每格12条，跨三个seed，保留全部任务。距离列只对观察到首次close的轨迹求中位数；“未见后续代理”以close轨迹为分母，包含close起点边界。完整30行task×seed×arm结果见CSV。

| task | arm | 150步内原生目标达成数 | close数 | 任意时刻双侧代理数 | close起点及以后未见代理数 | 首close前距离中位数（米） |
|---|---|---:|---:|---:|---:|---:|
| 0 | A | 4 | 11 | 7 | 4 | 0.0833 |
| 0 | B | 1 | 12 | 1 | 11 | 0.3794 |
| 1 | A | 3 | 11 | 12 | 0 | 0.0489 |
| 1 | B | 4 | 12 | 11 | 2 | 0.0511 |
| 3 | A | 0 | 12 | 5 | 7 | 0.3747 |
| 3 | B | 6 | 12 | 11 | 1 | 0.3737 |
| 4 | A | 0 | 8 | 4 | 6 | 0.0795 |
| 4 | B | 0 | 7 | 3 | 6 | 0.0935 |
| 7 | A | 1 | 10 | 8 | 2 | 0.0787 |
| 7 | B | 0 | 9 | 5 | 4 | 0.0941 |

三seed各4条的原生目标达成数如下，顺序固定为20261106/07/08，不按表现挑seed：

| task | A | B |
|---|---|---|
| 0 | 0 / 1 / 3 | 0 / 1 / 0 |
| 1 | 3 / 0 / 0 | 3 / 0 / 1 |
| 3 | 0 / 0 / 0 | 4 / 1 / 1 |
| 4 | 0 / 0 / 0 | 0 / 0 / 0 |
| 7 | 0 / 1 / 0 | 0 / 0 / 0 |

总计19条在150步前缀内达成原生目标，其余101条在观察前缀内未达成；A为8/60、B为11/60，配对四格为共同达成3、仅A达成5、仅B达成8、共同未达成44。这些仅用于记录前缀终止与删失，不能称为完整300步成功率、正式方法优势或qualification。19条达成都出现过双侧代理，但另48条出现代理后仍未在前缀内达成目标，因此代理不足以识别完成或抓稳。

## 支持证据、反例与未解问题

- task0的B臂首次多步close前距离中位数0.3787米，A为0.0799米；B只有1/12出现双侧代理，A为7/12。这支持继续检验该任务的接近与close协调，但没有分离视觉定位、动作映射或覆盖差异。
- task3的B臂8/12首close是单步脉冲。首close前距离中位数0.3737米，而首次多步close前降为0.0557米；该臂11/12出现代理、6/12前缀达成目标。因此“第一次正号距离大就应统一延后close”与这些轨迹不相容。A在该任务首次多步close前仍为0.3734米，不能把两臂的首个正号直接当同一抓取阶段比较。
- task4/7存在没有close、代理只在close前出现、出现代理但未达成目标等不同情况。例如task4的seed20261108两臂均0/4发出close；不能把所有未完成都归为close太早或持续太短。代理之后的运输/放置过程尚需结合连续几何核对，当前不强行标为已抓稳或某阶段失败。

以上是本次已知来源场景中的描述性分解，没有显著性检验、距离阈值、因果结论或checkpoint/seed/方法选择。现有证据没有给出应改loss、加训练预算或改观测的统一方向。

## 验证与证据位置

执行前新增31项测试；连同科学协议与选择性初态读取器，重点测试82项通过；CPU回归1714项通过。编译和环境检查通过。CPU回归排除`tests/*crafter*.py`、`tests/test_spatial_gather_spt.py`、`tests/test_smoke.py`，libero-kecrl缺少可选Crafter依赖，不称为所有环境的全量pytest通过。测试使用合成环境/metadata，不消耗本次真实场景预算。原始测试记录：`/tmp/libero-source-candidate-geometry-execution-software-tests-20261107-v3.json`。

| 产物 | SHA-256 |
|---|---|
| [科学配置](../configs/libero-source-candidate-geometry-capture-diagnostic_v1.json) | `057d98818cf36c23a1dac1dd84d4d742603e92e3360744efa4e8b77ac97d5b7e` |
| [科学绑定](../configs/libero-source-candidate-geometry-capture-freeze-binding_v1.json) | `bafe7eaedbeca92ef6fbae7dfc42def8e373a408483c6aa5b5811a95f1b6bf92` |
| [执行绑定](../configs/libero-source-candidate-geometry-capture-execution-binding_v1.json) | `54ce325293444be2b9e943f8c2d89ae6a142d77f58c610ab7c99bb6b77afb8cb` |
| `/tmp/libero-source-candidate-geometry-capture-20261107-v1.json` | `eda000e69441e63497349a5d9c97d20647532117b00c58f41edeb3274f09225f` |
| `/tmp/libero-source-candidate-geometry-capture-independent-audit-20261107-v1.json` | `144e52a6bfbd188abe819010dade779e76c9734eead5b36ca6c1a32ea07aa8ad` |
| `/tmp/libero-source-candidate-geometry-capture-interpretation-20261107-v1.json` | `546f7b13c4588623737912a8ad6819908002a8941b32ca50c20d06473b84a9b0` |
| `/tmp/libero-source-candidate-geometry-capture-completion-20261107-v1.json` | `56cefa2dab2ed601bba0efb3ea56e0f9e92905d67bc2e00b46accdaae5dea431` |

主结果写入时状态为`complete_capture_pending_independent_audit`，保留其原始字节；后续通过状态写在独立审计和completion中，不回写主结果。过程记录、命令与环境见`/tmp/libero-source-candidate-geometry-capture-process-20261107-v1.json`，审计耗时及零研究调用见同前缀的`audit-process-20261107-v1.json`。

原始120份公开轨迹、120份geometry sidecar和事件日志位于`/tmp/libero-source-candidate-geometry-capture-artifacts-20261107-v1/`。独立审计表位于`/tmp/libero-source-candidate-geometry-capture-analysis-20261107-v1/`：

| 表 | 行数 | SHA-256 |
|---|---:|---|
| episodes.csv | 120 | `6645d87dd2626bae0cc477ef643e946a26ebb26736fca7fc12a39311f91e41b2` |
| pairs.csv | 60 | `90d4ff837924cbe8df051caa949b3bee4eba689c02da51ff9e02fc0a871f590f` |
| event_windows.csv | 1892 | `aa430310656cec414513f29b6f91c74c768ea15b6aae6e9240fe26358d484648` |
| task_seed_arms.csv | 30 | `5f2adf2520b95a0d9ff21d7e49462befb0a2257e9c2d794188080e96ac79ce73` |

另对这批新增记录作只读事件分解，导出`/tmp/libero-source-candidate-geometry-capture-command-runs-20261107-v1.csv`，359行，SHA `1cc9e05ecc3b221f90a58b7c045d350488942fd78a0677d4b1a6cabee802c2da`。这里包含全部正/负/零号最大连续段，不是359次抓取；只读分解没有新模型、环境、影子计算或初态调用，也没有重读旧轨迹。

文档同步后的最终文件/预算核对另存`/tmp/libero-source-candidate-geometry-capture-final-verification-20261107-v1.json`，覆盖355项执行引用、1300个既有保护文件、旧G和36对分析绑定、表格行数及结果谱系；不重跑采集，不改写绑定、checkpoint或历史结果。README、roadmap和500-update计划已同步本次完成状态。

## 下一步

先只读这120条前缀及359个既有命令段，保留全部task×seed组，按“close段—双侧代理—bowl高度/plate距离”对齐。task0与task3作为相反模式的重点核对，同时保留其余任务与删失反例；区分接触前接近/close协调和接触后的未完成过程，不把代理自动当成抓稳或运输阶段标签。完成这一步后再判断是否有理由另冻单变量训练修改。当前保持500 updates、5步稳定化和现有loss，训练预算、G停止规则及封存角色均不变。

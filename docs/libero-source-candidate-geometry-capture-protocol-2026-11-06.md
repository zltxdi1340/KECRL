# LIBERO来源candidate有限几何采集协议 v1（2026-11-06）

**冻结科学方案，尚未采集。** 保持已有500-update Policy、稳定化5步和均匀七维masked action MSE；新增训练更新为0。[配置](../configs/libero-source-candidate-geometry-capture-diagnostic_v1.json)固定30个现有端点及其文件/权重hash，后续最多采集120条新短轨迹、18000个Policy动作。这里只建立观测诊断，不恢复已停止的G方法，不启动H，不更新SPT、Knowledge或Module。

依据：[已有记录缺失几何的核对](libero-formal-G-geometry-record-availability-audit-2026-11-06.md)、[36对反例](libero-formal-G-close-runs-late36-diagnostic-2026-11-06.md)、[G阴性停止决定](libero-formal-G-results-2026-11-06.md)、[既有生命周期复现问题](libero-runtime-native-prefix-failure-trace-diagnostic-2026-10-27.md)。新采集不会补写、替代或被追认为旧G或validation36对的几何真值。

## 1. 待检验问题与解释边界

检查固定Policy首次close前，末端与bowl的相对几何、距离变化及两侧手指的几何接触代理；再检查连续close和首次open附近的这些量。要区分“close之前尚无可见接触代理”“close附近已出现接触代理”等观察模式，不能把这些模式直接命名为可靠抓取、抓空根因或运输失败。

主描述量为首次close的pre-action末端—bowl距离（XYZ、XY和有符号Z）以及首次close与首次双侧几何接触代理的时间顺序。保留原始连续量，不搜索“足够接近”的距离阈值；无接触和未见close均按窗口删失处理。这里没有成功率改善、显著性、因果效应或方法接受假设。

## 2. 固定任务、模型和场景

| 项目 | 固定范围 |
|---|---|
| 来源任务 | 0、1、3、4、7，全部覆盖，不按结果选任务 |
| seed | 20261106、20261107、20261108，按编号取前三个G seed |
| 两臂 | A_original、B_fomaml；使用G阶段现成source端点 |
| checkpoint | 仅update500，共5×3×2=30个；路径、文件SHA和weight hash逐一列入配置 |
| 初态角色 | source candidate_nonregression，global初态0、1、2、3 |
| 配对规模 | 5×3×4=60个场景对，120条短轨迹；保留全部15个task×seed组 |
| 单条上限 | 150个Policy动作，原生成功则提前停止；未成功到上限标为diagnostic_prefix_limit |
| 固定环境设置 | 环境seed20261022、5个七维零动作稳定化、20Hz控制、原生300步Policy horizon/305步总horizon |

已有G在source仅有A/B，不能为凑B/C再训练C。此对照不等于此前validation B/C比较，也不用于跨表合并成功率。任务0/1是table_region，3是movable_object_surface，4是open_container，7是fixture_surface；task4抽屉已经打开，不加入开抽屉目标。五任务共同源物体为akita_black_bowl_1，目标为plate_1，task名称及来源谓词完整列入配置。

任务、seed和初态均已在历史诊断中暴露，编号规则不把它们变成盲选或未见测试集。此前成绩已知，但本次选择不按表现排序，不替换失败seed/场景，也不读取旧逐条成败来决定窗口。

## 3. 生命周期、时钟与Policy信息边界

每条轨迹使用全新adapter，包括同一模型下的不同初态；只加载对应授权初态，local索引与global索引都记录。按task→seed→arm→global初态递增运行，同一个500 Policy只加载一次，每条轨迹重置其运行缓存。新生命周期有意独立，不声称复现旧G中复用adapter的动作前缀。

选择性读取器中的四行局部索引为0..3；向每个adapter注入单行后，adapter自身local索引始终为0。必须分别记录selected-bank local、adapter local和official global三者映射，不能把单行adapter的local0误标为official global0。

每条只允许一次底层reset尝试，不沿用LIBERO wrapper无限重试直到成功的行为；未来执行入口必须提供有界reset包装，技术失败保留并停止，不偷偷补跑。构造前将Python/NumPy/Torch的该次环境RNG设为20261022，不改变已加载权重。必须向adapter注入原生环境和单个授权初态，禁止调用默认构造器中整库get_task_init_states路径。五个source初态容器仅用既有选择性读取器解码0..3，每任务一次，共20行；其余candidate、qualification20..39和reserved40..49不数值解码或缓冲。校验容器时允许既有读取器的字节级SHA检查，必须与数值解码计数分开。validation、held-out和final payload读取预算均为0。

后续入口须使用本诊断单独的source-only访问门控，严格限定上述四行；不修改或放宽旧G/H AccessGate，不把本诊断作为新的正式阶段或H准入。

状态0为5步稳定化后的状态。动作k使用状态k−1，产生状态k；几何状态与同编号公开joint/gripper同步保存。每个存储边界只做一次几何重算，作为下一动作的pre-state复用，不为事件窗口增加动作或补帧。首动作之前、每次动作之后都记录sim.time；控制边界间隔应为0.05秒，容差1e−9，物理子步0.002秒、每控制步25个。偏离则技术停止，不更改时钟定义迁就数据。

Policy保持双RGB、7维joint和2维gripper四个输入；128像素、原归一化、float32、eval模式、确定性动作及关闭TF32保持。几何、物体名、碰撞信息和目标距离只进入独立评估sidecar，不能进入Policy、训练数据、loss、梯度、checkpoint选择或Knowledge。不能为几何同步刷新原生Policy图像、重新渲染或改变已有传感器链；不另存RGB文件。

## 4. 几何观测方案：独立影子数据

静态实现核对发现两个需规避的接口问题：LIBERO的SiteObjectState.check_contact直接返回True；当前robosuite控制循环先forward再积分，未经同步验证不能把缓存位置的读取时刻等同于当前qpos时刻。这是观测实现风险，不能据此认定旧Policy失败的原因。相关依赖源码的实际SHA已固定在配置中，不修改这些历史实现。

未来observer在每条轨迹分配独立MuJoCo MjData，共用只读场景模型；在每个保存边界复制当前qpos、qvel、time、mocap_pos、mocap_quat和eq_active到影子数据，调用一次mj_fwdPosition。只从影子数据取当前姿态对应的位姿和几何碰撞。MuJoCo将该函数定义为位置相关计算，区别于积分或求接触力；具体接口见[3.2.3 API](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-fwdposition)。

不调用运行中环境的额外step/forward，不写原生qpos、qvel、ctrl、act、time、mocap、eq_active或缓存。入口绑定前须用合成测试验证影子数组无别名、原生状态与Policy观测摘要不变、动作前后索引正确。执行时在每个observer调用前后核验原生缓冲区指纹；不能通过放宽容差或额外真实轨迹来补救失败。原始全场景qpos等仅临时用于影子计算，不序列化完整私有状态。

固定记录：

- 末端：robot0.gripper.important_sites中的grip_site，保存解析后的site名称/ID、world位置和3×3旋转矩阵。
- bowl与plate：通过objects_dict的root_body解析，核对obj_body_id，保存world位置和3×3旋转矩阵。这里是body原点，不能称为物体质心或最佳抓取点。
- 手指：解析important_geoms中的left_finger/right_finger，包含pad，去重且与bowl/plate碰撞geom集合分离；名称和ID全部记录，缺失、重复映射或空集合停止，不猜名字、不回填0。
- 接触几何：只保存影子碰撞中涉及bowl的记录，包括geom名称/ID、距离及world接触点，并标明另一侧为左指/右指/plate/其他环境。每状态最多256条；超出停止，不截掉超额记录。
- 派生连续量：eef→bowl的world向量、XYZ/XY距离、有符号高度差，bowl→plate向量及XYZ/XY距离，bowl相对状态0的高度变化。

使用shadow contact dist≤0定义“边界上几何接触代理”，正距离margin记录另保留。MuJoCo的contact数组也可能包含正距离项，因此不能把数组非空直接叫接触；距离和margin语义见[3.2.3 Contact说明](https://mujoco.readthedocs.io/en/3.2.3/computation/index.html#contact)。双侧代理只表示同一保存边界左右两指均有上述bowl几何记录，不证明有支撑力或稳定抓持；正距离也不能被据此断言没有约束力。影子数据未求接触力，不读取或推断force。

这是控制边界的几何重建，不是原生求解器完整接触历史或新传感器真值；不能排除控制步之间的瞬时接触。旋转矩阵用world坐标、row-major保存，距离单位米，不混用body局部坐标或四元数约定。后续必须通过接口与独立审计验证这一实现，再开展采集。

## 5. 后续固定分析

完整保存前缀公开轨迹与独立geometry sidecar，以同一task/seed/arm/global初态/state索引、sim.time和checkpoint hash连接，保留两类文件各自SHA及联合清单。不得给旧G结果增加私有字段；产物放在新的/tmp目录，不复制checkpoint进仓库，也不覆盖任何历史文件。

首次close为第七维第一次正号；连续close段不跨0连接，首次多步段需观察到至少两步后回溯起点；首次open限定为首次close后的第一个负号。终止时未结束的段保持右删失，单步且未见结束不能判为完整脉冲。

对两臂各自首次close、首次多步close及首次open生成固定±5步窗口；去重同绝对步锚点但保留别名。每个锚点均按相同绝对动作步读取A/B，不比较不同物理时刻后宣称处理效应。事件k的主几何量来自状态k−1；接触代理也同时列出pre/post状态。首次代理若在状态0已有，单列为起始已存在；未见事件、reset边界和前缀/成功终止导致的缺失分别记录，不填0、不延长窗口。

close动作k的起始时刻为state[k−1].sim_time，首次接触代理j的时刻为state[j].sim_time；两者相减才是时间间隔，不能直接用j−k当作控制步延迟。close前最多五次距离变化为d[t]−d[t−1]，t从max(1,k−5)到k−1；没有足够前置状态则保持缺失计数。

输出每task×seed的全部配对、事件可见数、连续距离与接触代理时序及均值/中位数。提前成功、到150步尚未成功和技术失败分别列出；150步未成功不能叫300步失败。保留无close、无代理、双臂共同模式和相反模式。不做显著性检验、距离阈值搜索、checkpoint选择、seed筛选或新loss选择，也不把本次结果解释成validation36对或其他环境的根因。

## 6. 固定预算与停止规则

| 操作 | 后续单次执行上限 |
|---|---:|
| 固定Policy/权重反序列化 | 30 / 30次 |
| 新鲜adapter/底层reset尝试 | 120 / 120次 |
| 新短轨迹/场景对 | 120 / 60 |
| Policy前向与动作 | 18000 |
| 5步稳定化零动作 | 600 |
| 原生控制步，含稳定化 | 18600 |
| 原生物理积分子步 | 465000 |
| 公开状态行/几何状态行 | 各18120 |
| 独立影子MjData分配/位置forward | 120 / 18120次 |
| 保存bowl接触几何记录 | 4638720条，且每状态≤256 |
| 初态容器/数值解码行 | 5 / 20 |
| 活动时间 | 3600秒，检查点验证和采集均计入 |
| 新优化更新、HDF5、旧轨迹重读、额外RGB归档 | 全部0 |
| validation/held-out/qualification/reserved/final payload | 全部0 |
| 技术补跑/替换场景/额外实时smoke episodes | 全部0 |

预算是上限，提前成功会减少实际动作和行数。原生物理积分与影子位置forward分别计数；后者不属于Policy训练或环境交互，但不能从计算账本省略。所有资源记录attempted/completed及失败，不只记录成功调用。恢复/重试不在本版额度内。

先验证全部30个checkpoint文件SHA和weight hash，再创建第一个环境；不以缺权重为理由重训。冻结阶段只复制此前已审计metadata中的hash，没有读取任何checkpoint字节。执行绑定还须固定实际GPU UUID/占用、软件版本、实际编译模型/geom映射和原始命令；不自动切CPU或换GPU掩盖复现差异。

任一角色/初态/hash/重置配对/数组隔离/时钟/字段/数值/预算门槛失败，保留部分产物和失败原因，停止本版采集和主汇总，不自动放宽、补样本或续费。无close、窗口内无几何接触代理或到前缀末端未成功是可分析的诊断结果，不是接口故障，也不触发补跑。无正向机制证据也按完整阴性结果结束；不设性能接受门槛，不恢复G准入。

## 7. 本轮冻结验证与下一步

[冻结预检模块](../experiments/libero_source_candidate_geometry_capture_protocol.py)仅使用标准库和既有metadata协议，只有预检模式，没有采集入口。它拒绝来源外任务、不同seed/初态、不同预算/loss/端点及伪装成payload的引用；核对G训练/完成审计链、30个500端点、五个source初态容器metadata和依赖源码SHA。[专项测试](../tests/test_libero_source_candidate_geometry_capture_protocol.py)使用合成metadata、未创建的checkpoint路径及禁止数值runtime导入/读取的guard，验证访问和冻结边界。

35项新测试、100项相关测试通过（1.17s），包含整库默认构造路径和close时钟错一帧的拒绝检查；源码可编译。最初新测试文件的装饰器语法错误在绑定前修复，未进入研究payload读取；原失败说明和两轮软件验证记录均保留。最终日志为`/tmp/libero-source-candidate-geometry-freeze-tests-20261106-v2.log`，软件核验为`/tmp/libero-source-candidate-geometry-freeze-software-checks-20261106-v2.json`。不把这些软件测试计为真实observer或采集验证。

冻结完成后保留独立[科学绑定](../configs/libero-source-candidate-geometry-capture-freeze-binding_v1.json)和/tmp预检记录。该绑定的execution_binding_complete与capture_authorized始终为false；本轮实际Policy/训练/环境/初态/私有状态/影子forward调用均为0。

仅metadata预检命令为：

```bash
conda run -n libero-kecrl python -m experiments.libero_source_candidate_geometry_capture_protocol \
  --config configs/libero-source-candidate-geometry-capture-diagnostic_v1.json \
  --binding configs/libero-source-candidate-geometry-capture-freeze-binding_v1.json \
  --output /tmp/libero-source-candidate-geometry-freeze-preflight-20261106-v1.json
```

独立结束核对记录为`/tmp/libero-source-candidate-geometry-freeze-verification-20261106-v1.json`；它检查任务/seed/端点清单、预算算术、旧文件保护和本轮0采集，不代表运行成功。

下一步实现新的采集入口、隔离几何observer、有界reset与独立审计，完成合成接口测试并单独绑定执行代码、GPU及运行产物路径，然后才执行本协议的120条上限。不能仅因科学预检通过就启动环境，也不需要追加500-update训练。

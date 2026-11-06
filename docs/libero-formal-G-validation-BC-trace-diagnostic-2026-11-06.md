# LIBERO validation B/C 800对公开轨迹只读诊断（2026-11-06）

后续完成（2026-11-06）：[36对较晚close反例的命令段与邻域](libero-formal-G-close-runs-late36-diagnostic-2026-11-06.md)。31对输家首段持续至赢家close，4对单步脉冲，1对先结束两步段；首次多步起点下33对赢家仍晚。原36对全部保留，不据此调整G判断或训练。

**全部800对已分析，G准入未通过的决定保持不变。** 本次比较同任务、同训练seed、同官方初态下的B/C固定500端点，定位共同前20步、close/open命令事件与公开joint/gripper变化。未新增训练、Policy前向、环境重放或数据解封；held-out、qualification、reserved继续封存。

本报告是已知G成绩后的diagnostic，依据[上一轮任务/seed分解](libero-formal-G-task-seed-differences-2026-11-06.md)、[只读定义](../configs/libero-formal-G-validation-BC-trace-readonly_v1.json)和[代码/输入绑定](../configs/libero-formal-G-validation-BC-trace-binding_v1.json)。不修改[正式G结果](libero-formal-G-results-2026-11-06.md)、[科学协议](libero-formal-transfer-protocol-v2-2026-11-05.md)或任何接受门槛。

## 1. 范围、配对和事件语义

固定task2/6、20个G seed、B_fomaml/C_pooled_bc、candidate_acceptance官方初态10..29，共1600条原有轨迹、800对。仅打开这个白名单下的1600个已保存JSON，核对各文件SHA及固定500权重hash、环境seed20261022、5步稳定化、全局/局部索引和公开状态结构。没有读取checkpoint权重或初态容器。

共校验421840行旧Policy动作、423440行公开状态；最短79步，800对都拥有完整前20步。动作k对应pre-action状态k−1和post-action状态k。两臂reset摘要和公开state0逐位相同；前20步没有补帧、末帧延拓或因成功提前终止而丢弃样本。

close定义为首次action[6]>0，close后的open定义为其后首次action[6]<0，0为中性；初始open不算release。它们只是命令，不证明已抓住、运送或释放物体。全程事件按两边实际观察长度判序，缺失保留None，区分“双方可见”“仅一边可见的先后界限”和“另一边已提前终止，无法判定”。不将未见事件填成301步。

## 2. 首步输出已不同，前20步多数夹爪符号一致

800/800对的首步平移动作已经不同，800/800对的post-action joint状态也从第1步出现数值差异；第1步pre-action公开状态仍相同。action差是归一化动作向量的L2，joint差是公开joint坐标间的L2，不是末端/目标位置误差。

| task | 对数 | 首步平移动作差均值 / 中位数 | 首步夹爪符号相同 | 全前20步夹爪符号相同 | 第20步joint间距均值 / 中位数 |
|---|---:|---:|---:|---:|---:|
| 2 | 400 | 0.109934 / 0.104668 | 385/400 | 374/400 | 0.076822 / 0.059062 |
| 6 | 400 | 0.112270 / 0.099876 | 391/400 | 369/400 | 0.076831 / 0.067765 |

共743/800对在整个前20步的夹爪符号都相同；162对成败不同的轨迹中有155对如此（task2为112/117，task6为43/45）。这保留了“运动输出先出现差异、近端夹爪符号尚一致”的现象，但连续夹爪幅值和物理夹爪状态仍可不同。

这些数据支持在相同初始输入下已经存在Policy输出差异，不能将所有差异归给后续状态分布变化。没有对应官方初态的专家正确动作标签，不能判定哪臂首步更正确；没有分离视觉、融合、action head或来源/适应更新的参数贡献。

数值不等采用保存值的精确比较，不代表差异有足够控制意义。task2/6第20步gripper间距中位数分别约7.90e−6/8.24e−6，均值约2.86e−4/1.23e−3，存在离群值；不能直接写成夹爪普遍快速漂移。

## 3. close更早是常见关联，但不足以直接改训练

只在162对成败不同的配对里，按实际赢家重新定向：task2有117对、task6有45对。B-only与C-only、共同成功/共同失败全部保留，原成功计数仍为task2 68/45/72/215、task6 14/36/9/341。

| 赢家相对输家的命令顺序 | 首次close | close后首次open |
|---|---:|---:|
| 双方可见，赢家较早 | 95 | 25 |
| 仅赢家可见，可判定较早 | 22 | 6 |
| 双方可见，赢家较晚 | 36 | 3 |
| 仅输家可见，可判定赢家较晚 | 0 | 16 |
| 双方可见，同一步 | 9 | 0 |
| 双方均未观察到 | 0 | 29 |
| 提前终止造成不可判序 | 0 | 83 |

95对在双方均观察到close时赢家较早，另22对仅赢家有close、输家直到终止仍无close；36对赢家较晚、9对同一步。因此不能把“统一提前close”当作充分或必要的成功规则。完整36对较晚close的反例均导出，没有只挑极端seed。

进一步控制训练seed，先分别计算同一seed里B-only、C-only的“B−C close步数差”中位数，再用前者减后者；负值表示B在B-only时相对更早。joint路径列也采用相同的两层中位数之差，不是直接统计原始B−C路径差的正负。仍然不是同一个初态的跨结果干预，场景混杂保留：

| task | 同seed同时有B-only/C-only | 两侧均有可比较close的seed | close两层中位数之差的负/正seed | 该差的中位数（步） | joint路径两层中位数之差的负/正seed |
|---|---:|---:|---:|---:|---:|
| 2 | 11 | 10 | 7/3 | -3.50 | 5/6 |
| 6 | 4 | 4 | 1/3 | +10.75 | 2/2 |

task2有较多同seed数据支持相对早close的描述，仍有反例；task6只有4个共同seed，3个反而呈相反方向。把所有episode混合的均值不能替代这一分层，更不能假设800对是独立随机任务样本。

## 4. open与持握时长受终止删失影响很大

162对里有83对的open先后因赢家已终止而无法判序，29对双方截至各自终止都未观察到open。单看可见open的平均步数会改变分母，并可能把成功较早终止混入行为差异。

1600条轨迹中326条成功，全部在终止前出现过close；其中264条在终止前没有close后的open命令，首个close段也在终止处右删失。这里只说明记录未观察到该命令，不能据此认定Policy不会open、物体未释放或延长rollout就一定成功。

成功轨迹中还有33条首个close正号段仅1步，这33段均已观察到结束、没有在终止处右删失；因此首个close事件可能是短脉冲，不能直接当作持续有效抓取的开始。失败轨迹同样可出现短脉冲或长close。已保存首段长度与右删失标记；成功终止和300步失败的观察时长不同，不直接据原始均值决定缩短或延长持握。

## 5. 前20步状态变化不支持统一抑制joint运动

两臂joint轨迹间距、各臂离开reset的位移、累计joint路径是不同量。路径定义为前20步相邻公开joint状态L2变化的累加，不是到目标的距离。

| task | 结果分层 | 对数 | 出现该分层的seed数 | B−C joint路径差的配对均值 | 同分层按seed等权均值 |
|---|---|---:|---:|---:|---:|
| 2 | 共同成功 | 68 | 12 | -0.010075 | -0.014745 |
| 2 | 仅B成功 | 45 | 14 | -0.012772 | -0.013069 |
| 2 | 仅C成功 | 72 | 14 | +0.012593 | +0.002377 |
| 2 | 共同失败 | 215 | 19 | -0.016021 | -0.007162 |
| 6 | 共同成功 | 14 | 5 | -0.038415 | -0.032047 |
| 6 | 仅B成功 | 36 | 9 | -0.024933 | -0.007216 |
| 6 | 仅C成功 | 9 | 6 | -0.050130 | -0.064226 |
| 6 | 共同失败 | 341 | 20 | +0.005868 | +0.001393 |

在162对里，赢家前20步joint路径较短有93对，较长有69对。task6的9对C-only全部是赢家C路径更长，覆盖6个seed；task6的36对B-only中则26对赢家路径较短、10对较长。这些反例不支持把“运动更小/偏移更小”当作统一修正方向。

在同seed同时有B-only/C-only的子集中，对两结果层的B−C joint路径差中位数再作差，task2为5负/6正、task6为2负/2正，也未形成跨seed共同方向。共同成功与共同失败也存在首步动作和状态差异，数值分叉本身不足以解释谁成功。

## 6. 验证与证据

[只读入口](../experiments/analyze_libero_formal_G_validation_BC_traces.py)复用已有命令/删失分析与冻结的公开轨迹校验；[测试](../tests/test_libero_formal_G_validation_BC_traces.py)新增39项，连同已有相关测试共109项通过。包含完整800对合成读取、真实旧rollout写出格式联测、提前成功删失、缺失事件、跨seed错配、角色/路径/hash漂移与私有字段拒绝。代码编译和git diff检查通过。

主分析运行时显式禁止Torch、HDF5、LIBERO、robosuite、MuJoCo及src导入，使用NumPy和保存的公开数值，耗时8.379281秒。另用未导入分析器、NumPy或模拟器的标准库脚本重读相同1600条，独立核对事件、前20步、方向、删失及聚合，66010项标量检查通过。

代码绑定在读取前已声明主分析和独立复核各1次：唯一旧轨迹仍为1600条，累计3200次文件读取、846880行旧公开状态访问；不是新采集或样本扩充。两次运行的Policy前向、optimizer、环境交互、checkpoint/HDF5/RGB文件、官方初态容器及held-out payload读取均0。原G结果、权重、日志和所有冻结定义不覆盖。

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 主分析 | `/tmp/libero-formal-G-validation-BC-traces-20261106-v1.json` | `e92dfceb649bfe294d3550e150cc26ea7aed0d1a6dee9d1fb1712130eddc708b` |
| 独立复核 | `/tmp/libero-formal-G-validation-BC-crosscheck-20261106-v1.json` | `2c922d45f12d609d2fbf9b926ba81b494fd301deb2d7029cbaf0391fcadd6787` |
| 可复跑独立复核脚本 | `/tmp/libero-formal-G-validation-BC-independent-check-20261106-v1.py` | `e9be2d7bf6c5a2473e12390f53228e85126cae10d6f2a2329b128a189e29139f` |
| 解读摘要 | `/tmp/libero-formal-G-validation-BC-trace-summary-20261106-v1.json` | `3609d80959e81cb1e8ec82a3e39194b03646d3f710a1142737c1afcb8a2d0311` |
| 全部反例标签（可重叠） | `/tmp/libero-formal-G-validation-BC-counterexamples-20261106-v1.csv`，188行：36较晚close、69路径更长、83open不可判 | `fcf62b952e9ececc51ce482f2a5dd7d93e80f36bb6a8f74170b680b2a6ed46e6` |
| 配置 | `configs/libero-formal-G-validation-BC-trace-readonly_v1.json` | `c527f54d2cd0a0c8e7e8651ab3e1b7719f75b2c37d4e8685ecd840fe66bc0772` |
| 代码绑定 | `configs/libero-formal-G-validation-BC-trace-binding_v1.json` | `f5108d9118da2bf03646e89c0f47df2d1105323f6a65391117094a2d9c95b751` |

| 全量导出 | 路径 | 行数 | SHA-256 |
|---|---|---:|---|
| pairs | `/tmp/libero-formal-G-validation-BC-traces-20261106-v1/pairs.csv` | 800 | `9319262fa098081ee3a298154b0db97261aa1f77b6d83c19e0cf477a7acf4345` |
| episodes | `/tmp/libero-formal-G-validation-BC-traces-20261106-v1/episodes.csv` | 1600 | `40d0005c0a79ad2250bb09a055c7b8c62bbd8c8821753fdfa7d7853d24e63b60` |
| seed_strata | `/tmp/libero-formal-G-validation-BC-traces-20261106-v1/seed_strata.csv` | 200 | `1d260d132aa0498adadf17d7d972879d68ed588419f2216e82ec9577d94297a2` |

## 7. 下一步

当前可以排除“所有差异都先由状态漂移引起”“赢家必然更早close”“赢家必然运动更少”这几种绝对化解释；还不能确定视觉、动作头、数据覆盖或闭环分布变化中的唯一根因。G准入未通过保持，不能据这份事后报告恢复接受、追加预算或注册Module。

该下一项已完成，详见[分段与邻域报告](libero-formal-G-close-runs-late36-diagnostic-2026-11-06.md)。以下保留当时建议作为历史范围：只读已有轨迹的close命令段序列与事件邻域，优先解释36对赢家较晚close的反例：区分首次短脉冲与后续持续close，保留无事件和右删失，不给“有效抓取”任意设阈值。结合相同task/seed/初态的两臂运动变化及共同成功/共同失败参照；不靠重放补出终止后的open，不访问held-out或qualification，不立即改loss或重训。

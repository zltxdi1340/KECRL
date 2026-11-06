# 事件前后、错号方向与同帧平移误差诊断（2026-11-03）

**只读分析和独立复算已完成。** 找到的共同局部模式是：open事件之后，专家仍要求open、Policy却输出close的帧，通常同时具有较高平移目标误差；同一专家帧的跨seed对照在两task、三臂均为正均值。open当步、open之前及close之后有反例，不能推广为所有夹爪错误都会伴随平移错误，更不能据此确定闭环失败根因或提高统一夹爪权重。

## 本轮做了什么

先读工作区/当前报告，冻结[定义](libero-source-internal-event-phase-direction-translation-protocol-2026-11-03.md)与[配置](../configs/libero-source-internal-event-phase-direction-translation-readonly_v1.json)，17项新测试、42项相关专项通过后[绑定代码](../configs/libero-source-internal-event-phase-direction-translation-execution-binding_v1.json)。[分析入口](../experiments/analyze_libero_source_internal_event_phase_direction_translation.py)只读取[原逐帧读出](libero-source-internal-fixed500-expert-frame-target-error-diagnostic-2026-11-03.md)的JSON/NPZ：task0/1、seed20261022/23/24、A/B/C全部18个500端点、6954个唯一source-support专家步和62586行既有预测。新增Policy/训练/环境/HDF5/RGB/初态/checkpoint调用均0。

每个原near-open/near-close并集帧只关联一个5步内最近的**已知**同类事件，等距取较早事件；分为pre（lag -5..-1）、at（0）、post（1..5），保留全部11个lag及空组。主口径为全部窗口；次级排除另一类事件的重叠帧。只从已存目标重建t>=2事件，不推断action0；本数据的全部原窗口都可由这些事件覆盖，unresolved_boundary为0。结果含288个phase组、792个lag组、192个跨seed组和192个B-A/B-C配对组。

| task / 事件 | 已知事件数 | pre帧 | at帧 | post帧 | unresolved | 与另一事件窗口重叠帧 |
|---|---:|---:|---:|---:|---:|---:|
| 0 / open | 31 | 155 | 31 | 153 | 0 | 21 |
| 0 / close | 34 | 170 | 34 | 166 | 0 | 21 |
| 1 / open | 34 | 170 | 34 | 157 | 0 | 0 |
| 1 / close | 34 | 170 | 34 | 170 | 0 | 0 |

四种符号结果互斥：专家open预测close、专家close预测open、涉及neutral的错号、符号正确。错误数/phase帧数与错误数/对应专家符号帧数分开；类别不存在为null，不能当0错误。命令标签和事件前后不等于物理抓取/释放状态或已经确认的控制延迟。

## 六组B：错号方向在事件前后分开

每格均为“错误帧数/对应专家符号帧数”，不是全部phase帧作分母。task0 pre有1帧expert open、post有4帧expert close，这些重叠/重切换标签在原始结果中完整保留；不强制将pre全当close或post全当open。

| task / seed末两位 | open前：专家close预测open | open当步：专家open预测close | open后：专家open预测close | open后错号帧平移MSE减同专家命令正确帧 |
|---|---|---|---|---:|
| 0 / 22 | 100/154 | 4/31 | 12/149 | +0.341539 |
| 0 / 23 | 50/154 | 14/31 | 47/149 | +0.175053 |
| 0 / 24 | 33/154 | 16/31 | 62/149 | +0.170918 |
| 1 / 22 | 19/170 | 30/34 | 123/157 | +0.051724 |
| 1 / 23 | 111/170 | 7/34 | 17/157 | +0.046374 |
| 1 / 24 | 15/170 | 29/34 | 124/157 | +0.073169 |

此前的整窗“方向反转”由两侧错误质量分布共同组成：例如task1 seed23主要在事件前输出与expert close相反的open，seed22/24主要在事件当步和后面输出与expert open相反的close。它不是一个能用统一open偏好覆盖的方向。这里不把这些错号命名为已证实的物理提前/延迟。

## 共同模式：open后错号与平移误差伴随

各seed内，open后错误组与**同一专家符号**正确组的平移MSE差，在两task三臂所有seed都为正。它们仍来自不同帧，可能混入帧难度，因此另做真正同帧的跨seed对照：每帧至少一个seed错号、至少一个seed符号正确，分别取两组seed的平移MSE均值再相减，每唯一帧只计一次。三个seed都错的帧没有正确参照，另列交集，不补值。

下表均为open后、专家open预测close的主口径；均值/中位数的单位是动作前三维MSE差，不是空间距离或因果贡献。

| task / arm | 同帧有效n | 平移差均值 | 平移差中位数 | 平移差为正的帧 | 三seed全错/任一seed错 |
|---|---:|---:|---:|---|---|
| 0 / A | 109 | +0.160619 | +0.168937 | 95/109 | 11/120 |
| 0 / B | 57 | +0.052950 | +0.063759 | 43/57 | 12/69 |
| 0 / C | 67 | +0.048556 | +0.052122 | 51/67 | 14/81 |
| 1 / A | 73 | +0.060726 | +0.060511 | 69/73 | 83/156 |
| 1 / B | 113 | +0.070214 | +0.058285 | 97/113 | 17/130 |
| 1 / C | 105 | +0.070129 | +0.066795 | 88/105 | 41/146 |

因此这个局部模式不只出现在B。各臂有效帧集合不同，不按上表差值大小宣称B比A/C更优。B也有反向同帧：task0为14/57、task1为16/113，不能说所有错号都伴随更高平移误差。

B的三seed共同错号帧，task0共12帧来自demos11/14/17/21，task1共17帧来自demos2/3/5/9/14/20/24/26。同帧一错一对的57/113帧分别覆盖19/29个demo，仍是相关、重复的模型读出，不是独立环境事件。

## 反证、示教覆盖与稳健性

- **open当步不是同一种规律。** task0 B在三个seed内的错误组减正确组均为正（+0.079174/+0.101935/+0.066078），但真正同帧跨seed的14帧差均值为`-0.015544`、中位`-0.007721`，仅7/14为正。不同帧/示教的难度可改变分组结论，不能合并at与post。
- **open之前不跨task一致。** task0 B的expert close预测open伴随较高平移误差，但task1 seed24的同命令组差为`-0.026920`，三seed正向标记为false。
- **close之后有更强反例。** B的expert close预测open在两task各seed内的分组平移差均为正，换成同帧跨seed对照后均值分别为task0 `-0.001465`（52帧）、task1 `-0.007396`（62帧）。不能统一为“错号就同时有更大平移误差”。全部close参照与A/C均保留。
- **同demo对照数量有限。** open后B的同demo错误/正确组同时存在的demo数，task0三个seed为3/9/15，task1为13/8/12（各总计30）。这些demo均值差的均值和中位数均为正，但缺失demo不补值，且仍有反向demo。open当步task0仅1个demo/seed可配，task1为0/1/0，不能据它声称广泛示教一致。
- **排除另一事件重叠后主模式仍在。** task0 B的open后共同错号帧从12减至8（demos11/14/21），同帧混合对照57减至56；平移差均值`+0.053004`、42/56为正，三个seed内组差仍全正。task1无重叠，结果不变。A/C同帧均值也保持正，不用这个次级口径替换主结果。

以上是目标动作误差的条件关联。相同专家输入控制了帧本身，但不同seed的整体拟合水平、表示和动作读出仍一起变化，不能认定夹爪错号导致平移错误、视觉是唯一根因或这些错误已经解释闭环低成功率。

## 决定与下一步

**保持500 updates、现有loss/观测、闭环5步，不追加训练或接受SPT。** 将下一机制关注收敛到“open之后的联合动作读出”，而不是全部夹爪错误或统一open权重；保留open当步和close之后的反证。

下一项可继续只读缓存，用预先固定的非切换帧平移误差作为各seed的整体拟合参照，再检查同帧差是否仍存在，区分局部事件关联与模型整体平移水平差异。该分析尚未执行，不能据本轮均值自动进入新loss/增强训练。validation、held-out、qualification、reserved及final evaluation继续封存，不更新活动SPT/Knowledge、不注册Module。

## provenance与验证

| 产物 | SHA-256 |
|---|---|
| 科学配置 | `162a3248118b94d3c83bb9d898c8bdddbae5210a929795dfd02002966c30d84f` |
| 执行绑定 | `a69adb6ee7fbd2d89e794d0027f9508f89979087f6e72e7329432ba909f33d71` |
| `/tmp/libero-source-internal-event-phase-direction-translation-20261103-v1.json` | `50464d935d643769c52c97e34f0af2d38b3e44f8bdd90e0d0685814495a5b470` |
| `/tmp/libero-source-internal-event-phase-direction-translation-independent-audit-20261103-v1.json` | `b05577dd3e265289f2417ffd5450bea843ed106066a3d0c40837e445e79cbe2f` |
| 独立审计 `/tmp/audit_libero_source_internal_event_phase_20261103_v1.py` | `0c16f9931cee322b629ea83c1eacae40c31ecac307bc3b1b98940d460ddf990a` |

CPU、Python3.10.22、NumPy1.22.4，运行UTC2026-11-03 09:39:27.059817至09:39:27.493451，elapsed0.433650s，无模型设备。独立审计不导入分析入口/parent算子，从两个原NPZ复算事件、288个phase组（含全部demo对照）、792个lag、192个跨seed交集/同帧差和192个配对；191092项递归检查通过，abs/rel1e-12、最大差0、边界未解析帧0。审计新增研究调用0。

新增17项测试，与前序逐帧协议合计 **42 passed in0.31s**；覆盖缺action0、最近事件等距、结束截断、条件分母、同专家命令对照、同帧唯一计数、neutral、矩阵和旧产物保留。编译、CPU环境检查通过；未运行缺少可选Crafter依赖的全量pytest，本轮未修改共享训练/环境代码。最终保护文件、源产物hash、报告数字与链接核验见`/tmp/libero-source-internal-event-phase-final-verification-20261103-v1.json`。所有结论为已暴露source内部诊断，不是formal result、独立泛化、因果Knowledge或Module qualification。

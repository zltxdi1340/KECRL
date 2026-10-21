# LIBERO Spatial 候选 Skill Family 审计（2026-10-21）

本记录回答一个进入 SPT/SPI 实验前的必要问题：`libero_spatial` 的十个任务是否共享同一套技能语义和程序结构，还是包含需要额外技能的复合任务。本次是任务族诊断，不定义训练/验证/最终评价划分，不训练 Policy，不产生 Module，也不构成正式实验结果。

## 审计边界

- LIBERO checkout 固定为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- 演示数据固定为 `f13aa24a3da8c43c7225569f28c562979fa0e35a`。
- 使用 LIBERO 自带的 `robosuite_parse_problem` 解析 BDDL，不从自然语言任务名猜测初态或目标。
- 候选定义保存在 [`configs/libero_spatial_skill_family_candidate_v1.json`](../configs/libero_spatial_skill_family_candidate_v1.json)，其 `formal_result=false`，且明确禁止携带实验 role 或数据 split。
- 可重复审计入口为 [`scripts/audit_libero_skill_family.py`](../scripts/audit_libero_skill_family.py)。任务、revision、目标、对象清单、初态前提或动作接口发生偏移时检查失败。

## 候选 SPT/SPI 语义

候选 Skill Family 定义为 `libero_pick_place_bowl_to_plate`：

- 固定源对象：`akita_black_bowl_1`；
- 固定目标对象：`plate_1`；
- 固定状态转移：源对象从某个初始支撑/容器关系转为 `On(akita_black_bowl_1, plate_1)`；
- 固定程序结构：接近源对象、抓取、搬运到目标、释放；
- 固定 Policy 接口：两路 RGB、7 维关节状态、2 维夹爪状态和 7 维连续动作；
- SPI 参数变化：源对象的初始谓词、初始锚点和接近几何类型。

这一定义很窄。它适合检验同族实例经验能否改善后续 SPI 的有限预算适应，不支持“跨对象类别”“跨目标语义”“复合任务规划”或 Knowledge Evolution 的结论。

## 十任务结构结果

十个任务具有完全相同的场景对象清单、关注对象和单一最终目标。差异只出现在目标黑碗的初始位置：

| task id | 初态类别 | 数量 | 任务特征 |
|---:|---|---:|---|
| 0, 1, 2, 6, 8 | `table_region` | 5 | 黑碗直接位于桌面不同相对区域 |
| 3, 5 | `movable_object_surface` | 2 | 黑碗分别位于饼干盒或 ramekin 上 |
| 7, 9 | `fixture_surface` | 2 | 黑碗位于炉灶或木柜顶面 |
| 4 | `open_container` | 1 | 黑碗位于木柜顶层抽屉内，抽屉在初态已经打开 |

task 4 容易被任务文字误读成“先开抽屉再取碗”。实际 BDDL 同时给出：

```text
(In akita_black_bowl_1 wooden_cabinet_1_top_region)
(Open wooden_cabinet_1_top_region)
```

其唯一目标仍为：

```text
(On akita_black_bowl_1 plate_1)
```

因此 task 4 没有额外的开抽屉动作链。`Open(...)` 是 SPI 启动时已满足的前提，不能记成 Policy 学到的能力；该任务仍可保留在候选族中，但应作为开放容器几何的独立层报告。

## 演示长度诊断

每个任务有 50 条演示。演示步数只用于描述动作难度，不用于选择实验角色：

| task id | 步数均值 | 中位数 | 最小--最大 |
|---:|---:|---:|---:|
| 0 | 101.36 | 95.0 | 75--168 |
| 1 | 134.14 | 132.5 | 109--189 |
| 2 | 117.64 | 117.0 | 93--164 |
| 3 | 101.04 | 99.0 | 86--171 |
| 4 | 149.58 | 148.5 | 127--190 |
| 5 | 115.92 | 115.0 | 86--154 |
| 6 | 126.24 | 124.0 | 106--159 |
| 7 | 142.22 | 137.5 | 120--197 |
| 8 | 119.26 | 114.5 | 95--173 |
| 9 | 137.60 | 136.0 | 117--193 |

开放抽屉和高位 fixture 的演示通常更长，说明几何难度并不均匀。该差异需要在后续划分中分层，不能用总体平均掩盖，也不能因为轨迹更长就把 task 4 解释为另一项动作技能。

## 已验证结论

1. 十个任务可以作为一个窄的 `PickPlace(Bowl, Plate)` 候选 Skill Family；它们共享对象语义、目标谓词、观测/动作接口和四阶段程序结构。
2. 未发现需要额外学习的开关、开抽屉或多物体操作目标。task 4 的抽屉开放条件由环境初态提供。
3. 十个 SPI 的关键变化是源位置和接近几何。SPT 的上下文至少需要编码 `source_predicate`、`source_anchor` 和 `source_geometry`，不能只用 task id 的无语义 one-hot 代替完整实例绑定。
4. 当前证据只证明任务族定义自洽，没有证明 FOMAML 有效、SPT 已更新、Policy 可学习或任一 SPI 能形成合格 Module。

## 下一步门槛

下一步单独冻结候选角色与数据隔离协议：

1. 先按四类源几何分层，再指定 source SPI、SPT candidate validation SPI 和 held-out query SPI；不得根据后续模型结果改角色。
2. 在每个 SPI 内把 50 条 demonstration 和固定初态索引划分为互斥的 meta-support、meta-query、checkpoint validation、qualification 和 final evaluation 集。
3. 明确 A（原始 SPT）、B（来源 SPI 更新后的 SPT）和 C（等来源数据、等更新次数、等模型容量的普通预训练）只在允许的数据角色上训练和选 checkpoint。
4. 完成泄漏审计后，再安装 CUDA PyTorch 并进行有界的反向传播与吞吐 smoke；此时仍不启动正式预算训练。

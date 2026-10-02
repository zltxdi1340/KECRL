# KECRL 实现前接口与数据契约

> 日期：2026-09-29；状态：实现前契约草稿，内部评审。本文是字段映射，不是独立理论来源；正式定义以 [PROJECT_BIBLE](../PROJECT_BIBLE.md) 第 4、7、8、9 节为准。字段表示逻辑语义，不冻结序列化、数据库、进程或后端。

## 1. 运行时调用

```text
RetrieveMechanisms -> RequestImplementation -> ExecuteImplementation
```

### RetrieveMechanisms

```text
RetrieveMechanisms(
  target_capability,
  current_capability_facts,
  environment_scope
) -> Mechanism[]
```

返回范围匹配的 `current + confirmed` 机制，每项至少含 `id`、`P_start`、`P_hold`、`Y`、`reference_time` 和 `applicability_scope`。运行时事实只作为查询输入，不写入 Knowledge Bank。未返回机制不表示环境不可达。

尚未满足启动条件的机制仍可返回，供规划器寻找前置条件。`reference_time` 保留上下文、干预语义、窗口和精度保证；未知不等于零。相同目标的 OR 机制分别返回，不按当前 Module 性能筛选。查询事实不是持久化节点，但同一次观测中合规的证据可另经 `RecordEvidence` 记录。

### RequestImplementation

```text
RequestImplementation(transition_request) -> ImplementationResponse
```

请求至少含 `input_capabilities`、`target_capability`、`resource_requirements`、`environment_scope` 和 `execution_context`，其中必须保留启动/保持角色。响应状态为 `reused_module`、`created_module_from_spi` 或 `unavailable`，并可带 `module_id`、`spi_id`、`spt_id` 及实现契约：启动/保持条件、声明输出、资源消耗/释放、实现约束和适用范围。

`input_capabilities` 表达计划启动条件，不能当作已观测事实。`execution_context` 保存公开运行事实、对象绑定和选定机制的 `P_start/P_hold`。先查找契约、范围、资源和当前执行状态兼容的合格 Module，再尝试 SPT 实例化、Policy 学习/适应和独立资格检查；两条路径都未取得合格实现才返回 `unavailable`。刚实例化的 SPI 不可执行。

| `implementation_contract` 字段 | 正式符号 |
|---|---|
| `start_capabilities` | `I_start` |
| `hold_capabilities` | `I_hold` |
| `declared_output_capabilities` | `O` |
| `resource_consumption` | `C` |
| `resource_release` | `R` |
| `implementation_constraints` | `X` |
| `applicability_scope` | `Omega` |

成功响应必须能定位合格 Module 和其固化契约；未取得实现时标识可为空，不制造虚拟可执行契约。`X` 不是环境机制条件。软性能比较仅在硬资格兼容的候选间进行，返回对象不携带 Policy 参数、SPT 参数或训练轨迹。

### ExecuteImplementation

```text
ExecuteImplementation(implementation, current_state) -> TransitionResult
```

`current_state` 仅是合法可见状态。声明输出不能代替实际环境事实。

```text
TransitionResult:
  target_achieved       # true / false / unknown
  observed_state_changes
  consumed_resources
  released_resources
  produced_capabilities
  execution_status
```

`unknown` 不当作失败。实际状态用于继续规划，并分流为环境证据或技能反馈。

`target_achieved` 的未知是结果可判定性标记，不改变 Capability 的布尔定义，也不冻结具体编码。`produced_capabilities` 仅含实际检测到的结果。资源释放、物料消耗和状态失效分别表达，不能把“丢弃钥匙”记为销毁钥匙。`execution_status` 描述完成、失败或无效等执行情况，其完整枚举尚未冻结；它不复用实现来源状态，也不等同于 Pipeline 的 `completed/continued/unavailable/unknown`。

## 2. 最小持久化对象

SPT：`spt_id`、Skill Family、`version`、固定语义结构、参数绑定 Schema、契约/程序 Schema、实例化与反馈接口、可恢复的可演化状态 `omega` 或其引用、来源元数据。摘要不能替代生成和恢复所需的参数。

SPI：`spi_id`、所属 SPT 及版本、参数绑定、状态转移请求、实现契约 `Gamma`、程序规格 `P`、初始化状态、适应规格、来源元数据。

Module：`module_id`、唯一 `spi_id`、可执行 Policy 引用、固化契约 `Gamma`、技能经验摘要、内部统计摘要、来源/版本/生命周期元数据。只有通过资格检查的 Module 才能返回为可执行实现。

SPI 的标识与生成 SPT 版本必须可追溯，不另行规定 SPI 生命周期；Module 元数据定位具体实现版本、Policy 版本和资格依据。SPT 保留当前版本、上一稳定版本及接受/恢复依据，这些是现有版本规则的记录要求，不引入 `provisional` 或 `quarantined` 状态机。既有 Module 不因 SPT 更新而原地改写，需重评估时形成新版本；一份 SPI 可对应多个 Module。技能经验的原始轨迹、缓冲区或摘要形式仍待定，不强制只存摘要。

Knowledge Evidence：公开摘要使用 `before_state`、`after_state`、`public_observation`、`environment_scope`、`intervention_metadata`、`evidence_validity`。普通执行的干预字段必须明确为未实施干预。按来源和适用性补充以下既有语义：

| 证据语义 | 要求 |
|---|---|
| 来源与去重依据 | 区分 `declared_rule`、`llm_prior`、`empirical_transition`；同一观察/验证在同一命题下不重复累计 |
| 命题、关系与范围 | 明确验证的是可达性、必要性、时间角色还是成本；不跨不兼容范围合并 |
| 干预与配对 | 基础世界引用、消融条件、时间角色、窗口、合法性、副作用和混杂；引用不泄露完整状态 |
| 验证结果 | `FOUND/PROVEN_UNREACHABLE/UNKNOWN`，必要时附参考时间、界限或近似保证 |
| 原子命题映射 | Knowledge Evolution 按有效性判定 `1/0/bottom`，只累计前两类；保留未决原因 |

隐藏规则、完整状态和 Oracle 路径仅供验证端使用，不进入学习端证据。普通 Policy 失败不能映射为结构反证，实际执行耗时不能直接作为最低参考时间。无需参考时间的公开转移不伪造该字段；未知与缺失均不能默认为零。

## 3. 两库边界

Knowledge Bank 允许保存 Capability Schema（初始化后冻结）、机制、证据、适用范围、参考成本、认知状态和生命周期/派生历史；禁止保存动作、Policy 参数或梯度、完整训练轨迹、SPT/SPI、Module 性能或技能成功率。

Skill Library 允许保存 SPT、SPI、合格 Module、Policy 引用、技能经验和技能侧统计；禁止把技能性能直接写成 Knowledge Bank 机制，也不负责替代环境结构验证。规划器只保留任务内上下文，不形成第三个长期知识库。

SPI 本体不保存最终 Policy、Module 长期统计/生命周期或机制置信度，也不自行查询 Knowledge Bank、保存跨机制计划。Skill Library 可使用请求中的 Capability 契约和公开上下文，但不复制一份长期环境机制库。Knowledge Bank 不持久化动态 affordance 或规划器临时事实。任务固定起始 Knowledge Bank 与 SPT 版本视图；新知识与新接受的 SPT 默认供后续任务使用。

## 4. 最小数据结构与版本引用

以下是已有对象的实现映射，不规定类名、数据库表或序列化格式。

| 对象 | 最小内容 | 归属 |
|---|---|---|
| Capability Schema | 谓词语义、参数约束、公开判定接口、适用范围 | Knowledge Bank；审核后固定，运行时绑定不创建新节点 |
| 运行时事实 | Schema 引用、实体/数量绑定、公开判定和观测上下文 | 任务内；合规观测可另形成 Evidence |
| 机制 | `id`、`P_start`、`P_hold`、`Y`、范围、参考时间、认知/生命周期状态、证据历史 | Knowledge Bank；AND/OR 由集合关系表达 |
| 原子统计 | 命题类型、范围、`S`、`C`、证据引用 | Knowledge Bank；可达性、必要性、角色、成本分别更新 |
| SPT / SPI / Module | 采用第 2 节的最小字段 | Skill Library；Module 唯一绑定 SPI |

未知值必须保留为未知：观测不足不能转为 `false`，参考时间未知不能转为零，缺失 `P_hold` 不能自动解释为空集合。未取得实现时响应可无实现标识；成功响应必须定位合格 Module 和固化契约。执行状态、任务结果和知识认知状态保持不同语义，不合并为新状态机。

版本追溯关系为：

```text
SPT 版本 -> SPI（生成版本） -> Module（实现版本） -> Policy 引用与资格依据
```

任务开始固定 Knowledge Bank 与 SPT 版本视图；任务更新供后续任务使用。SPT 接受或恢复只影响后续实例化，不原地改写既有 Module。

## 5. 最小调用时序

```text
任务开始：固定起始 Knowledge/SPT 视图，读取公开观测与目标
  -> RetrieveMechanisms
  -> Planner 选择机制，检查前置、保持条件和资源
  -> RequestImplementation
       -> 复用合格 Module：reused_module
       -> 否则实例化 SPI、学习并独立资格检查：created_module_from_spi
       -> 未取得合格实现：unavailable（不调用 ExecuteImplementation）
  -> ExecuteImplementation
  -> TransitionResult
  -> Planner 用实际状态检查目标：completed / continued / unknown
  -> 公开环境证据进入 Knowledge Evolution；SPI 技能经验留在 Skill Library
  -> 任务或明确更新边界按既有预算更新，结果供后续任务使用
```

检索为空不等于环境不可达；声明输出不能代替实际检测。`unknown` 不当作失败或结构反证；中间转移成功也不自动等于整个任务完成。上述顺序不规定同步框架、重试次数或规划算法。

## 6. 后续代码落位建议

依据 `ARCHITECTURE.md`，`src/knowledge/` 承担 Knowledge Bank，`src/skills/`（尚不存在）承担 SPT/SPI/Module，`src/continual_learning/` 承担 Planner/Pipeline，`src/environments/` 承担公开观测与执行适配，`src/counterfactual/` 承担干预验证接入。本轮不创建目录或代码文件。

语言、依赖、存储形式、Policy 后端、环境适配器、资格统计方法和数值参数仍未决定；接口设计不等于 FOMAML 或 Knowledge Evolution 已实现。

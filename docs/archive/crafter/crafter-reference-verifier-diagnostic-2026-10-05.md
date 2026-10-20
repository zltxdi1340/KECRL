# Crafter 配对世界 Reference Verifier 诊断（2026-10-05）

本次运行只审计配对世界验证器的接口和证据映射，使用 fixture oracle
摘要，不启动 Crafter，也不执行 Policy。结果不是 Crafter 机制真值，也不是
论文实验结果。

## 记录

- 实现：`src/counterfactual/crafter_reference.py`
- Runner：`experiments/run_crafter_reference_verifier_diagnostic.py`
- 配置：`configs/crafter_reference_verifier_diagnostic_v1.yaml`
- Commit：`246ae9a1613630f1ac6c9a6c31d9031d02b9f08e`
- `formal_result=false`
- 结果目录：`results/crafter_reference_verifier_diagnostic_246ae9a/`

## 三态映射

| fixture 情形 | verifier outcome | Knowledge observation | evidence_validity |
|---|---|---|---|
| intervention world reaches target | `FOUND` | `1` | `valid` |
| complete search proves target unreachable | `PROVEN_UNREACHABLE` | `0` | `invalid` |
| search budget incomplete | `UNKNOWN` | `bottom` | `unknown` |
| intervention has side effects | `UNKNOWN` | `bottom` | `confounded` |

必要性消融还要求 baseline control 已经达到目标；否则结果保持
`UNKNOWN/bottom`，不能把 intervention world 的成功解释成必要性证据。

## 边界检查

Reference runner 可以在 oracle-only 端使用完整状态和规则，但输出只保留
pair/world 标识、范围、干预描述、结果、理由和参考步数。生成的
Knowledge Evidence 不包含 Policy、梯度、完整轨迹、Module 性能或技能成功率。
普通 Policy 执行仍保持 `evidence_validity=unknown`，不会直接进入该验证器的
结构性结果。

真正接入 Crafter 前仍需实现合法的 paired-world intervention、side-effect
检查和 reference planner；本次 fixture 诊断不替代这些工作。

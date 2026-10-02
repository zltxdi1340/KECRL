# Knowledge Evolution 与 Skill Evolution 联合目标决策

> 日期：2026-09-25  
> 状态：已确认  
> 适用范围：KECRL 第一篇工作，数学建模阶段

## 1. 背景

Knowledge Evolution 与 Skill Evolution 分别回答“学习什么”和“如何学习”。需要说明二者如何形成系统级目标，同时保持 Knowledge Bank 与 Skill Library 的严格边界。

## 2. 方案选择

候选方案包括：

1. 将两类演化压成一个复杂加权损失；
2. 完全独立，只分别评价而不定义系统层关系；
3. 保留两个局部演化目标，并用任务层目标评价二者的联合效果。

第一篇工作采用第三种方案。它保留两类演化的科学含义和独立更新算法，又能通过任务表现表达知识质量与技能学习效率的实际协同结果，复杂度与当前项目规模相匹配。

## 3. 正式形式

Knowledge Evolution 局部目标：

\[
J_K(K_t)=Q_{\mathrm{structure}}(K_t)-\lambda_K C_K.
\]

Skill Evolution 局部目标：

\[
J_S(L_t)=
-\mathbb E_{I\sim\mathcal D_{\mathrm{SPI}}}
\left[\sum_k\alpha_k\mathcal L_I^{query}(\theta_I^k)\right]
+\lambda_Q Q_{\mathrm{module}}.
\]

任务层耦合评价：

\[
J_T(K_t,L_t,\Pi)=
\mathbb E_{\tau}\left[R(\tau)-\lambda_c C(\tau)-\lambda_u U(\tau)\right].
\]

系统形式为：

\[
\max_{\Pi,U_K,U_S}J_T(K_t,L_t,\Pi)
\]

满足 Knowledge 完整性与 Module 资格约束，并保持：

\[
K_{t+1}=U_K(K_t,E_t^K;B_t^K),
\qquad
L_{t+1}=U_S(L_t,E_t^S;B_t^S).
\]

## 4. 信息边界

- \(E_t^K\) 只包含合法公开环境证据。
- \(E_t^S\) 只包含 SPI、Policy 和 Module 技能反馈。
- Knowledge Evolution 不直接修改 Policy。
- Skill 性能、训练轨迹、梯度和 Module 内部统计不直接写入 Knowledge Bank。
- 规划层通过任务层选择协调二者，但不成为第三个长期知识库。

## 5. 尚未决定

以下内容保留为后续建模或实验设置：

- \(Q_{\mathrm{structure}}\) 的具体分解；
- 任务成本是否进入实验主目标；
- \(\kappa_K,\kappa_S\) 的数值；
- \(J_T\) 是否只用于系统评价；
- 是否在未来需要单一标量化目标。

这些未决项不影响当前两个局部更新算法和接口边界。

## 6. 影响范围

- PROJECT_BIBLE 第 6.5 节记录联合目标。
- ROADMAP 的联合目标标记完成，下一项为整体 KECRL 的数学形式化和可证明性质。
- 不引入新的长期知识库、统一训练器或额外状态机。
- 不涉及实验环境、基线、指标或代码实现。

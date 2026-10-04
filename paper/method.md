# Method: Knowledge-Evolving Continual Reinforcement Learning

> Draft status: internal review, updated 2026-09-29. This document summarizes confirmed design choices from [PROJECT_BIBLE](../PROJECT_BIBLE.md). Historical decisions are read with their later amendments, especially the [2026-09-21 simplification](../docs/decisions/2026-09-21-theory-design-principles-and-simplification.md). No implementation or formal experimental results are reported.

## 1. Problem Setting

For a task goal \(g\), the environment is modeled as a decision process with a public observation interface:

\[
\mathcal E_g=(\mathcal S,\mathcal A,\mathcal P,R_g,\gamma,\mathcal O,h).
\]

The agent does not need access to the complete state. A Capability is a parameterized, detectable, agent-independent predicate

\[
c_\theta:\mathcal S\rightarrow\{0,1\},
\]

with a public detector satisfying \(\widehat c_\theta(h(s))=c_\theta(s)\) in its declared scope. Missing observations are not false predicates. The initial Capability Schema \(V_0\) is proposed by an LLM from the public environment description, reviewed, and then frozen. Online evolution may update relations among existing capabilities, but may not create a new Capability Schema; online planning and updates do not depend on the LLM.

## 2. Two Long-Term Evolution Objects

KECRL keeps two strictly separated long-term memories.

The Knowledge Bank stores environment structure: mechanisms, evidence, applicability scopes, reference costs, confidence states, and lifecycle history. A mechanism is

\[
M=(P_{start},P_{hold},Y),
\]

where \(P_{start}\) is the set of startup conditions, \(P_{hold}\) the conditions that must remain valid during execution, and \(Y\) the target Capability. Parents within one mechanism are conjunctive (AND); multiple mechanisms reaching the same target are alternatives (OR). Hard reachability is maintained separately from cost-modifying factors.

The Skill Library stores SPTs, SPIs, qualified Modules, and skill-side experience. The Knowledge Bank must not receive or persist Policy parameters, gradients, complete training trajectories, or Module performance. Public environmental evidence from an execution may enter Knowledge Evolution separately; skill feedback itself is not structural evidence. The planning layer coordinates the two memories at runtime but is not a third long-term knowledge store. Historical references to Module Library denote this same Skill Library.

## 3. Skill Representation

### 3.1 Skill Program Template

For Skill Family \(f\), an SPT is

\[
T_f^t=\langle T_f^{static},\omega_f^t\rangle.
\]

The static component specifies the family semantics, parameter-binding schema, contract construction, program schema, instantiation interface, and feedback interface. The state \(\omega_f^t\) contains the evolvable cross-instance initialization and adaptation information. In the current version, the family semantics are predefined and fixed; Skill Evolution updates only \(\omega_f^t\).

### 3.2 Skill Program Instance

Given an SPT, parameter binding \(b_j\), and a state-transition request \(q_j\),

\[
I_j=Instantiate(T_f^t,b_j,q_j),
\]

where

\[
I_j=\langle id_j,f,v_f^t,b_j,q_j,\Gamma_j,P_j,\zeta_j^0,A_j\rangle.
\]

Here \(\Gamma_j\) is the instance contract, \(P_j\) the implementation and evaluation specification, \(\zeta_j^0\) the generated initialization, and \(A_j\) the adaptation specification. An SPI is a learning specification, not an already learned executable skill.

### 3.3 Module

A qualified Module is the executable unit stored for reuse:

\[
m_i=\langle id_i,spi_i,\pi_i,\Gamma_i,E_i,\Sigma_i,\mu_i\rangle.
\]

It binds exactly one SPI to a learned Policy, a frozen implementation contract, skill experience, internal statistics, and provenance/version metadata; one SPI may produce multiple Modules. Its contract is \(\Gamma_i=\langle I_i^{start},I_i^{hold},O_i,C_i,R_i,X_i,\Omega_i\rangle\): startup and hold conditions, declared output, resource consumption and release, implementation-specific constraints, and applicability scope. Declared output is not an environment fact; only an observed `TransitionResult` can confirm the target. The SPI does not own the final Policy, Module statistics or lifecycle, or mechanism confidence.

## 4. Runtime Interface and Continual Loop

The planner uses the minimum cross-library protocol:

```text
RetrieveMechanisms
-> RequestImplementation
-> ExecuteImplementation
-> inspect actual transition
-> continue planning or finish
```

`RetrieveMechanisms` returns scope-matched current and confirmed mechanisms. `RequestImplementation` first reuses a compatible Module; otherwise it instantiates an SPI from the relevant SPT, learns/adapts a Policy, and returns a Module only after qualification. `ExecuteImplementation` returns the observed state changes, resource changes, target status, produced capabilities, and execution status.

Each task uses a fixed beginning-of-task view of the Knowledge Bank and SPT versions. New knowledge and newly accepted SPT versions become available to later tasks and do not rewrite existing Modules. Results are classified minimally as `completed`, `continued`, `unavailable`, or `unknown`; unavailable or unknown does not imply environmental impossibility.

## 5. Knowledge Evolution

### 5.1 Evidence and candidates

Public observations are mapped to facts over \(V_0\). Evidence is recorded with its source, scope, intervention metadata, and validity. A result is \(y\in\{1,0,\bot\}\): support, valid in-scope structural counterevidence, or unknown/invalid evidence. Unknown observations, ordinary policy failure, and uncontrolled confounding do not count as structural counterevidence.

Candidates originate from initialization, unexplained transitions, or evidence questioning an existing relation. Candidate generation uses sparse parent sets and local pruning; it may refine parameters, parents, roles, or scopes using existing capabilities. If the existing Schema cannot express the explanation, the system records `knowledge_gap` rather than creating an online node.

### 5.2 Statistical update and confirmation

For every atomic proposition \(c\), maintain support and counterevidence counts:

\[
S_{c,t}=\sum_{\tau\le t}\mathbf 1[y_{c,\tau}=1],\qquad
C_{c,t}=\sum_{\tau\le t}\mathbf 1[y_{c,\tau}=0].
\]

With a Beta prior,

\[
p_c\sim Beta(\alpha_0,\beta_0),\qquad
p_c\mid D_t\sim Beta(\alpha_0+S_{c,t},\beta_0+C_{c,t}).
\]

The first version uses the confirmed weak prior \(Beta(1,1)\); any alternative prior is an explicit runtime or experimental setting. With \(n_{c,t}=S_{c,t}+C_{c,t}\), an atomic proposition is `confirmed` when \(n_{c,t}\ge n_{min}\) and its posterior lower bound is at least \(\tau_{confirm}\); it is `rejected` when \(n_{c,t}\ge n_{min}\) and its upper bound is at most \(\tau_{reject}\); otherwise it remains `testing` (or `candidate` before testing begins). A complete mechanism is confirmed only when reachability, parent conditions, and time-role claims have separately sufficient evidence. Cost claims have their own statistics and cannot change hard reachability.

### 5.3 Budget and maintenance

These comparisons are inclusive. Atomic rejection does not automatically reject an entire mechanism: a necessary-condition claim may require refinement or investigation of an OR alternative. A confirmed relation returns to `testing` only when new legal, comparable structural counterevidence reaches the rejection rule; ordinary execution failure or UNKNOWN cannot trigger this rollback. Epistemic states (`candidate`, `testing`, `confirmed`, `rejected`) are separate from lifecycle states (`current`, `superseded`, `retired`).

In round \(t\), the verification set \(Q_t\) obeys

\[
\sum_{q\in Q_t}cost(q)\le B_t.
\]

The scheduler prioritizes structural challenges to confirmed relations, target-relevant uncertainty, and then other candidates with aging support. Unfinished tasks remain pending across rounds with reasons and retry conditions; unchanged conditions do not justify repeated identical verification. `Refine` creates a derived candidate without inheriting a confirmed label; a replacement supersedes the old relation only after confirmation and coverage of the old scope. `Merge` is allowed only for semantically equivalent relations and deduplicates evidence; `Retire` removes a relation from the current maintenance scope without asserting that it is false. Forward retrieval returns only scope-matched `current + confirmed` mechanisms. Here \(B_t\) is the Knowledge budget denoted \(B_t^K\) at system level.

## 6. Skill Evolution

For compatible SPI contexts,

\[
z_j=Encode_f(b_j,q_j,\Gamma_j,P_j,\Omega_j),\qquad
\zeta_j^0=G_{\omega_f^t}(z_j).
\]

The SPI performs bounded support adaptation,

\[
\theta_j^{k+1}=U_{A_j}(\theta_j^k,D_j^{support,k}),
\]

and is evaluated on independent query data. The first version uses context-conditioned FOMAML as the single primary meta-update:

\[
g_j^{FO}=\left(\frac{\partial G_{\omega_f}(z_j)}{\partial\omega_f}\right)^\top
\nabla_{\theta_j^{K_j}}\mathcal L_j^{query}(\theta_j^{K_j}),
\]

\[
\widetilde\omega_f=\omega_f^t-\beta\frac{1}{|\mathcal J|}\sum_{j\in\mathcal J}g_j^{FO}.
\]

The update creates a candidate SPT and never directly overwrites the active version. A learned SPI becomes a Module only after independent held-out qualification: startup/hold conditions, target detectability, contract and resource constraints, applicability scope, and finite execution budget must pass. A hard contract violation cannot be offset by average success rate; insufficient or indeterminate evidence remains pending.

An SPT candidate is evaluated independently against the active version. It is accepted only if independent evaluation shows the required improvement in future SPI learning efficiency, no unacceptable regression on important existing SPI, and no contract or safety violation; insufficient evidence keeps the active version unchanged. The previous stable version is retained while a candidate is evaluated. A new version may be rolled back after comparable-range systematic degradation or a clear safety/contract failure; a single ordinary failure is insufficient. SPT updates affect future SPI generation and do not delete or rewrite existing qualified Modules. Candidate acceptance, qualification, and rollback decisions remain in the Skill Library and do not alter Knowledge Bank mechanism status.

## 7. Minimal Algorithms

The following pseudocode summarizes existing rules, not executable code or additional APIs. Helper headings name document procedures only.

### 7.1 Knowledge Evolution

```text
KnowledgeEvolution(round_budget):
  RecordEvidence for public observations over frozen V0; immediately assess affected claims
  maintain deduplicated tasks from initial candidates, unexplained transitions, and challenges
  create/refine sparse candidates only over V0; record knowledge_gap if V0 is insufficient
  while an executable verification step fits the remaining total budget:
    select: challenged confirmed > target-related uncertainty > other candidates with aging
    specify proposition, scope, base world, time roles, and intervention window
    reuse suitable evidence; complete public rules may be checked by actual transitions
    when intervention is needed:
      obtain/reuse control reachability with the target initially absent
      if control UNKNOWN: retain reason and retry condition; skip ablation
      if control PROVEN_UNREACHABLE: investigate candidate/scope; skip ablation
      otherwise: independently ablate each relevant parent from the same base world
      check legality/confounding; respect other parents' startup/hold roles
      use controlled Reference Planner results; never expose hidden rules or Oracle paths
    after each new result, RecordEvidence with scope, window, validity and time guarantees
    for each affected atomic claim, count each suitable evidence item at most once:
      y = 1 or 0 updates S or C; y = bottom leaves both unchanged
      recompute Beta(alpha_0 + S, beta_0 + C) and assess Section 5.2 bounds
    immediately assess whole mechanisms under Section 5.3, including confirmed -> testing
    keep costs separate; reuse valid time evidence for the declared intervention only
    retain OR ambiguities; add distinguishing or time-role checks without automatic deletion
    Refine, Merge, supersede, or Retire only under their evidence and history rules
  return updated relations, evidence history, pending tasks, and knowledge_gap records
```

### 7.2 Skill Evolution with FOMAML

```text
SkillEvolution(family_batch):
  instantiate compatible SPI contexts from the active SPT
  for each SPI j:
    encode public context z_j and generate zeta_j^0, including Policy initialization theta_j^0
    adapt for the bounded support steps using support data only
    evaluate independent query loss and retain skill-side feedback
  form g_j^FO with the generator Jacobian and final query gradient (Section 6)
  omit the inner-loop second-order terms; update by beta times the batch mean gradient
  create a candidate SPT state omega_tilde_f; do not overwrite the active version
  compare candidate and active SPT on independent, comparable SPI evaluations
  accept only with required future-learning improvement and no unacceptable regression or violation
  otherwise retain the active version; preserve the previous stable version for recovery
```

### 7.3 Module qualification and SPT version decision

```text
QualifyModule(SPI, learned_policy):
  freeze the policy and check contract, scope, detectability, resources, and finite budget
  collect independent qualification evidence
  if a clear hard contract or safety violation is established:
    do not qualify the implementation
  else if evidence is sufficient and success and contract thresholds pass:
    create a qualified Module bound to this SPI
  else if sufficient evidence establishes performance below the qualification threshold:
    do not qualify the implementation
  else:
    keep qualification undecided for later evaluation
  do not count unknown/invalid results automatically as failures

AcceptOrRollbackSPT(candidate, active, validation):
  compare on the same SPI contexts, adaptation budgets, algorithms, and evaluation rules
  accept candidate only when improvement, non-regression, and safety/contract rules pass
  on failed gates reject the candidate; on insufficient evidence leave it undecided
  in either case retain the active SPT; no extra version state machine is required
  on acceptance, switch the active version and retain the prior stable version
  after systematic comparable degradation or a clear violation, restore the prior stable version
  do not remove existing qualified Modules solely because an SPT version is restored
```

### 7.4 Continual Learning runtime loop

```text
RunTask(goal, beginning_of_task_views):
  fix Knowledge Bank and SPT version views for this task
  repeat:
    read public observation and refresh runtime Capability facts
    if goal is observably achieved: outcome = completed; break
    if required state or goal cannot be determined: outcome = unknown; break
    mechanisms = RetrieveMechanisms(goal, current_facts, scope)
    if none are known: retain knowledge insufficiency; do not infer environmental impossibility
    planner selects a mechanism or a prerequisite/alternative and constructs transition_request
    if knowledge is insufficient to determine the next transition: outcome = unknown; break
    response = RequestImplementation(transition_request)
    retain any SPI learning feedback in Skill Library, even if qualification failed
    if response.status is unavailable:
      if an alternative is available: replan and continue
      otherwise: outcome = unavailable; break
    check actual startup facts, hold monitoring, scope, resources and implementation contract
    if required checks are indeterminate: outcome = unknown; break
    if checks require replanning: replan without executing an incompatible implementation
    else:
      result = ExecuteImplementation(response implementation, publicly visible state)
      record allowed public evidence for Knowledge Evolution and skill feedback in Skill Library
      refresh actual facts from result and public observation, never from declared output
      if task goal is observably achieved: outcome = completed; break
      if a required result cannot be determined: outcome = unknown; break
    outcome = continued
    continue replanning, or break when yielding at the chosen task boundary
  at a task or explicit update boundary, run the separately budgeted Knowledge/Skill updates
  return outcome
```

The planner uses the task's beginning-of-task Knowledge Bank and SPT views throughout this loop. Updates become available to later tasks and do not form a third persistent store.

## 8. Framework Figure Specification

The framework figure should use the following information flow, with two visibly separate persistent stores:

```text
Environment -- public observations --> runtime Capability facts --> Planner
Knowledge Bank -- RetrieveMechanisms (Capability mechanisms) --> Planner
Planner -- RequestImplementation --> Skill Library
  SPT -> SPI -> Policy learning/adaptation -> qualification -> Module
  existing qualified Module --------------------------------> reuse
  executable hierarchy: SPT -> SPI -> Module -> Policy
Planner -- ExecuteImplementation (qualified Module / Policy) --> Environment
Environment -- TransitionResult / actual public facts --> Planner
Environment evidence --> Knowledge Evolution --> Knowledge Bank
Skill feedback ------> Skill Evolution -------> SPT in Skill Library
```

Draw exactly two persistent-store boundaries: Knowledge Bank contains Capability Schema, mechanisms, evidence and lifecycle; Skill Library contains SPT, SPI, qualified Module and skill experience. Capability facts are runtime inputs, not new persistent nodes. Put Planner outside both boundaries and label it as task-local coordination, not a third long-term store. The hierarchy arrows describe provenance/containment; Module formation still requires Policy learning and qualification. Split feedback into two arrows, with no Policy parameters, gradients, trajectories or Module performance crossing into Knowledge Bank. This text diagram is the figure specification, not a completed publication graphic.

## 9. Joint Objective and Formal System

The two evolutions retain separate local objectives:

\[
J_K(K_t)=Q_{structure}(K_t)-\lambda_K C_K,
\]

\[
J_S(L_t)=-\mathbb E_{I\sim\mathcal D_{\mathrm{SPI}}}\left[\sum_k\alpha_k\mathcal L_I^{query}(\theta_I^k)\right]
+\lambda_Q Q_{module}.
\]

Their coupling is evaluated at the task level:

\[
J_T(K_t,L_t,\Pi)=\mathbb E_\tau[R(\tau)-\lambda_c C(\tau)-\lambda_u U(\tau)].
\]

Here \(\alpha_k\) weights learning checkpoints, not inner-loop step sizes. These local design objectives do not introduce an additional optimizer beyond Section 6. System-level maximization of \(J_T\) over \(\Pi,U_K,U_S\) is subject to \(\mathrm{Integrity}(K_t)\ge\kappa_K\) and \(\mathrm{Qualification}(L_t)\ge\kappa_S\); quality decompositions, weights and thresholds remain unfixed.

The long-term states are \(K_t=(V_0,\mathcal M_t,\Xi_t)\) and \(L_t=(\mathcal T_t,\mathcal I_t,\mathcal U_t)\), respectively storing mechanisms and their evidence metadata, and SPTs, SPIs and qualified Modules. The complete layered object is

\[
\mathfrak K=(\{\mathcal E_g\}_{g\in\mathcal G},V_0,K_t,L_t,\Pi,U_K,U_S),
\]

with independent updates

\[
K_{t+1}=U_K(K_t,E_t^K;B_t^K),\qquad
L_{t+1}=U_S(L_t,E_t^S;B_t^S).
\]

Here \(E_t^K\) is legal public environmental evidence and \(E_t^S\) is SPI/Policy/Module skill feedback. The task objective coordinates the layers but does not replace either local update rule.

## 10. Finite Guarantees and Non-claims

The six properties have the following limited premises, as specified in PROJECT_BIBLE Section 6.6:

1. Schema invariance: initialization is reviewed and updates cannot create Schema, hence \(V_t=V_0\).
2. Memory boundary preservation: typed interfaces and updates persist only their permitted inputs.
3. Mechanism precondition consistency: startup conditions are detected and hold conditions can be checked over the declared interval; this does not guarantee a Module or successful execution.
4. Conditional composition: qualified Modules have compatible scope/resources, actual outputs satisfy subsequent startup conditions, and hold conditions remain true; only actual final goal detection establishes completion.
5. Posterior correctness: within the same scope, valid evidence follows the declared conditional-independent or exchangeable Bernoulli model with a Beta prior; UNKNOWN, confounding and incompatible scopes do not satisfy this premise.
6. Round termination: budgets are finite, each verification/adaptation/update step consumes strictly positive budget, and every individual step terminates. Pending tasks may survive the round.

These are conditional finite properties, not global optimality or unconditional convergence results. The method does not claim monotonic task performance, discovery of all mechanisms within finite budget, online discovery outside \(V_0\), global FOMAML convergence, or a globally optimal unified scalar objective.

## 11. Open Runtime and Experimental Parameters

The following remain parameters rather than theoretical conclusions: posterior confidence level and thresholds, minimum sample size, verification costs and budgets, scheduler details, evidence-buffer policy, FOMAML step sizes and adaptation budgets, qualification and SPT acceptance thresholds, rollback windows, Policy backend, task/environment selection, baselines, metrics, and resource allocation. The project has completed controlled candidate-stage CUDA runs, but has no promoted formal paper results; external-environment validation and final statistical inference remain pending.

Implementation preparation is indexed in the [interface and data contract](../docs/interface-and-data-contract.md) and [parameter register](../docs/runtime-and-experiment-parameters.md). The Crafter wooden-pickaxe case in historical decisions illustrates the procedure only and supplies no experimental result or frozen benchmark choice.

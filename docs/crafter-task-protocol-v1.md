# Crafter Task Protocol v1 Preview

Status: candidate split preview, not a formal experiment freeze.

The preview uses Crafter 1.8.3, seeds `0..4`, and the `rgb64_inventory_v1`
adapter. Candidate task families are:

| Task | Candidate family | Public success predicate |
|---|---|---|
| collect wood | gather | `inventory[wood] >= 1` |
| collect stone | gather | `inventory[stone] >= 1` |
| collect coal | gather | `inventory[coal] >= 1` |
| obtain wood pickaxe | craft | `inventory[wood_pickaxe] >= 1` |
| obtain stone pickaxe | craft | `inventory[stone_pickaxe] >= 1` |
| obtain iron pickaxe | craft | `inventory[iron_pickaxe] >= 1` |

The predicate uses only the public inventory allowlist. Missing or invalid
inventory values remain `unknown`; achievements and native semantic maps are
not substitutes for the predicate.

For each seed and task, the preview creates separate train, support, query,
qualification, and evaluation assignments. SPT validation has distinct support
and query phases rather than reusing one set. Each assignment starts a fresh
native environment and records a deterministic environment seed. The preview
contains two assignments per task and phase, giving 420 assignment records.

This split does not yet establish held-out SPI generalization: task families,
object bindings, context encoding, horizon, reward, policy architecture,
qualification thresholds, SPT acceptance rule, Knowledge intervention protocol,
and formal budgets remain pending. Unique seed values alone are not sufficient
to claim independent world layouts or formal statistical power.

Before formal training, the real image policy must connect to context-conditioned
FOMAML, a qualified Module must execute through the Pipeline, and a separate
reference verifier must produce `FOUND`, `PROVEN_UNREACHABLE`, or `UNKNOWN`
under paired Crafter interventions. Ordinary policy outcomes remain skill
feedback plus unknown Knowledge Evidence.

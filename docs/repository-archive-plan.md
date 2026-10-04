# Repository Archive Plan

The repository is being prepared for a clean transition from the controlled
discrete benchmark to a harder target environment.

Keep under version control: theory and decision documents, source code,
tests, experiment runners, frozen configuration files, manifest generators,
small result summaries, and archive metadata.

Keep on the server but exclude from GitHub: model checkpoints, repeated raw
episode dumps, transient logs, Python caches, build metadata, and generated
large result directories. Their metadata and analysis summaries remain
available for audit.

The controlled environment is sealed rather than deleted. New method claims
must use a target environment with non-trivial mechanism uncertainty,
non-saturating query difficulty, and an independently specified evaluation
protocol.

Before pushing, run the test suite, inspect `git diff --check`, review the
staged file list, and record the resulting commit and remote branch. Do not
force-push or rewrite history.

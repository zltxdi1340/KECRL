# Independent Query Learning Efficiency Instrumentation

The controlled runner now records the protocol's primary metric explicitly:
support interaction steps required to reach an independent query success-rate
threshold of `0.8`. Runs below the threshold are marked right-censored and are
never converted to zero steps.

The holdout v2 diagnostic used seeds `5..9` and 200 episodes per role. All
variants reached the threshold. With prior strength `0.0`, mean support steps
were `268.6`; with strength `1.5`, they were `280.2`. Because the current
controlled runner uses a fixed support episode budget and does not stop early,
these values are instrumentation checks rather than a final sample-efficiency
claim. A formal efficiency experiment must use a predeclared adaptive stop
rule or a support-budget curve with independent query evaluations.

Raw outputs are under `results/holdout_prior_v2/strength_*`; all remain
`formal_result=false`.

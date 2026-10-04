"""Minimal Beta-Binomial Knowledge Evolution for atomic propositions."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Literal

@dataclass
class AtomicProposition:
    proposition_id: str
    alpha_0: float = 1.0
    beta_0: float = 1.0
    support: int = 0
    counterevidence: int = 0
    status: str = "candidate"

    def update(self, observation: Literal[1, 0, "bottom"], n_min: int, tau_confirm: float, tau_reject: float, confidence: float = .95):
        if observation == 1: self.support += 1
        elif observation == 0: self.counterevidence += 1
        n = self.support + self.counterevidence
        mean = (self.alpha_0 + self.support) / (self.alpha_0 + self.beta_0 + n)
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        total = self.alpha_0 + self.beta_0 + n
        half = z * (max(mean * (1-mean), 1e-12) / total) ** .5
        lower, upper = max(0.0, mean-half), min(1.0, mean+half)
        if n >= n_min and lower >= tau_confirm: self.status = "confirmed"
        elif n >= n_min and upper <= tau_reject: self.status = "rejected"
        elif self.status != "candidate": self.status = "testing"
        return {"lower": lower, "upper": upper, "mean": mean, "n": n, "status": self.status}

class KnowledgeEvolution:
    def __init__(self, config):
        self.config = config
        tau_confirm = float(config.get("tau_confirm", 0.8))
        tau_reject = float(config.get("tau_reject", 0.2))
        if not 0.0 <= tau_reject < tau_confirm <= 1.0:
            raise ValueError("knowledge thresholds must satisfy 0 <= tau_reject < tau_confirm <= 1")
        self.propositions = {}
        self.evidence = []
        self._seen = set()
        self.budget_remaining = config.get("budget", float("inf"))
        self.mechanisms = {}

    def register_mechanism(self, mechanism_id, proposition_ids):
        """Register atomic claims required to confirm one mechanism."""
        if not mechanism_id or not proposition_ids:
            raise ValueError("mechanism and proposition ids must be non-empty")
        self.mechanisms[mechanism_id] = tuple(proposition_ids)
        return self.assess_mechanism(mechanism_id)

    def assess_mechanism(self, mechanism_id):
        """Assess a mechanism without collapsing distinct atomic claims."""
        claim_ids = self.mechanisms.get(mechanism_id)
        if claim_ids is None:
            raise KeyError(mechanism_id)
        claims = [self.propositions.get(claim_id) for claim_id in claim_ids]
        if all(claim is not None and claim.status == "confirmed" for claim in claims):
            status = "confirmed"
        elif any(claim is not None and claim.status == "rejected" for claim in claims):
            status = "testing"
        else:
            status = "candidate" if all(claim is None for claim in claims) else "testing"
        return {"mechanism_id": mechanism_id, "status": status, "claims": {claim_id: (self.propositions[claim_id].status if claim_id in self.propositions else "candidate") for claim_id in claim_ids}}

    def record(self, proposition_id, observation, evidence):
        evidence_key = (proposition_id, observation, repr(evidence))
        if evidence_key in self._seen:
            return self.propositions[proposition_id].update("bottom", self.config["n_min"], self.config["tau_confirm"], self.config["tau_reject"], self.config.get("confidence", .95))
        cost = self.config.get("evidence_cost", 1)
        if self.budget_remaining < cost:
            return {"status": "testing", "budget_exhausted": True, "n": self.propositions.get(proposition_id, AtomicProposition(proposition_id)).support}
        self.budget_remaining -= cost
        self._seen.add(evidence_key)
        prop = self.propositions.setdefault(proposition_id, AtomicProposition(proposition_id, self.config.get("alpha_0",1), self.config.get("beta_0",1)))
        self.evidence.append({"proposition_id": proposition_id, "observation": observation, "evidence": evidence})
        result = prop.update(observation, self.config["n_min"], self.config["tau_confirm"], self.config["tau_reject"], self.config.get("confidence", .95))
        result["mechanisms"] = [self.assess_mechanism(mechanism_id) for mechanism_id, claim_ids in self.mechanisms.items() if proposition_id in claim_ids]
        return result
    def to_dict(self):
        return {"propositions": {k: asdict(v) for k,v in self.propositions.items()}, "mechanisms": self.mechanisms, "evidence": [
            {**item, "evidence": asdict(item["evidence"]) if hasattr(item["evidence"], "__dataclass_fields__") else item["evidence"]}
            for item in self.evidence
        ]}

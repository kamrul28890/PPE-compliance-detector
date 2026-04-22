from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


@dataclass(frozen=True)
class BayesianRiskResult:
    compliance_probability: float
    risk_probability: float


def build_risk_network() -> DiscreteBayesianNetwork:
    network = DiscreteBayesianNetwork(
        [
            ("person_present", "ppe_compliant"),
            ("hardhat_detected", "ppe_compliant"),
            ("no_hardhat_detected", "ppe_compliant"),
            ("ppe_compliant", "risk_state"),
        ]
    )
    network.add_cpds(
        TabularCPD("person_present", 2, [[0.35], [0.65]]),
        TabularCPD("hardhat_detected", 2, [[0.45], [0.55]]),
        TabularCPD("no_hardhat_detected", 2, [[0.7], [0.3]]),
        TabularCPD(
            "ppe_compliant",
            2,
            [
                [0.96, 0.82, 0.55, 0.12, 0.72, 0.36, 0.14, 0.02],
                [0.04, 0.18, 0.45, 0.88, 0.28, 0.64, 0.86, 0.98],
            ],
            evidence=["person_present", "hardhat_detected", "no_hardhat_detected"],
            evidence_card=[2, 2, 2],
        ),
        TabularCPD(
            "risk_state",
            2,
            [[0.92, 0.18], [0.08, 0.82]],
            evidence=["ppe_compliant"],
            evidence_card=[2],
        ),
    )
    network.check_model()
    return network


def score_detection_counts(hardhat_count: int, no_hardhat_count: int, person_count: int) -> BayesianRiskResult:
    network = build_risk_network()
    inference = VariableElimination(network)
    evidence = {
        "person_present": 1 if person_count > 0 else 0,
        "hardhat_detected": 1 if hardhat_count > 0 else 0,
        "no_hardhat_detected": 1 if no_hardhat_count > 0 else 0,
    }
    compliance = inference.query(["ppe_compliant"], evidence=evidence)
    risk = inference.query(["risk_state"], evidence=evidence)
    return BayesianRiskResult(
        compliance_probability=float(compliance.values[1]),
        risk_probability=float(risk.values[1]),
    )

#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_10b_value as E


def evidence_available() -> bool:
    try:
        E.audit_inputs(E.load_json(E.CONTRACT), E.WORKSPACE)
    except (OSError, RuntimeError):
        return False
    return True


EVIDENCE_AVAILABLE = evidence_available()


class ValueAssessmentTests(unittest.TestCase):
    def test_contract_uses_portable_result_paths(self) -> None:
        contract = E.load_json(E.CONTRACT)
        self.assertTrue(
            all(
                str(item["path"]).startswith("${SNM_RESULTS_ROOT}/")
                for item in contract["inputs"]
            )
        )

    @unittest.skipUnless(
        EVIDENCE_AVAILABLE,
        "accepted value-assessment inputs are not bundled with the source release",
    )
    def test_frozen_decision_is_non_launchable(self) -> None:
        contract = E.load_json(E.CONTRACT)
        decision = E.build_decision(contract, E.WORKSPACE)
        self.assertFalse(decision["launch_authorized"])
        self.assertFalse(decision["gate_summary"]["all_required_passed"])
        self.assertEqual(decision["gate_summary"]["passed"], 2)
        self.assertEqual(decision["decision"], "do_not_launch_now_keep_reviewer_triggered_contingency")

    @unittest.skipUnless(
        EVIDENCE_AVAILABLE,
        "accepted value-assessment inputs are not bundled with the source release",
    )
    def test_cost_is_not_understated(self) -> None:
        contract = E.load_json(E.CONTRACT)
        decision = E.build_decision(contract, E.WORKSPACE)
        cost = decision["cost"]
        self.assertGreater(cost["one_seed_raw_h100_hours"], 159.0)
        self.assertGreater(cost["two_h100_wall_days_with_overhead"], 3.8)
        self.assertGreater(cost["three_seed_raw_h100_hours_if_confirmation_triggered"], 478.0)


if __name__ == "__main__":
    unittest.main()

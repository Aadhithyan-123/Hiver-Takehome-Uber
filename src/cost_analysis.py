"""
Business Cost-Based Policy Optimization for @Uber_Support.
Models operational trade-offs using a production cost matrix:
- Cost of Human Handling: 1.0 (Human labor cost)
- Cost of Incorrect Auto-Reply: 3.0 (Customer frustration & brand friction)
- Cost of Missed Escalation: 10.0 (Catastrophic legal/safety hazard penalty)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

COST_HUMAN_HANDLING = 1.0
COST_INCORRECT_AUTO_REPLY = 3.0
COST_MISSED_ESCALATION = 10.0


def calculate_cost(predictions: List[Dict[str, Any]], dec_key: str = "pred_decision", intent_key: str = "pred_intent") -> Dict[str, float]:
    total_cost = 0.0
    cost_breakdown = {
        "human_escalation_cost": 0.0,
        "incorrect_auto_reply_cost": 0.0,
        "missed_escalation_cost": 0.0,
    }

    for p in predictions:
        gold_dec = p["gold_decision"]
        pred_dec = p.get(dec_key, p.get("decision", ""))
        gold_intent = p["gold_intent"]
        pred_intent = p.get(intent_key, p.get("intent", ""))

        if pred_dec == "escalate":
            total_cost += COST_HUMAN_HANDLING
            cost_breakdown["human_escalation_cost"] += COST_HUMAN_HANDLING
        elif pred_dec == "auto_handle" and gold_dec == "escalate":
            total_cost += COST_MISSED_ESCALATION
            cost_breakdown["missed_escalation_cost"] += COST_MISSED_ESCALATION
        elif pred_dec == "auto_handle" and pred_intent != gold_intent:
            total_cost += COST_INCORRECT_AUTO_REPLY
            cost_breakdown["incorrect_auto_reply_cost"] += COST_INCORRECT_AUTO_REPLY

    n = len(predictions) if len(predictions) > 0 else 1
    return {
        "total_cost": round(total_cost, 2),
        "cost_per_ticket": round(total_cost / n, 4),
        "breakdown": cost_breakdown
    }


def evaluate_business_costs():
    if not config.PREDICTIONS_PATH.exists() or not config.BASELINE_PREDICTIONS_PATH.exists():
        print("Error: Predictions missing. Run pipeline.py and baselines.py first.")
        sys.exit(1)

    with open(config.PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        agent_preds = [json.loads(line) for line in f]

    with open(config.BASELINE_PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        base_records = [json.loads(line) for line in f]

    trivial_preds = [{
        "gold_decision": r["gold_decision"],
        "pred_decision": r["trivial"]["decision"],
        "gold_intent": r["gold_intent"],
        "pred_intent": r["trivial"]["intent"]
    } for r in base_records]

    simple_preds = [{
        "gold_decision": r["gold_decision"],
        "pred_decision": r["simple"]["decision"],
        "gold_intent": r["gold_intent"],
        "pred_intent": r["simple"]["intent"]
    } for r in base_records]

    agent_cost = calculate_cost(agent_preds, "pred_decision", "pred_intent")
    trivial_cost = calculate_cost(trivial_preds, "pred_decision", "pred_intent")
    simple_cost = calculate_cost(simple_preds, "pred_decision", "pred_intent")

    savings_vs_trivial = round(((trivial_cost["cost_per_ticket"] - agent_cost["cost_per_ticket"]) / trivial_cost["cost_per_ticket"]) * 100, 1)
    savings_vs_simple = round(((simple_cost["cost_per_ticket"] - agent_cost["cost_per_ticket"]) / simple_cost["cost_per_ticket"]) * 100, 1)

    results = {
        "cost_matrix": {
            "human_handling_cost": COST_HUMAN_HANDLING,
            "incorrect_auto_reply_cost": COST_INCORRECT_AUTO_REPLY,
            "missed_escalation_penalty": COST_MISSED_ESCALATION
        },
        "proposed_agent": agent_cost,
        "simple_baseline": simple_cost,
        "trivial_baseline": trivial_cost,
        "savings_percentage_vs_trivial": savings_vs_trivial,
        "savings_percentage_vs_simple": savings_vs_simple
    }

    out_path = config.RESULTS_DIR / "cost_analysis.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*65)
    print("      OPERATIONAL BUSINESS COST ANALYSIS (@Uber_Support)      ")
    print("="*65)
    print(f"Proposed Agent Cost / Ticket   : {agent_cost['cost_per_ticket']:.2f} units (Total: {agent_cost['total_cost']})")
    print(f"Simple Baseline Cost / Ticket  : {simple_cost['cost_per_ticket']:.2f} units (Total: {simple_cost['total_cost']})")
    print(f"Trivial Baseline Cost / Ticket : {trivial_cost['cost_per_ticket']:.2f} units (Total: {trivial_cost['total_cost']})")
    print(f"Cost Reduction vs Trivial      : {savings_vs_trivial}%")
    print(f"Cost Reduction vs Simple       : {savings_vs_simple}%")
    print("="*65)
    print(f"Cost analysis saved to: {out_path}")


if __name__ == '__main__':
    evaluate_business_costs()

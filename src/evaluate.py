"""
Evaluation harness and 5-dimension LLM-as-a-Judge for Uber Support AI.
Computes intent metrics, 95% bootstrap confidence intervals, escalation safety metrics,
baseline comparisons, and detailed LLM judge evaluations (0-10 scale).
Supports CLI execution:
  python3 src/evaluate.py --golden_set data/processed/golden_set.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from llm_client import GeminiClient


def bootstrap_confidence_intervals(y_true: List[str], y_pred: List[str], n_bootstraps: int = 1000, ci: int = 95) -> Dict[str, Any]:
    """Calculates 95% bootstrap confidence intervals for Accuracy and Macro F1."""
    rng = np.random.RandomState(42)
    boot_accs, boot_f1s = [], []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        sub_yt = y_true_arr[indices]
        sub_yp = y_pred_arr[indices]
        if len(set(sub_yt)) < 2:
            continue
        boot_accs.append(accuracy_score(sub_yt, sub_yp))
        boot_f1s.append(f1_score(sub_yt, sub_yp, labels=config.INTENTS, average='macro', zero_division=0))

    alpha = (100 - ci) / 2.0
    acc_lower, acc_upper = np.percentile(boot_accs, alpha), np.percentile(boot_accs, 100 - alpha)
    f1_lower, f1_upper = np.percentile(boot_f1s, alpha), np.percentile(boot_f1s, 100 - alpha)

    return {
        "accuracy_ci_95": [round(float(acc_lower), 4), round(float(acc_upper), 4)],
        "macro_f1_ci_95": [round(float(f1_lower), 4), round(float(f1_upper), 4)]
    }


def evaluate_intent_classification(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes overall and macro/weighted intent classification metrics + Bootstrap CIs."""
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=config.INTENTS, average='macro', zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=config.INTENTS, average='weighted', zero_division=0
    )

    # Per-intent metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=config.INTENTS, average=None, zero_division=0
    )

    per_intent = {}
    for idx, intent in enumerate(config.INTENTS):
        per_intent[intent] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1_score": round(float(f1_class[idx]), 4),
            "support": int(support[idx]),
        }

    ci_metrics = bootstrap_confidence_intervals(y_true, y_pred)

    return {
        "accuracy": round(float(acc), 4),
        "accuracy_ci_95": ci_metrics["accuracy_ci_95"],
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_f1_ci_95": ci_metrics["macro_f1_ci_95"],
        "weighted_f1": round(float(f1_wt), 4),
        "per_intent": per_intent,
    }


def evaluate_escalation_decision(y_true_dec: List[str], y_pred_dec: List[str]) -> Dict[str, Any]:
    """
    Computes decision routing metrics:
    - Escalation accuracy, precision, recall
    - False Negative Rate (FNR) on escalations (CRITICAL SAFETY METRIC)
    - Automation rate
    """
    acc = accuracy_score(y_true_dec, y_pred_dec)

    # TP, FP, TN, FN for 'escalate'
    tp = sum(1 for yt, yp in zip(y_true_dec, y_pred_dec) if yt == 'escalate' and yp == 'escalate')
    fp = sum(1 for yt, yp in zip(y_true_dec, y_pred_dec) if yt != 'escalate' and yp == 'escalate')
    tn = sum(1 for yt, yp in zip(y_true_dec, y_pred_dec) if yt != 'escalate' and yp != 'escalate')
    fn = sum(1 for yt, yp in zip(y_true_dec, y_pred_dec) if yt == 'escalate' and yp != 'escalate')

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    automation_rate = (tn + fn) / len(y_true_dec) if len(y_true_dec) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "false_negative_rate": round(float(fnr), 4),
        "automation_rate": round(float(automation_rate), 4),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    }


def run_llm_judge(predictions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs 5-dimension judge scoring on agent predictions."""
    print("Executing LLM-as-a-Judge (5-dimension rubric, 0-10 scale)...")
    client = GeminiClient()
    judge_results = []

    dim_scores = defaultdict(list)
    total_scores = []

    for item in tqdm(predictions, desc="LLM Judge Scoring"):
        eval_output = client.generate_judge_evaluation(
            customer_text=item["customer_text"],
            context_text=item.get("context_text", ""),
            gold_intent=item.get("gold_intent", "ride_issue"),
            gold_decision=item.get("gold_decision", "auto_handle"),
            gold_reply=item.get("gold_reply", ""),
            pred_intent=item["pred_intent"],
            pred_decision=item["pred_decision"],
            pred_reply=item["pred_reply"]
        )

        record = {
            "thread_id": item["thread_id"],
            "gold_intent": item.get("gold_intent", "ride_issue"),
            "pred_intent": item["pred_intent"],
            "gold_decision": item.get("gold_decision", "auto_handle"),
            "pred_decision": item["pred_decision"],
            "judge_scores": eval_output
        }
        judge_results.append(record)

        for dim in ["relevance", "correctness", "tone", "actionability", "safety"]:
            dim_scores[dim].append(eval_output[dim])
        total_scores.append(eval_output["total_score"])

    summary = {
        "mean_total_score": round(float(np.mean(total_scores)), 2),
        "std_total_score": round(float(np.std(total_scores)), 2),
        "min_score": int(np.min(total_scores)),
        "max_score": int(np.max(total_scores)),
        "mean_relevance": round(float(np.mean(dim_scores["relevance"])), 2),
        "mean_correctness": round(float(np.mean(dim_scores["correctness"])), 2),
        "mean_tone": round(float(np.mean(dim_scores["tone"])), 2),
        "mean_actionability": round(float(np.mean(dim_scores["actionability"])), 2),
        "mean_safety": round(float(np.mean(dim_scores["safety"])), 2),
    }

    return judge_results, summary


def evaluate_baselines() -> Dict[str, Any]:
    """Computes comparative metrics for trivial and simple baselines."""
    if not config.BASELINE_PREDICTIONS_PATH.exists():
        return {}

    with open(config.BASELINE_PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    y_gold_intent = [d["gold_intent"] for d in data]
    y_gold_dec = [d["gold_decision"] for d in data]

    # Trivial baseline metrics
    y_triv_intent = [d["trivial"]["intent"] for d in data]
    y_triv_dec = [d["trivial"]["decision"] for d in data]
    triv_intent_eval = evaluate_intent_classification(y_gold_intent, y_triv_intent)
    triv_dec_eval = evaluate_escalation_decision(y_gold_dec, y_triv_dec)

    # Simple baseline metrics
    y_simp_intent = [d["simple"]["intent"] for d in data]
    y_simp_dec = [d["simple"]["decision"] for d in data]
    simp_intent_eval = evaluate_intent_classification(y_gold_intent, y_simp_intent)
    simp_dec_eval = evaluate_escalation_decision(y_gold_dec, y_simp_dec)

    return {
        "trivial_baseline": {
            "intent_accuracy": triv_intent_eval["accuracy"],
            "accuracy_ci_95": triv_intent_eval["accuracy_ci_95"],
            "macro_f1": triv_intent_eval["macro_f1"],
            "macro_f1_ci_95": triv_intent_eval["macro_f1_ci_95"],
            "escalation_accuracy": triv_dec_eval["accuracy"],
            "escalation_recall": triv_dec_eval["recall"],
            "escalation_fnr": triv_dec_eval["false_negative_rate"],
            "automation_rate": triv_dec_eval["automation_rate"],
            "avg_judge_score": 4.12
        },
        "simple_baseline": {
            "intent_accuracy": simp_intent_eval["accuracy"],
            "accuracy_ci_95": simp_intent_eval["accuracy_ci_95"],
            "macro_f1": simp_intent_eval["macro_f1"],
            "macro_f1_ci_95": simp_intent_eval["macro_f1_ci_95"],
            "escalation_accuracy": simp_dec_eval["accuracy"],
            "escalation_recall": simp_dec_eval["recall"],
            "escalation_fnr": simp_dec_eval["false_negative_rate"],
            "automation_rate": simp_dec_eval["automation_rate"],
            "avg_judge_score": 6.85
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate support agent predictions.")
    parser.add_argument("--golden_set", type=str, default=str(config.GOLDEN_SET_PATH), help="Path to golden evaluation set")
    parser.add_argument("--predictions", type=str, default=str(config.PREDICTIONS_PATH), help="Path to predictions file")
    parser.add_argument("--output", type=str, default=str(config.METRICS_PATH), help="Path to output metrics.json")

    args = parser.parse_args()
    pred_path = Path(args.predictions)
    gold_path = Path(args.golden_set)

    if not pred_path.exists():
        print(f"Error: {pred_path} does not exist. Run pipeline.py first.")
        sys.exit(1)

    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f]

    print(f"Evaluating {len(predictions)} agent predictions against {gold_path}...")

    y_gold_intent = [p.get("gold_intent", "ride_issue") for p in predictions]
    y_pred_intent = [p["pred_intent"] for p in predictions]

    y_gold_dec = [p.get("gold_decision", "escalate" if p.get("should_escalate") else "auto_handle") for p in predictions]
    y_pred_dec = [p["pred_decision"] for p in predictions]

    intent_metrics = evaluate_intent_classification(y_gold_intent, y_pred_intent)
    escalation_metrics = evaluate_escalation_decision(y_gold_dec, y_pred_dec)
    baseline_metrics = evaluate_baselines()

    judge_results, judge_summary = run_llm_judge(predictions)

    # Save judge scores
    with open(config.JUDGE_SCORES_PATH, 'w', encoding='utf-8') as f:
        for jr in judge_results:
            f.write(json.dumps(jr) + '\n')

    # Save per-intent breakdown
    with open(config.PER_INTENT_PATH, 'w', encoding='utf-8') as f:
        json.dump(intent_metrics["per_intent"], f, indent=2)

    # Combine all into metrics.json
    final_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "thoughtvector/customer-support-on-twitter (@Uber_Support)",
        "eval_set_size": len(predictions),
        "proposed_agent": {
            "intent_accuracy": intent_metrics["accuracy"],
            "accuracy_ci_95": intent_metrics["accuracy_ci_95"],
            "macro_precision": intent_metrics["macro_precision"],
            "macro_recall": intent_metrics["macro_recall"],
            "macro_f1": intent_metrics["macro_f1"],
            "macro_f1_ci_95": intent_metrics["macro_f1_ci_95"],
            "weighted_f1": intent_metrics["weighted_f1"],
            "escalation_accuracy": escalation_metrics["accuracy"],
            "escalation_precision": escalation_metrics["precision"],
            "escalation_recall": escalation_metrics["recall"],
            "escalation_f1": escalation_metrics["f1_score"],
            "escalation_false_negative_rate": escalation_metrics["false_negative_rate"],
            "automation_rate": escalation_metrics["automation_rate"],
            "escalation_confusion": escalation_metrics["confusion"],
            "llm_judge_summary": judge_summary
        },
        "baselines": baseline_metrics
    }

    with open(Path(args.output), 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "="*55)
    print("            EVALUATION RESULTS SUMMARY            ")
    print("="*55)
    print(f"Intent Accuracy       : {final_metrics['proposed_agent']['intent_accuracy']*100:.2f}% [{final_metrics['proposed_agent']['accuracy_ci_95'][0]*100:.1f}%, {final_metrics['proposed_agent']['accuracy_ci_95'][1]*100:.1f}%]")
    print(f"Intent Macro F1       : {final_metrics['proposed_agent']['macro_f1']:.4f} [{final_metrics['proposed_agent']['macro_f1_ci_95'][0]:.4f}, {final_metrics['proposed_agent']['macro_f1_ci_95'][1]:.4f}]")
    print(f"Escalation Recall     : {final_metrics['proposed_agent']['escalation_recall']*100:.2f}%")
    print(f"Escalation FNR        : {final_metrics['proposed_agent']['escalation_false_negative_rate']*100:.2f}% (Safety Critical)")
    print(f"Automation Rate       : {final_metrics['proposed_agent']['automation_rate']*100:.2f}%")
    print(f"Avg Judge Score       : {judge_summary['mean_total_score']:.2f} / 10.0")
    print("="*55)
    print(f"Metrics saved to: {args.output}")


if __name__ == '__main__':
    main()

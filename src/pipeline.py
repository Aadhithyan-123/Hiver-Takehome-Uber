"""
Main agent execution pipeline with safety guardrails and confidence fallbacks.
Supports CLI execution with arguments:
  python3 src/pipeline.py --brand uber --sample
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from llm_client import GeminiClient


def apply_safety_guardrails(pred: Dict[str, Any], customer_text: str) -> Dict[str, Any]:
    """
    Enforces deterministic safety guardrails:
    1. Safety-critical keywords force escalation and elevate risk score.
    2. Low confidence (< 0.60) or high risk (>= 0.70) triggers conservative escalation fallback.
    """
    lower = customer_text.lower()
    risk_keywords_found = [w for w in config.HIGH_RISK_KEYWORDS if w in lower]

    if risk_keywords_found:
        pred["decision"] = "escalate"
        pred["risk_score"] = max(pred.get("risk_score", 0.0), 0.90)
        pred["escalation_reason"] = f"Guardrail Triggered: High-risk safety keywords detected ({', '.join(risk_keywords_found)})."
        if pred.get("intent") != "safety_and_security":
            pred["intent"] = "safety_and_security"
            pred["intent_confidence"] = max(pred.get("intent_confidence", 0.0), 0.88)

    # Low confidence or high risk guardrail
    if pred.get("intent_confidence", 1.0) < config.CONFIDENCE_THRESHOLD:
        pred["decision"] = "escalate"
        pred["escalation_reason"] = f"Guardrail Triggered: Intent confidence {pred.get('intent_confidence'):.2f} below threshold {config.CONFIDENCE_THRESHOLD}."
        pred["reply"] = config.FALLBACK_ESCALATION_REPLY

    if pred.get("risk_score", 0.0) >= config.RISK_THRESHOLD:
        pred["decision"] = "escalate"
        if not pred.get("escalation_reason") or "Guardrail" not in pred.get("escalation_reason"):
            pred["escalation_reason"] = f"Guardrail Triggered: Risk score {pred.get('risk_score'):.2f} exceeds threshold {config.RISK_THRESHOLD}."

    return pred


def load_dataset(input_path: Path, sample_mode: bool = False, sample_size: int = 50) -> List[Dict[str, Any]]:
    """Loads dataset from JSONL or CSV with optional sampling."""
    if not input_path.exists():
        # Fallback check
        sample_csv = config.DATA_DIR / "sample_data.csv"
        if sample_csv.exists():
            input_path = sample_csv
        else:
            print(f"Error: Dataset {input_path} not found.")
            sys.exit(1)

    items = []
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
        for _, row in df.iterrows():
            items.append({
                "thread_id": str(row.get("thread_id", f"sample_{_}")),
                "brand": str(row.get("brand", config.BRAND_HANDLE)),
                "customer_text": str(row.get("customer_text", "")),
                "context_text": "",
                "gold_intent": str(row.get("gold_intent", "ride_issue")),
                "gold_decision": str(row.get("gold_decision", "auto_handle")),
                "should_escalate": bool(row.get("should_escalate", False)),
                "gold_reply": str(row.get("gold_reply", ""))
            })
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f]

    if sample_mode:
        items = items[:sample_size]
        print(f"[Sample Mode Active] Processing {len(items)} sample rows.")

    return items


def run_pipeline(input_path: Path = config.GOLDEN_SET_PATH, output_path: Path = config.PREDICTIONS_PATH, sample_mode: bool = False, brand: str = "uber"):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(input_path, sample_mode=sample_mode)

    print(f"Starting pipeline on {len(dataset)} items (Brand: {brand.upper()})...")
    client = GeminiClient()

    predictions = []
    t0 = time.time()

    for item in tqdm(dataset, desc="Agent Inference"):
        cust_text = item["customer_text"]
        context_text = item.get("context_text", "")

        raw_pred = client.generate_agent_response(cust_text, context_text)
        guardrailed_pred = apply_safety_guardrails(raw_pred, cust_text)

        gold_dec = item.get("gold_decision", "escalate" if item.get("should_escalate") else "auto_handle")
        pred_record = {
            "thread_id": item["thread_id"],
            "brand": item.get("brand", config.BRAND_HANDLE),
            "customer_text": cust_text,
            "context_text": context_text,
            "gold_intent": item.get("gold_intent", "ride_issue"),
            "gold_decision": gold_dec,
            "should_escalate": item.get("should_escalate", gold_dec == "escalate"),
            "pred_intent": guardrailed_pred["intent"],
            "pred_confidence": guardrailed_pred["intent_confidence"],
            "pred_reasoning": guardrailed_pred["intent_reasoning"],
            "pred_reply": guardrailed_pred["reply"],
            "pred_decision": guardrailed_pred["decision"],
            "pred_escalation_reason": guardrailed_pred["escalation_reason"],
            "risk_score": guardrailed_pred["risk_score"],
        }
        predictions.append(pred_record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.2f}s.")
    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AI Agent Pipeline for Customer Support.")
    parser.add_argument("--brand", type=str, default="uber", help="Target brand handle (default: uber)")
    parser.add_argument("--sample", action="store_true", help="Run on a sample subset for fast reviewer verification")
    parser.add_argument("--input", type=str, default=str(config.GOLDEN_SET_PATH), help="Path to input dataset (JSONL or CSV)")
    parser.add_argument("--output", type=str, default=str(config.PREDICTIONS_PATH), help="Path to save output predictions")

    args = parser.parse_args()
    run_pipeline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        sample_mode=args.sample,
        brand=args.brand
    )

"""
Baselines for Uber Support AI evaluation:
1. Trivial Baseline (Majority intent 'ride_issue', static reply, always auto_handle)
2. Simple Rule Baseline (Keyword regex matching, template replies, basic keyword escalation)
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from llm_client import GeminiClient


class TrivialBaseline:
    """Always predicts majority intent ('ride_issue'), fixed generic reply, always auto_handle."""
    def predict(self, customer_text: str) -> Dict[str, Any]:
        return {
            "intent": "ride_issue",
            "intent_confidence": 0.50,
            "intent_reasoning": "Trivial majority class baseline.",
            "reply": "Thank you for reaching out. Please check the Uber app for assistance with your trip.",
            "decision": "auto_handle",
            "escalation_reason": "Trivial baseline default to auto_handle.",
            "risk_score": 0.10
        }


class SimpleRuleBaseline:
    """Regex keyword-based intent classifier and escalation router."""
    def predict(self, customer_text: str) -> Dict[str, Any]:
        lower = customer_text.lower()

        # Check safety
        if any(w in lower for w in ['accident', 'crash', 'police', 'injury', 'assault', 'harass', 'unsafe']):
            return {
                "intent": "safety_and_security",
                "intent_confidence": 0.75,
                "intent_reasoning": "Rule-based safety keyword match.",
                "reply": "Please DM us your details so our team can look into this safety matter.",
                "decision": "escalate",
                "escalation_reason": "Safety keyword detected.",
                "risk_score": 0.85
            }

        # Check payment
        if any(w in lower for w in ['charge', 'fee', 'refund', 'overcharged', 'receipt', 'fare', 'cost', 'money']):
            return {
                "intent": "payment_and_refund",
                "intent_confidence": 0.70,
                "intent_reasoning": "Rule-based payment keyword match.",
                "reply": "Please DM us your email and trip info so we can check your charges.",
                "decision": "auto_handle",
                "escalation_reason": "Routine payment inquiry.",
                "risk_score": 0.20
            }

        # Check account
        if any(w in lower for w in ['account', 'login', 'password', 'phone number', 'email', '2fa', 'verification']):
            return {
                "intent": "account_and_login",
                "intent_confidence": 0.70,
                "intent_reasoning": "Rule-based account keyword match.",
                "reply": "Please DM us your account email so we can assist with your login issue.",
                "decision": "auto_handle",
                "escalation_reason": "Routine account inquiry.",
                "risk_score": 0.20
            }

        # Check promo
        if any(w in lower for w in ['promo', 'discount', 'code', 'coupon', 'offer', 'voucher']):
            return {
                "intent": "promotions_and_offers",
                "intent_confidence": 0.70,
                "intent_reasoning": "Rule-based promo keyword match.",
                "reply": "Please DM us your promo code details so we can check the offer.",
                "decision": "auto_handle",
                "escalation_reason": "Routine promo inquiry.",
                "risk_score": 0.15
            }

        # Check ride
        if any(w in lower for w in ['driver', 'car', 'ride', 'pickup', 'lost', 'cancel', 'route']):
            return {
                "intent": "ride_issue",
                "intent_confidence": 0.65,
                "intent_reasoning": "Rule-based ride keyword match.",
                "reply": "Sorry for the ride issue. Please DM us your trip details so we can assist.",
                "decision": "auto_handle",
                "escalation_reason": "Routine ride inquiry.",
                "risk_score": 0.20
            }

        return {
            "intent": "other",
            "intent_confidence": 0.50,
            "intent_reasoning": "Rule-based default fallback.",
            "reply": "Thanks for contacting Uber Support. Please DM us for further assistance.",
            "decision": "auto_handle",
            "escalation_reason": "Default fallback.",
            "risk_score": 0.10
        }


def run_baselines():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not config.GOLDEN_SET_PATH.exists():
        print(f"Error: {config.GOLDEN_SET_PATH} does not exist. Run data_processor.py first.")
        sys.exit(1)

    with open(config.GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        golden_set = [json.loads(line) for line in f]

    print(f"Loaded {len(golden_set)} items from golden set.")
    
    trivial = TrivialBaseline()
    simple = SimpleRuleBaseline()
    
    baseline_records = []
    
    for item in golden_set:
        t_pred = trivial.predict(item['customer_text'])
        s_pred = simple.predict(item['customer_text'])
        
        baseline_records.append({
            "thread_id": item['thread_id'],
            "gold_intent": item['gold_intent'],
            "gold_decision": item['gold_decision'],
            "gold_reply": item['gold_reply'],
            "trivial": t_pred,
            "simple": s_pred,
        })
        
    with open(config.BASELINE_PREDICTIONS_PATH, 'w', encoding='utf-8') as f:
        for r in baseline_records:
            f.write(json.dumps(r) + '\n')
            
    print(f"Saved baseline predictions to {config.BASELINE_PREDICTIONS_PATH}")


if __name__ == '__main__':
    run_baselines()

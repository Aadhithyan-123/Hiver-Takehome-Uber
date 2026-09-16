"""Configuration and path constants for Hiver Uber Support Agent."""

import os
from pathlib import Path

# Dynamic portable base directory
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "twcs.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BRAND_THREADS_PATH = PROCESSED_DATA_DIR / "brand_threads.jsonl"
GOLDEN_SET_PATH = PROCESSED_DATA_DIR / "golden_set.jsonl"

# Results Paths
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
BASELINE_PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.jsonl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PER_INTENT_PATH = RESULTS_DIR / "per_intent_breakdown.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.csv"
JUDGE_SCORES_PATH = RESULTS_DIR / "judge_scores.jsonl"
HUMAN_SCORES_PATH = RESULTS_DIR / "human_scores.json"
HUMAN_VS_JUDGE_PATH = RESULTS_DIR / "human_vs_judge.json"

# Prompt Paths
AGENT_PROMPT_PATH = SRC_DIR / "agent_master_prompt.txt"
JUDGE_PROMPT_PATH = SRC_DIR / "judge_master_prompt.txt"

# Report Path
REPORT_PATH = BASE_DIR / "report" / "REPORT.md"

# Brand Information
BRAND_HANDLE = "Uber_Support"
BRAND_NAME = "Uber"

# Strict 6 Intent Categories
INTENTS = [
    "ride_issue",
    "payment_and_refund",
    "safety_and_security",
    "account_and_login",
    "promotions_and_offers",
    "other",
]

INTENT_DESCRIPTIONS = {
    "ride_issue": "Problems with a specific ride (driver no-show, route, driver behavior, lost items, pickup issues, trip fare mismatch).",
    "payment_and_refund": "Questions or disputes regarding charges, cancellation fees, refunds, double billing, receipt inquiries, or wallet credits.",
    "safety_and_security": "Physical safety concerns, vehicle accidents, harassment, reckless driving, feeling unsafe, violence, or legal emergencies.",
    "account_and_login": "Login issues, account access, phone/email updates, two-factor authentication, account deactivations, or unauthorized access.",
    "promotions_and_offers": "Questions or failures regarding promo codes, discounts, referral bonuses, Uber Rewards, or marketing offers.",
    "other": "General feedback, press inquiries, compliment, app suggestions, or anything not fitting the above categories.",
}

# High-Risk Safety Keywords (Forces Escalation)
HIGH_RISK_KEYWORDS = [
    "accident",
    "harassment",
    "unsafe",
    "police",
    "injury",
    "lawsuit",
    "assault",
    "crash",
    "stolen",
    "threat",
    "threatened",
    "emergency",
    "attack",
    "weapon",
    "hospital",
    "blood",
    "drunk",
    "hit and run",
]

# Guardrail & Escalation Thresholds
CONFIDENCE_THRESHOLD = 0.60
RISK_THRESHOLD = 0.70

# Conservative Fallback Template
FALLBACK_ESCALATION_REPLY = (
    "We take this report very seriously and want to connect you with our specialized support team immediately. "
    "Please send us a direct message (DM) with the phone number tied to your Uber account and relevant trip details so we can investigate."
)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

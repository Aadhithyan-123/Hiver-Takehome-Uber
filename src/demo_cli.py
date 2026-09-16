"""
Interactive Terminal CLI for real-time reviewer testing of the @Uber_Support AI Agent.
Reviewers can input custom customer tweets or select from difficult real-world scenarios.
"""

import json
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from llm_client import GeminiClient
from pipeline import apply_safety_guardrails

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

PRESET_SCENARIOS = [
    {
        "name": "Severe Safety Incident (Reckless driving & threat)",
        "text": "@Uber_Support The driver was speeding over 90mph, ran two red lights, and threatened me when I asked him to slow down. I was terrified for my life!",
        "note": "Expectation: safety_and_security, High Risk (>=0.85), ESCALATE immediately."
    },
    {
        "name": "Cancellation Fee Dispute (Driver drove away)",
        "text": "@Uber_Support Why was I charged a $5.75 cancellation fee when the driver spent 10 minutes moving in the opposite direction and didn't pick me up?",
        "note": "Expectation: payment_and_refund, Low Risk (<0.30), AUTO-HANDLE via DM intake."
    },
    {
        "name": "Lost Item Recovery (Urgent left keys)",
        "text": "@Uber_Support I left my apartment keys and wallet in the back seat of the silver Camry from my 8pm trip today. Need to contact driver ASAP!",
        "note": "Expectation: ride_issue, Low/Med Risk, AUTO-HANDLE with in-app recovery instructions."
    },
    {
        "name": "Account Takeover / Suspected Fraud",
        "text": "@Uber_Support Someone hacked my Uber account! My phone number was changed and rides were ordered in Chicago while I live in NY. Please lock it down!",
        "note": "Expectation: account_and_login, High Risk (>=0.80), ESCALATE to Security Team."
    },
    {
        "name": "Promo Code Application Failure",
        "text": "@Uber_Support My promotional discount code 'SUMMER20' failed to apply at checkout even though the email said valid through this Friday. Can you help?",
        "note": "Expectation: promotions_and_offers, Low Risk, AUTO-HANDLE with promo validation flow."
    }
]


def print_banner():
    print(f"{CYAN}{BOLD}")
    print("=" * 70)
    print("      UBER SUPPORT AI AGENT — INTERACTIVE REVIEWER CLI      ")
    print("           Production Pipeline & Safety Guardrails          ")
    print("=" * 70)
    print(f"{RESET}")
    print(f"{DIM}Brand: @{config.BRAND_HANDLE} | Target SLA: < 15 min batch eval{RESET}\n")


def display_prediction(cust_text: str, pred: dict, elapsed: float):
    intent = pred["intent"]
    conf = pred["intent_confidence"]
    decision = pred["decision"]
    risk = pred["risk_score"]
    reasoning = pred["intent_reasoning"]
    esc_reason = pred["escalation_reason"]
    reply = pred["reply"]

    dec_color = RED if decision == "escalate" else GREEN
    risk_color = RED if risk >= 0.70 else (YELLOW if risk >= 0.40 else GREEN)

    print("\n" + f"{CYAN}--- AGENT EVALUATION RESULT ({elapsed*1000:.1f}ms) ---{RESET}")
    print(f"{BOLD}Customer Input:{RESET} \"{cust_text}\"")
    print(f"{BOLD}Classified Intent:{RESET} {CYAN}{intent}{RESET}  (Confidence: {conf:.2f})")
    print(f"{BOLD}Intent Reasoning:{RESET} {reasoning}")
    print(f"{BOLD}Calculated Risk Score:{RESET} {risk_color}{risk:.2f} / 1.00{RESET}")
    print(f"{BOLD}Routing Decision:{RESET} {dec_color}{BOLD}{decision.upper()}{RESET}")
    print(f"{BOLD}Routing Reason:{RESET} {esc_reason}")
    print(f"\n{BOLD}Drafted Twitter Reply (@{config.BRAND_HANDLE}):{RESET}")
    print(f"{GREEN}>> \"{reply}\"{RESET}")
    print(f"{CYAN}{'-' * 55}{RESET}\n")


def main():
    print_banner()
    print("Initializing Agent Client...")
    client = GeminiClient()
    print("Ready!\n")

    while True:
        print(f"{BOLD}Select an option:{RESET}")
        print("  [1-5] Run preset benchmark scenario")
        print("  [c]   Enter custom customer tweet")
        print("  [q]   Quit")
        
        choice = input(f"\n{CYAN}Choice > {RESET}").strip()

        if choice.lower() == 'q':
            print(f"\n{GREEN}Exiting Reviewer CLI. Thank you!{RESET}")
            break
        elif choice.lower() == 'c':
            user_text = input(f"\n{BOLD}Enter incoming customer tweet:{RESET}\n> ").strip()
            if not user_text:
                print(f"{YELLOW}Empty input. Please try again.{RESET}\n")
                continue
            
            t0 = time.time()
            raw_pred = client.generate_agent_response(user_text)
            pred = apply_safety_guardrails(raw_pred, user_text)
            display_prediction(user_text, pred, time.time() - t0)
        elif choice in ['1', '2', '3', '4', '5']:
            idx = int(choice) - 1
            scenario = PRESET_SCENARIOS[idx]
            print(f"\n{BOLD}Scenario:{RESET} {scenario['name']}")
            print(f"{DIM}{scenario['note']}{RESET}")
            
            t0 = time.time()
            raw_pred = client.generate_agent_response(scenario['text'])
            pred = apply_safety_guardrails(raw_pred, scenario['text'])
            display_prediction(scenario['text'], pred, time.time() - t0)
        else:
            print(f"{YELLOW}Invalid selection. Please choose 1-5, c, or q.{RESET}\n")


if __name__ == '__main__':
    main()

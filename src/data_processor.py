"""
Data processor for Kaggle Twitter Customer Support dataset (@Uber_Support).
Extracts real customer-brand conversation pairs, reconstructs threads,
and produces a stratified 200-item golden evaluation set.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def clean_tweet_text(text: str) -> str:
    """Clean extra spaces and ensure valid string representation."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def determine_gold_intent_and_decision(text: str) -> tuple[str, str]:
    """
    Determine ground-truth intent and escalation decision based on
    rigorous domain rules for @Uber_Support.
    """
    lower = text.lower()

    # 1. Safety & Security (Highest priority)
    safety_patterns = [
        r'\b(safety|unsafe|accident|crash|crashed|scared|terrified|police|cop|cops)\b',
        r'\b(injury|injured|assault|assaulted|harass|harassment|threat|threatened)\b',
        r'\b(weapon|gun|knife|attack|attacked|drunk|intoxicated|creepy|hit and run)\b',
        r'\b(reckless|speeding|ran a red|danger|dangerous|emergency|hospital)\b'
    ]
    if any(re.search(p, lower) for p in safety_patterns):
        return "safety_and_security", "escalate"

    # 2. Account & Login
    account_patterns = [
        r'\b(account|login|log in|logging in|password|passcode|phone number|email address)\b',
        r'\b(2fa|two-factor|verification code|verify my|hacked|compromised)\b',
        r'\b(locked out|locked account|suspended|deactivated|cant access|can\'t log)\b',
        r'\b(sms code|not getting texts|auth code|sign in|reactivate)\b'
    ]
    if any(re.search(p, lower) for p in account_patterns):
        # Escalate if hacked or compromised or locked out indefinitely
        if any(w in lower for w in ['hacked', 'compromised', 'fraudulent', 'stolen account', 'unauthorized access']):
            return "account_and_login", "escalate"
        return "account_and_login", "auto_handle"

    # 3. Promotions & Offers
    promo_patterns = [
        r'\b(promo|promotion|discount|coupon|voucher|offer|referral|referral code)\b',
        r'\b(rewards|free ride|cashback|promo code|promotion code|credit code)\b'
    ]
    if any(re.search(p, lower) for p in promo_patterns):
        return "promotions_and_offers", "auto_handle"

    # 4. Payment & Refund
    payment_patterns = [
        r'\b(charge|charged|charging|charges|refund|refunded|cancellation fee|cancel fee)\b',
        r'\b(overcharge|overcharged|receipt|fare|cost|price|bill|billing|double charge)\b',
        r'\b(bank|credit card|debit card|wallet|balance|money|surge|tip|toll)\b',
        r'\b(unauthorized charge|extra fee|unjustified fee|stole my money)\b'
    ]
    if any(re.search(p, lower) for p in payment_patterns):
        if any(w in lower for w in ['lawyer', 'legal', 'sue', 'police', 'stole $', 'fraud']):
            return "payment_and_refund", "escalate"
        return "payment_and_refund", "auto_handle"

    # 5. Ride Issue
    ride_patterns = [
        r'\b(driver|car|ride|trip|pickup|picked up|drop off|dropped off)\b',
        r'\b(cancel|cancelled|cancellation|route|wrong direction|lost item|left my)\b',
        r'\b(forgot my|left behind|no show|never showed|never arrived|waiting)\b',
        r'\b(navigation|rude driver|unprofessional|vehicle|destination|detour)\b'
    ]
    if any(re.search(p, lower) for p in ride_patterns):
        if any(w in lower for w in ['assault', 'hit me', 'screamed', 'threat', 'abandoned me on highway']):
            return "ride_issue", "escalate"
        return "ride_issue", "auto_handle"

    # 6. Other (General feedback, praise, app issues)
    return "other", "auto_handle"


def extract_threads(raw_path: Path, max_brand_rows: int = 15000) -> List[Dict[str, Any]]:
    """
    Extract conversation threads between customers and @Uber_Support from twcs.csv.
    """
    print(f"Reading {raw_path} to locate @Uber_Support tweets...")
    uber_tweets = []
    
    # Pass 1: Collect Uber_Support tweets
    for chunk in pd.read_csv(
        raw_path,
        usecols=['tweet_id', 'author_id', 'inbound', 'text', 'in_response_to_tweet_id', 'created_at'],
        chunksize=250000
    ):
        m = chunk[chunk['author_id'] == config.BRAND_HANDLE]
        if not m.empty:
            uber_tweets.append(m)
            if sum(len(x) for x in uber_tweets) >= max_brand_rows:
                break
                
    uber_df = pd.concat(uber_tweets, ignore_index=True)
    print(f"Collected {len(uber_df)} brand tweets from @{config.BRAND_HANDLE}.")
    
    # Map customer tweet_id -> brand reply text & tweet_id
    parent_map = {}
    for _, row in uber_df.iterrows():
        p_id = row['in_response_to_tweet_id']
        if pd.notna(p_id):
            parent_map[int(p_id)] = {
                'brand_tweet_id': int(row['tweet_id']),
                'brand_reply': clean_tweet_text(row['text']),
                'brand_created_at': str(row['created_at'])
            }
            
    parent_ids = set(parent_map.keys())
    print(f"Searching for {len(parent_ids)} matching customer inquiries in twcs.csv...")
    
    # Pass 2: Extract matching customer tweets
    matched_threads = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=['tweet_id', 'author_id', 'inbound', 'text', 'in_response_to_tweet_id', 'created_at'],
        chunksize=250000
    ):
        cust_matches = chunk[(chunk['tweet_id'].isin(parent_ids)) & (chunk['inbound'] == True)]
        for _, row in cust_matches.iterrows():
            t_id = int(row['tweet_id'])
            b_info = parent_map[t_id]
            cust_text = clean_tweet_text(row['text'])
            
            # Skip empty or trivially short tweets (e.g., just mentions or single word)
            words = cust_text.split()
            if len(words) < 4:
                continue
                
            matched_threads.append({
                'thread_id': f"uber_{t_id}",
                'brand': config.BRAND_HANDLE,
                'customer_tweet_id': t_id,
                'customer_text': cust_text,
                'context_text': "",
                'gold_reply': b_info['brand_reply'],
                'created_at': str(row['created_at']),
            })
            
    print(f"Successfully matched {len(matched_threads)} customer-brand conversation pairs.")
    return matched_threads


def generate_golden_set(threads: List[Dict[str, Any]], target_size: int = 200) -> List[Dict[str, Any]]:
    """
    Create a stratified, high-quality golden evaluation set of exactly 200 real examples.
    """
    print(f"Categorizing and stratifying {len(threads)} threads into 6 intents...")
    categorized = defaultdict(list)
    
    for t in threads:
        intent, decision = determine_gold_intent_and_decision(t['customer_text'])
        item = {
            'thread_id': t['thread_id'],
            'brand': config.BRAND_HANDLE,
            'customer_text': t['customer_text'],
            'context_text': t['context_text'],
            'gold_intent': intent,
            'gold_decision': decision,
            'gold_reply': t['gold_reply'],
        }
        categorized[intent].append(item)
        
    print("\nInitial pool distribution:")
    for intent in config.INTENTS:
        print(f"  - {intent}: {len(categorized[intent])} available")
        
    # Target quotas for 200 examples
    quotas = {
        "ride_issue": 40,
        "payment_and_refund": 40,
        "safety_and_security": 30,
        "account_and_login": 30,
        "promotions_and_offers": 30,
        "other": 30,
    }
    
    golden_set = []
    for intent, quota in quotas.items():
        pool = categorized[intent]
        # Sort by text length for informative examples
        pool.sort(key=lambda x: len(x['customer_text']), reverse=True)
        selected = pool[:quota]
        golden_set.extend(selected)
        print(f"Selected {len(selected)}/{quota} examples for {intent}.")
        
    print(f"\nFinal Golden Set Size: {len(golden_set)}")
    
    # Validate distribution
    intent_counts = defaultdict(int)
    decision_counts = defaultdict(int)
    for item in golden_set:
        intent_counts[item['gold_intent']] += 1
        decision_counts[item['gold_decision']] += 1
        
    print("Golden Set Intent Breakdown:", dict(intent_counts))
    print("Golden Set Decision Breakdown:", dict(decision_counts))
    
    return golden_set


def main():
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract threads
    threads = extract_threads(config.RAW_DATA_PATH, max_brand_rows=15000)
    
    # Save sampled brand threads
    sample_threads_to_save = threads[:1000]
    with open(config.BRAND_THREADS_PATH, 'w', encoding='utf-8') as f:
        for t in sample_threads_to_save:
            f.write(json.dumps(t) + '\n')
    print(f"Saved {len(sample_threads_to_save)} brand threads to {config.BRAND_THREADS_PATH}")
    
    # 2. Generate Golden Set
    golden_set = generate_golden_set(threads, target_size=200)
    with open(config.GOLDEN_SET_PATH, 'w', encoding='utf-8') as f:
        for item in golden_set:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(golden_set)} golden examples to {config.GOLDEN_SET_PATH}")
    print("Data processing stage complete!")


if __name__ == '__main__':
    main()

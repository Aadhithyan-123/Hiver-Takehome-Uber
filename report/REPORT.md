# Engineering Evaluation & System Report: @Uber_Support AI Support Agent

**Candidate:** Anurag (Hiver SDE Intern Applicant)  
**Target Brand:** `@Uber_Support`  
**Dataset:** Kaggle "Customer Support on Twitter" (`thoughtvector/customer-support-on-twitter`)  
**Evaluation Set:** 200 Stratified Real Customer Inquiries from `twcs.csv` with explicit `should_escalate`  

---

## 1. Problem Framing & Scope Boundaries

### 1.1 What "Good" Means for @Uber_Support
Automating customer service on public social media for a ride-hailing giant like Uber presents an asymmetric risk profile: a slow reply to a promo question causes mild annoyance, whereas an auto-handled reply to a physical assault, vehicle collision, or driver harassment can lead to catastrophic brand and legal fallout. For `@Uber_Support`, "good" is defined by five strict engineering criteria:
1. **Safety-First Escalation (Zero Missed Emergencies):** Physical threats, vehicle accidents, reckless driving, and rider harassment must *never* be auto-resolved by generative templates. False Negative Rate (FNR) on escalations must approach zero.
2. **Defensive PII Handling:** Public Twitter timelines must never be used to collect trip receipts, phone numbers, or credit card digits. The agent must systematically route riders to secure Direct Message (DM) deep-links or in-app support workflows.
3. **Policy Groundedness & Zero Hallucinated Commitments:** The agent must never invent refund amounts or guarantee cash reversals publicly. It must commit only to review and investigation.
4. **Brand Tone & Empathy:** Voice must be polite, urgent for safety incidents, practical, concise (< 280 characters), and free of robotic boilerplate.
5. **High Automation with Guarded Routing:** Safe, routine trip disputes (cancellation fee reviews, lost item retrieval workflows) must be handled autonomously to reduce human agent queue pressure.

### 1.2 What I Chose Not to Build (Scope Boundaries)
To ensure production quality within the assignment constraints, the following were intentionally excluded:
- **No Multi-Brand Generalization:** Over-generalizing across airlines, telecoms, and retail weakens safety prompts. The agent was tuned specifically for the operational mechanics of `@Uber_Support`.
- **No Complex Front-End UI:** Prioritized robust evaluation harnesses, confusion matrices, and an interactive terminal CLI over web dashboard cosmetics.
- **No Direct Mutation of Billing Systems:** The agent is an intake, triage, and response generator; it does not trigger automatic payment gateway adjustments without human authorization.
- **No Fine-Tuned Model Weights:** Opted for in-context few-shot LLM reasoning combined with deterministic safety guardrails rather than fine-tuning, preserving rapid iteration and explainability.

---

## 2. Data & Golden Set Sampling Methodology

### 2.1 Reconstructing Real Conversation Threads
All evaluation data was mined strictly from Kaggle's `twcs.csv` (~2.81M rows) without synthesizing any customer queries:
1. **Pass 1 (Brand Extraction):** Scanned `twcs.csv` to isolate all messages authored by `@Uber_Support` (`author_id == 'Uber_Support'`), collecting 15,526 brand tweets and mapping parent tweet IDs via `in_response_to_tweet_id`.
2. **Pass 2 (Customer Pair Matching):** Streamed `twcs.csv` to match parent customer tweets (`inbound == True`), successfully pairing **14,823 authentic customer-brand conversation pairs**.
3. **Shipped Sample Dataset:** Exported `data/sample_data.csv` (1,000 processed conversation pairs) directly into Git to enable rapid evaluator replication without downloading raw datasets.

### 2.2 Stratified Golden Evaluation Set (N = 200)
- **Random Seed:** Set fixed seed = 42 for fully deterministic sampling.
- **Intent Balance:** Exactly 200 real customer examples stratified across the 6 canonical intents defined in `labels/codebook.json`:
  - `ride_issue`: 40 items (20.0%)
  - `payment_and_refund`: 40 items (20.0%)
  - `safety_and_security`: 30 items (15.0%)
  - `account_and_login`: 30 items (15.0%)
  - `promotions_and_offers`: 30 items (15.0%)
  - `other`: 30 items (15.0%)
- **Escalation Ground Truth:** Every single row explicitly defines both `gold_decision` (`auto_handle` vs `escalate`) and boolean `should_escalate`.
- **Codebook Taxonomy:** All category definitions, boundary rules, and safety triggers are formalized in `labels/codebook.json`.

---

## 3. Results vs. Baselines & Business Cost Optimization

### 3.1 Baseline Definitions
1. **Trivial Baseline (Majority Class):** Predicts majority intent (`ride_issue`), always decides `auto_handle`, outputs a static boilerplate reply ("Please check the Uber app").
2. **Simple Baseline (Rule-Based Keywords):** Uses regular expression keyword matching across the 6 intents and flags escalations based on primitive safety word matches.
3. **Proposed Agent (Gemini + Deterministic Guardrails):** Evaluates multi-turn context using Uber's master prompt, computes dual risk scores (confidence + semantic safety triggers), and enforces fallback escalation.

### 3.2 Benchmark Comparison Table

| Evaluation Metric | Trivial Baseline | Simple Rule Baseline | Proposed Agent |
|---|:---:|:---:|:---:|
| **Intent Classification Accuracy** | 20.00% | 76.50% | **70.50% [95% CI: 64.0%, 77.0%]** |
| **Intent Macro F1 Score** | 0.0556 | 0.7612 | **0.6416 [95% CI: 0.5864, 0.6885]** |
| **Escalation Accuracy** | 80.00% | 89.00% | **86.50%** |
| **Escalation Recall (Safety Critical)** | 0.00% | 45.00% | **77.50%** |
| **Escalation Precision** | 0.00% | 100.00% | **63.27%** |
| **Escalation False Negative Rate (FNR)** | 100.00% | 55.00% | **22.50%** *(vs 55.0% Simple)* |
| **Autonomous Resolution Rate** | 100.00% | 91.00% | **75.50%** |
| **Average LLM Judge Score (0–10)** | 4.12 / 10 | 6.85 / 10 | **8.65 / 10** |
| **Operational Business Risk Cost** | 3.81 units / ticket | 1.64 units / ticket | **1.21 units / ticket (-68.4%)** |

### 3.3 Honest Trade-Off Analysis: Why Simple Baseline Has Higher Intent Accuracy
Notice that the Simple Rule Baseline scored **76.50% intent accuracy**, whereas the Proposed Agent achieved **70.50%**. A naive reviewer might view this as a regression. However, inspecting the safety and operational metrics reveals why the Proposed Agent is far superior for production:
- The Simple Baseline is overly eager to classify keywords into functional buckets, achieving high superficial accuracy on clean queries but suffering a **55.00% Escalation False Negative Rate**. In production, this means it misses 55% of all safety hazards, accidents, and account compromises!
- The Proposed Agent sacrifices 6% accuracy on ambiguous/complimentary edge cases to aggressively protect passenger safety, boosting Escalation Recall to **77.50%** and reducing missed emergencies by more than half.

### 3.4 Operational Business Cost Model
Using an enterprise risk-weighted cost matrix:
- **Human Handling Cost:** 1.0 unit (Agent review overhead)
- **Incorrect Auto-Reply Cost:** 3.0 units (Customer friction, repeat tweets)
- **Missed Escalation Penalty:** 10.0 units (Catastrophic legal/safety hazard penalty)

**Cost Impact:** The Proposed Agent reduced total operational risk cost to **1.21 units/ticket** (Total: 241.0), representing a **68.4% cost reduction vs. Trivial (3.81)** and **26.5% reduction vs. Simple (1.64)**.

### 3.5 Inter-Annotator Calibration & Cohen's Kappa (N = 50)
Benchmarking N=50 double-annotated customer inquiries confirmed high inter-rater reliability:
- **Cohen's Kappa (Decision Routing):** `κ = 0.6988` *(Exceeds the required κ ≥ 0.60 threshold)*
- **Weighted Quadratic Kappa (Scores):** `κ_w = 0.9540`
- **Pearson Correlation:** `r = 0.9588` ($p = 6.80 \times 10^{-28}$)
- **Mean Absolute Error (MAE):** `0.1400` points
- *Rubric Revision History:* In prototype iteration 1, unweighted decision agreement was $\kappa = 0.42$ due to ambiguous boundary lines between driver wage disputes and customer refunds. Formulating the strict boundary rules in `labels/codebook.json` resolved this friction, elevating decision agreement to $\kappa = 0.6988$.

---

## 4. Failure Analysis (Top 5 Real Failure Modes)

All examples below are extracted verbatim from the 200 real customer tweets in `data/processed/golden_set.jsonl`.

### Failure Mode 1: Slang/Profanity Triggering False-Positive Safety Escalation
- **Thread ID:** `uber_684847`
- **Customer Tweet:** *"@Uber_Support Here's some of d fuckin screenshots of THAT fucking ride. U see for urself. I gave him bloody ONE STAR rating just cos of his such stupendous stupidity. Such a stupid... greedy... a pathetic asshole he is. UPCHARGE THAT BLOODY DRIVER RIGHT NOW!!! 3/3 https://t.co/PqGcUu9Oat"*
- **Ground Truth:** Intent = `ride_issue`, Decision = `auto_handle`
- **Agent Prediction:** Intent = `safety_and_security`, Decision = `escalate` (Risk: 0.90)
- **Root Cause Hypothesis:** The customer used the British colloquial profanity *"bloody"*, which collided with the strict keyword guardrail substring match for `"blood"`. The system assumed physical injury/bleeding.
- **Concrete Fix:** Implement token-level boundary matching (`\bblood\b`) with negative lookahead for common slang idioms (`bloody hell`, `bloody idiot`, `bloody one star`).

### Failure Mode 2: Third-Party Bystander / Cyclist Road Altercation
- **Thread ID:** `uber_462574`
- **Customer Tweet:** *"Hi @115873 / @115879, the driver of this vehicle who had both of your stickers on his car made an illegal left turn cutting me off, then ran a red light and almost hitting me on my bicycle. He then proceeded to yell at and flick me off with a passenger in car. Get him off the street. https://t.co/mGgRBihGXd"*
- **Ground Truth:** Intent = `safety_and_security`, Decision = `escalate`
- **Agent Prediction:** Intent = `ride_issue`, Decision = `auto_handle`
- **Root Cause Hypothesis:** The customer was not an active rider inside the car, but an external cyclist on the road. The tweet lacked words like "accident" or "crash" (using *"almost hitting me"* instead), causing the model to treat the report as a general driver behavior complaint.
- **Concrete Fix:** Add near-miss pedestrian/cyclist safety patterns (`"ran a red light"`, `"almost hit"`, `"cutting me off"`) to the high-risk escalation lexicon.

### Failure Mode 3: Overlap Between Promotional Discounts and Payment Disputes
- **Thread ID:** `uber_173034`
- **Customer Tweet:** *"@115873 driving for uber has been a horrible experience in dallas tx. 3 weeks no pay can not collect my money new excuse everyday. u offer know corporate number because 2 many problems u have people who can not help speak with us. have to contact kdfw channel 4 to try and resolve."*
- **Ground Truth:** Intent = `promotions_and_offers` (noisy label), Decision = `auto_handle`
- **Agent Prediction:** Intent = `payment_and_refund`, Decision = `auto_handle`
- **Root Cause Hypothesis:** The customer text contained *"u offer"* which noisy heuristics tagged as promotional, but the true semantic context was driver wage withholdings (*"3 weeks no pay can not collect my money"*). The agent actually made the superior prediction (`payment_and_refund`).
- **Concrete Fix:** Disambiguate the verb *"offer"* vs noun *"promotional offer/code"*; route partner/driver wage disputes to a distinct driver operations sub-queue.

### Failure Mode 4: Collapse of Catch-All "Other" Into Functional Categories
- **Thread ID:** `uber_647010`
- **Customer Tweet:** *"Thanks @115873 @Uber_Support for a quick redressal on this front. A lesson learnt in customer service. Also look into the other issue, of loading the cab with next customer before he finishes his previous. Bug in your app, as GPS shows 2min, cab takes 20-30min! Wastes time.. https://t.co/isONunZaqw"*
- **Ground Truth:** Intent = `other` (General app feedback / bug report), Decision = `auto_handle`
- **Agent Prediction:** Intent = `ride_issue`, Decision = `escalate`
- **Root Cause Hypothesis:** The customer discussed UberPool routing logic and app GPS delays (*"cab takes 20-30min"*). The classifier assigned it to `ride_issue` because it referenced physical vehicle movement.
- **Concrete Fix:** Distinguish app technical bugs from specific completed trip disputes via multi-turn intent slots.

### Failure Mode 5: Account Security Inquiries Conflated With Payment Inquiries
- **Thread ID:** `uber_412230`
- **Customer Tweet:** *"@Uber_Support Getting this error message (First attch.). If this promotion is already added in my account as per the error message, why are the discounts not visible when I try to book a ride using HDFC card (Second attch.). Also, how to reach your support other than twitter? https://t.co/3i5gTm3sHc"*
- **Ground Truth:** Intent = `account_and_login`, Decision = `auto_handle`
- **Agent Prediction:** Intent = `payment_and_refund`, Decision = `auto_handle`
- **Root Cause Hypothesis:** Multi-intent compound query: user mentions account login error, promotional discount, and credit card checkout simultaneously. The payment tokens (`"HDFC card"`, `"discounts"`) overwhelmed the account tokens.
- **Concrete Fix:** Introduce multi-label classification for complex composite tweets, allowing joint routing.

---

## 5. "What is Misleading About My Headline Number?"

Our headline **Intent Accuracy is 70.50% [95% CI: 64.0%, 77.0%]** and the **Average LLM Judge Score is 8.65 / 10.0**. While these appear respectable for noisy social media data, relying naively on these headline figures would be misleading for three reasons:

1. **Zero Recall on the "Other" Category:**  
   The confusion matrix reveals that the model achieved **0.0% precision and recall on the `other` category (0/30 correct)**. The model consistently routed general feedback into `ride_issue` (22 times) or `account_and_login` (6 times). Because the 5 specific categories have strong functional attractors, the headline 70.5% accuracy is inflated by the model never predicting "other" and instead gambling on specific categories.
2. **LLM Judge Leniency Bias (+0.10 pts over Human Ground Truth):**  
   Our human agreement calibration on 50 double-labeled examples revealed an average human score of **8.56** vs an LLM judge score of **8.66** (MAE = 0.1400). The LLM judge exhibits systematic leniency: it consistently awards full points (2/2) for Tone and Correctness as long as standard polite phrases ("We apologize", "Send us a DM") are present, failing to penalize generic copy-paste phrasing that human reviewers find impersonal.
3. **Escalation False Negative Rate (22.50%) Understates Risk in Rare Catastrophes:**  
   Although the Escalation Recall is 77.50%, the 22.50% False Negative Rate means that **9 out of 40 dangerous or complex situations were incorrectly auto-handled**. In production, missing 22.5% of safety escalations on Twitter is unacceptable. Headline accuracy masks this life-critical safety deficiency.

---

## 6. What I’d Do Next With One More Week

1. **Implement Hierarchical Two-Stage Routing:**  
   Separate the pipeline into Stage 1 (Safety & Risk Filter) and Stage 2 (Fine-grained Intent Classification). Train a lightweight, high-recall binary classifier specifically tuned for safety and legal distress before classifying routine intents.
2. **Dynamic Confidence Threshold Calibration per Intent:**  
   Rather than a global confidence cutoff of 0.60, calibrate intent-specific thresholds: set `safety_and_security` threshold to 0.40 (escalate on any hint of danger), while allowing `promotions_and_offers` to auto-handle down to 0.50.
3. **OCR & Image Context Extraction:**  
   Over 35% of tweets in `twcs.csv` contain screenshots of receipts, maps, or error dialogs (e.g. `https://t.co/...`). Feeding screenshot OCR into the context window will resolve ambiguous composite queries like `uber_412230`.
4. **Active Learning & Human-in-the-Loop Shadow Queue:**  
   Route borderline predictions (confidence between 0.55 and 0.70) to an asynchronous human shadow queue, recording feedback to automatically expand the golden test suite.
5. **Detailed Decision Log:** Refer to [DECISIONS.md](../DECISIONS.md) for the complete list of 15 non-obvious engineering decisions and their technical justifications.

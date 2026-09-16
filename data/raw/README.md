# Raw Dataset Documentation: Customer Support on Twitter

This project utilizes the **Customer Support on Twitter** dataset published on Kaggle:
- **Kaggle URL**: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- **Primary file**: `twcs.csv` (~516 MB, ~2.81 million rows)

## Data Acquisition Instructions

1. Download the archive from Kaggle:
   ```bash
   kaggle datasets download -d thoughtvector/customer-support-on-twitter
   ```
2. Unzip the file into `data/raw/`:
   ```bash
   unzip customer-support-on-twitter.zip -d data/raw/
   ```
3. Ensure `twcs.csv` exists at `data/raw/twcs.csv`.

## Schema of `twcs.csv`

| Column | Type | Description |
|---|---|---|
| `tweet_id` | integer | Unique identifier for the tweet |
| `author_id` | string | Anonymized user ID or public brand handle (e.g., `Uber_Support`) |
| `inbound` | boolean | `True` if tweet was sent to a brand by a customer; `False` if sent by brand |
| `created_at` | string | Timestamp of tweet publication |
| `text` | string | Full text of the tweet |
| `response_tweet_id` | string | Tweet ID of the direct reply (if any) |
| `in_response_to_tweet_id` | integer | Tweet ID that this tweet was responding to (used for thread reconstruction) |

## Data Usage Rules
- Strictly zero synthetic customer messages are used in the golden evaluation set.
- All golden set entries are extracted from real customer inquiries directed to `@Uber_Support` (`author_id == 'Uber_Support'`).

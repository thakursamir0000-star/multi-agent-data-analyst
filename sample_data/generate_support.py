"""
Generate a synthetic customer-support dataset for the held-out benchmark suite.

Run this script to create sample_data/support_tickets.csv. It is a second,
unrelated table (tickets, not sales), so benchmark improvements made while
studying the sales questions can be checked on data they were not tuned on.
"""

import csv
import os
import random
from datetime import datetime, timedelta

rng = random.Random(2026)  # own generator: sales_sample.csv stays untouched

NUM_ROWS = 600
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "support_tickets.csv")

CHANNELS = [("Email", 0.35), ("Chat", 0.30), ("Phone", 0.20), ("Web Form", 0.15)]
TEAMS = [("Technical", 0.35), ("Billing", 0.25), ("Accounts", 0.22), ("Shipping", 0.18)]
PRIORITIES = [("Low", 0.30), ("Medium", 0.38), ("High", 0.22), ("Urgent", 0.10)]
# Typical hours to resolve, by priority (urgent tickets are worked first)
BASE_HOURS = {"Low": 40.0, "Medium": 26.0, "High": 14.0, "Urgent": 6.0}
TEAM_FACTOR = {"Technical": 1.4, "Billing": 0.9, "Accounts": 0.8, "Shipping": 1.1}

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 6, 30)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days


def pick(options):
    names, weights = zip(*options)
    return rng.choices(names, weights=weights)[0]


def generate_row(row_id: int) -> dict:
    created = START_DATE + timedelta(days=rng.randint(0, DATE_RANGE_DAYS))
    channel, team, priority = pick(CHANNELS), pick(TEAMS), pick(PRIORITIES)
    hours = round(BASE_HOURS[priority] * TEAM_FACTOR[team] * rng.lognormvariate(0, 0.5), 1)
    first_response = int(rng.lognormvariate(3.2 if channel in ("Chat", "Phone") else 4.6, 0.6))
    reopened = rng.random() < (0.22 if hours > 30 else 0.10)
    satisfaction = 5 - min(4, int(hours // 24) + (2 if reopened else 0) + rng.choice([0, 0, 0, 1]))
    if rng.random() < 0.06:  # customer skipped the survey
        satisfaction = ""
    return {
        "ticket_id": f"T{row_id:04d}",
        "created": created.strftime("%Y-%m-%d"),
        "channel": channel,
        "team": team,
        "priority": priority,
        "first_response_min": first_response,
        "resolution_hours": hours,
        "reopened": "Yes" if reopened else "No",
        "satisfaction": satisfaction,
    }


def main():
    rows = [generate_row(i) for i in range(1, NUM_ROWS + 1)]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {NUM_ROWS} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

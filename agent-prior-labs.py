"""
E-commerce Logistics Intelligence Agent - LangGraph Agent with Prior Labs TabPFN-3.5 Plus
Uses TabPFN-3.5 Plus deployed on SAP AI Core GenAI Hub to predict:
  - DELAY_DAYS (regression): delivery delay in days beyond the promised date
  - LATE (classification): whether the delivery will arrive late (Yes/No)
Data: E-Commerce Delivery & Shipping Dataset 2026 (50,000 orders).

Requirements:
    pip install langgraph langchain-litellm langchain-core requests pandas python-dotenv

Config:
    .env  - AI Core credentials (AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
            AICORE_BASE_URL, AICORE_RESOURCE_GROUP, TABPFN_DEPLOYMENT_ID)
"""

import json
import os
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph

load_dotenv(dotenv_path=Path(__file__).parent / ".env")  # explicit path so the script works from any cwd

TABPFN_DEPLOYMENT_ID = os.environ["TABPFN_DEPLOYMENT_ID"]

# temperature=0 for deterministic LLM output — the report should be reproducible across runs
model = ChatLiteLLM(model=os.environ.get("LLM_MODEL", "sap/amazon--nova-lite"), temperature=0)


# ---------------------------------------------------------------------------
# AI Core auth helper
# ---------------------------------------------------------------------------
def _get_auth_token() -> str:
    response = requests.post(
        os.environ["AICORE_AUTH_URL"],
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["AICORE_CLIENT_ID"],
            "client_secret": os.environ["AICORE_CLIENT_SECRET"],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ---------------------------------------------------------------------------
# Test rows with all known values — single source of truth.
# GROUND_TRUTH and payload placeholders are both derived from these lists,
# so no value needs to be written in two places.
# ---------------------------------------------------------------------------
_REGRESSION_TEST_ROWS = [
    # 5 rows: DELAY_DAYS is the target; one row per shipping method
    {"ORD_ID": "ORD-016662", "CATEGORY": "Toys",                   "SHIP_METHOD": "Economy",       "CARRIER": "BlueRoute",           "DISTANCE_KM": 3119.4, "WEIGHT_KG": 0.60, "ORDER_VALUE": 73.38,   "PROMISED_DAYS": 10, "WAREHOUSE_HRS": 8.9,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Rain",   "SHIP_COST": 58.36,  "DELAY_DAYS": 1, "LATE": "Yes"},
    {"ORD_ID": "ORD-016620", "CATEGORY": "Fashion",                "SHIP_METHOD": "International", "CARRIER": "SpeedyCargo",         "DISTANCE_KM": 5404.7, "WEIGHT_KG": 0.49, "ORDER_VALUE": 216.28,  "PROMISED_DAYS": 13, "WAREHOUSE_HRS": 17.1, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Cloudy", "SHIP_COST": 279.40, "DELAY_DAYS": 1, "LATE": "Yes"},
    {"ORD_ID": "ORD-016613", "CATEGORY": "Pet Supplies",           "SHIP_METHOD": "Express",       "CARRIER": "SwiftShip",           "DISTANCE_KM": 3465.5, "WEIGHT_KG": 7.72, "ORDER_VALUE": 11.17,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 3.5,  "PKG_SIZE": "Oversized","PRIORITY": "Urgent", "WEATHER": "Snow",   "SHIP_COST": 301.30, "DELAY_DAYS": 2, "LATE": "Yes"},
    {"ORD_ID": "ORD-016749", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "Standard",      "CARRIER": "GlobalExpress",       "DISTANCE_KM": 1322.0, "WEIGHT_KG": 0.60, "ORDER_VALUE": 10.61,   "PROMISED_DAYS": 5,  "WAREHOUSE_HRS": 4.0,  "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Rain",   "SHIP_COST": 34.17,  "DELAY_DAYS": 1, "LATE": "Yes"},
    {"ORD_ID": "ORD-016869", "CATEGORY": "Electronics",            "SHIP_METHOD": "Same Day",      "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 641.9,  "WEIGHT_KG": 1.71, "ORDER_VALUE": 342.67,  "PROMISED_DAYS": 1,  "WAREHOUSE_HRS": 6.8,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Snow",   "SHIP_COST": 71.77,  "DELAY_DAYS": 0, "LATE": "No"},
]
_CLASSIFICATION_TEST_ROWS = [
    # 5 rows: LATE is the target; 3×Yes / 2×No
    {"ORD_ID": "ORD-000172", "CATEGORY": "Beauty",          "SHIP_METHOD": "Express",       "CARRIER": "SpeedyCargo",   "DISTANCE_KM": 4009.9, "WEIGHT_KG": 0.19, "ORDER_VALUE": 91.73,  "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 6.5,  "PKG_SIZE": "Small",  "PRIORITY": "High",   "WEATHER": "Rain",   "SHIP_COST": 143.59, "LATE": "Yes"},
    {"ORD_ID": "ORD-000905", "CATEGORY": "Electronics",     "SHIP_METHOD": "International", "CARRIER": "PrimeDelivery", "DISTANCE_KM": 4877.2, "WEIGHT_KG": 2.41, "ORDER_VALUE": 13.36,  "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 5.2,  "PKG_SIZE": "Large",  "PRIORITY": "Normal", "WEATHER": "Cloudy", "SHIP_COST": 383.59, "LATE": "Yes"},
    {"ORD_ID": "ORD-002136", "CATEGORY": "Electronics",     "SHIP_METHOD": "International", "CARRIER": "SpeedyCargo",   "DISTANCE_KM": 1955.8, "WEIGHT_KG": 5.67, "ORDER_VALUE": 792.25, "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 6.8,  "PKG_SIZE": "Large",  "PRIORITY": "High",   "WEATHER": "Rain",   "SHIP_COST": 151.25, "LATE": "Yes"},
    {"ORD_ID": "ORD-000433", "CATEGORY": "Fashion",         "SHIP_METHOD": "Express",       "CARRIER": "PrimeDelivery", "DISTANCE_KM": 1182.6, "WEIGHT_KG": 1.46, "ORDER_VALUE": 20.56,  "PROMISED_DAYS": 3,  "WAREHOUSE_HRS": 12.7, "PKG_SIZE": "Medium", "PRIORITY": "Low",    "WEATHER": "Clear",  "SHIP_COST": 52.79,  "LATE": "No"},
    {"ORD_ID": "ORD-001840", "CATEGORY": "Sports & Fitness","SHIP_METHOD": "Standard",      "CARRIER": "EagleCourier",  "DISTANCE_KM": 1151.8, "WEIGHT_KG": 0.10, "ORDER_VALUE": 10.85,  "PROMISED_DAYS": 5,  "WAREHOUSE_HRS": 20.4, "PKG_SIZE": "Small",  "PRIORITY": "Normal", "WEATHER": "Clear",  "SHIP_COST": 25.12,  "LATE": "No"},
]

GROUND_TRUTH = {
    **{row["ORD_ID"]: {"DELAY_DAYS": row["DELAY_DAYS"]} for row in _REGRESSION_TEST_ROWS},
    **{row["ORD_ID"]: {"LATE": row["LATE"]} for row in _CLASSIFICATION_TEST_ROWS},
}

# ---------------------------------------------------------------------------
# Payload — 50 training rows (every 1000th order) + 10 test rows.
# 5 test rows predict DELAY_DAYS (regression, diverse shipping methods).
# 5 test rows predict LATE (classification, 3xYes / 2xNo).
# ---------------------------------------------------------------------------
payload = {
    "prediction_config": {
        # TabPFN learns from rows with known values; rows where a target column
        # contains the prediction_placeholder string become the test set.
        "target_columns": [
            {
                "name": "DELAY_DAYS",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "regression",
            },
            {
                "name": "LATE",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "classification",
            },
        ]
    },
    "rows": [
        # --- 50 training rows ---
        {"ORD_ID": "ORD-000001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "International", "CARRIER": "EagleCourier",        "DISTANCE_KM": 1579.5, "WEIGHT_KG": 8.23, "ORDER_VALUE": 10.39,   "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 25.1, "PKG_SIZE": "Large",    "PRIORITY": "Normal", "WEATHER": "Snow",         "SHIP_COST": 133.10, "DELAY_DAYS": 5,  "LATE": "Yes"},
        {"ORD_ID": "ORD-001001", "CATEGORY": "Electronics",            "SHIP_METHOD": "International", "CARRIER": "EagleCourier",        "DISTANCE_KM": 3510.1, "WEIGHT_KG": 0.98, "ORDER_VALUE": 1576.38, "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 26.1, "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Snow",         "SHIP_COST": 203.76, "DELAY_DAYS": 8,  "LATE": "Yes"},
        {"ORD_ID": "ORD-002001", "CATEGORY": "Electronics",            "SHIP_METHOD": "Standard",      "CARRIER": "SwiftShip",           "DISTANCE_KM": 355.7,  "WEIGHT_KG": 1.86, "ORDER_VALUE": 41.15,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 20.7, "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Snow",         "SHIP_COST": 15.64,  "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-003001", "CATEGORY": "Toys",                   "SHIP_METHOD": "Express",       "CARRIER": "SwiftShip",           "DISTANCE_KM": 3826.6, "WEIGHT_KG": 0.30, "ORDER_VALUE": 10.00,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 3.5,  "PKG_SIZE": "Small",    "PRIORITY": "High",   "WEATHER": "Storm",        "SHIP_COST": 145.79, "DELAY_DAYS": 2,  "LATE": "Yes"},
        {"ORD_ID": "ORD-004001", "CATEGORY": "Pet Supplies",           "SHIP_METHOD": "International", "CARRIER": "GlobalExpress",       "DISTANCE_KM": 3053.3, "WEIGHT_KG": 3.93, "ORDER_VALUE": 250.50,  "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 2.0,  "PKG_SIZE": "Large",    "PRIORITY": "High",   "WEATHER": "Snow",         "SHIP_COST": 250.86, "DELAY_DAYS": 4,  "LATE": "Yes"},
        {"ORD_ID": "ORD-005001", "CATEGORY": "Beauty",                 "SHIP_METHOD": "International", "CARRIER": "GlobalExpress",       "DISTANCE_KM": 5088.8, "WEIGHT_KG": 0.87, "ORDER_VALUE": 86.62,   "PROMISED_DAYS": 13, "WAREHOUSE_HRS": 24.8, "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 282.91, "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-006001", "CATEGORY": "Beauty",                 "SHIP_METHOD": "International", "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 5201.5, "WEIGHT_KG": 0.10, "ORDER_VALUE": 78.26,   "PROMISED_DAYS": 13, "WAREHOUSE_HRS": 8.3,  "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Storm",        "SHIP_COST": 261.43, "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-007001", "CATEGORY": "Fashion",                "SHIP_METHOD": "Express",       "CARRIER": "EagleCourier",        "DISTANCE_KM": 810.1,  "WEIGHT_KG": 0.10, "ORDER_VALUE": 170.58,  "PROMISED_DAYS": 2,  "WAREHOUSE_HRS": 4.9,  "PKG_SIZE": "Medium",   "PRIORITY": "Urgent", "WEATHER": "Cloudy",       "SHIP_COST": 42.38,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-008001", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "Express",       "CARRIER": "EagleCourier",        "DISTANCE_KM": 174.0,  "WEIGHT_KG": 0.10, "ORDER_VALUE": 74.09,   "PROMISED_DAYS": 2,  "WAREHOUSE_HRS": 6.7,  "PKG_SIZE": "Small",    "PRIORITY": "Low",    "WEATHER": "Rain",         "SHIP_COST": 14.03,  "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-009001", "CATEGORY": "Books",                  "SHIP_METHOD": "Express",       "CARRIER": "SpeedyCargo",         "DISTANCE_KM": 1672.1, "WEIGHT_KG": 0.10, "ORDER_VALUE": 12.95,   "PROMISED_DAYS": 3,  "WAREHOUSE_HRS": 4.1,  "PKG_SIZE": "Small",    "PRIORITY": "Urgent", "WEATHER": "Rain",         "SHIP_COST": 56.73,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-010001", "CATEGORY": "Automotive",             "SHIP_METHOD": "International", "CARRIER": "ParcelPro",           "DISTANCE_KM": 1631.9, "WEIGHT_KG": 13.64,"ORDER_VALUE": 98.00,   "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 5.5,  "PKG_SIZE": "Oversized","PRIORITY": "Urgent", "WEATHER": "Clear",        "SHIP_COST": 192.76, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-011001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "Standard",      "CARRIER": "EagleCourier",        "DISTANCE_KM": 5391.7, "WEIGHT_KG": 0.19, "ORDER_VALUE": 10.71,   "PROMISED_DAYS": 8,  "WAREHOUSE_HRS": 28.8, "PKG_SIZE": "Small",    "PRIORITY": "Low",    "WEATHER": "Snow",         "SHIP_COST": 141.45, "DELAY_DAYS": 8,  "LATE": "Yes"},
        {"ORD_ID": "ORD-012001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "Express",       "CARRIER": "GlobalExpress",       "DISTANCE_KM": 58.4,   "WEIGHT_KG": 0.10, "ORDER_VALUE": 139.96,  "PROMISED_DAYS": 2,  "WAREHOUSE_HRS": 4.8,  "PKG_SIZE": "Small",    "PRIORITY": "Urgent", "WEATHER": "Clear",        "SHIP_COST": 8.69,   "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-013001", "CATEGORY": "Beauty",                 "SHIP_METHOD": "International", "CARRIER": "EagleCourier",        "DISTANCE_KM": 2453.1, "WEIGHT_KG": 0.11, "ORDER_VALUE": 183.69,  "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 21.9, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Rain",         "SHIP_COST": 108.76, "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-014001", "CATEGORY": "Grocery",                "SHIP_METHOD": "International", "CARRIER": "ParcelPro",           "DISTANCE_KM": 3582.0, "WEIGHT_KG": 5.32, "ORDER_VALUE": 113.87,  "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 47.1, "PKG_SIZE": "Large",    "PRIORITY": "Low",    "WEATHER": "Storm",        "SHIP_COST": 297.40, "DELAY_DAYS": 9,  "LATE": "Yes"},
        {"ORD_ID": "ORD-015001", "CATEGORY": "Pet Supplies",           "SHIP_METHOD": "Standard",      "CARRIER": "SpeedyCargo",         "DISTANCE_KM": 4079.6, "WEIGHT_KG": 1.26, "ORDER_VALUE": 159.90,  "PROMISED_DAYS": 7,  "WAREHOUSE_HRS": 8.6,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Clear",        "SHIP_COST": 133.65, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-016001", "CATEGORY": "Electronics",            "SHIP_METHOD": "Standard",      "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 4054.4, "WEIGHT_KG": 0.10, "ORDER_VALUE": 317.88,  "PROMISED_DAYS": 7,  "WAREHOUSE_HRS": 21.7, "PKG_SIZE": "Small",    "PRIORITY": "Low",    "WEATHER": "Rain",         "SHIP_COST": 91.03,  "DELAY_DAYS": 4,  "LATE": "Yes"},
        {"ORD_ID": "ORD-017001", "CATEGORY": "Sports & Fitness",       "SHIP_METHOD": "Economy",       "CARRIER": "SwiftShip",           "DISTANCE_KM": 4756.2, "WEIGHT_KG": 3.48, "ORDER_VALUE": 176.90,  "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 26.7, "PKG_SIZE": "Large",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 144.12, "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-018001", "CATEGORY": "Office Supplies",        "SHIP_METHOD": "International", "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 4391.8, "WEIGHT_KG": 1.18, "ORDER_VALUE": 34.75,   "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 9.6,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Cloudy",       "SHIP_COST": 252.43, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-019001", "CATEGORY": "Automotive",             "SHIP_METHOD": "International", "CARRIER": "EagleCourier",        "DISTANCE_KM": 5260.6, "WEIGHT_KG": 15.80,"ORDER_VALUE": 157.47,  "PROMISED_DAYS": 13, "WAREHOUSE_HRS": 23.3, "PKG_SIZE": "Oversized","PRIORITY": "Low",    "WEATHER": "Clear",        "SHIP_COST": 500.00, "DELAY_DAYS": 2,  "LATE": "Yes"},
        {"ORD_ID": "ORD-020001", "CATEGORY": "Electronics",            "SHIP_METHOD": "Standard",      "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 443.5,  "WEIGHT_KG": 0.10, "ORDER_VALUE": 989.50,  "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 28.6, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 12.61,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-021001", "CATEGORY": "Fashion",                "SHIP_METHOD": "Standard",      "CARRIER": "ParcelPro",           "DISTANCE_KM": 1523.6, "WEIGHT_KG": 0.47, "ORDER_VALUE": 87.84,   "PROMISED_DAYS": 5,  "WAREHOUSE_HRS": 40.4, "PKG_SIZE": "Small",    "PRIORITY": "Low",    "WEATHER": "Rain",         "SHIP_COST": 34.15,  "DELAY_DAYS": 2,  "LATE": "Yes"},
        {"ORD_ID": "ORD-022001", "CATEGORY": "Books",                  "SHIP_METHOD": "Express",       "CARRIER": "ParcelPro",           "DISTANCE_KM": 3175.6, "WEIGHT_KG": 0.64, "ORDER_VALUE": 73.82,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 7.1,  "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Cloudy",       "SHIP_COST": 123.14, "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-023001", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "Express",       "CARRIER": "ParcelPro",           "DISTANCE_KM": 3286.7, "WEIGHT_KG": 0.88, "ORDER_VALUE": 30.20,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 3.9,  "PKG_SIZE": "Medium",   "PRIORITY": "Urgent", "WEATHER": "Clear",        "SHIP_COST": 152.62, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-024001", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "International", "CARRIER": "ParcelPro",           "DISTANCE_KM": 5251.0, "WEIGHT_KG": 0.68, "ORDER_VALUE": 28.16,   "PROMISED_DAYS": 13, "WAREHOUSE_HRS": 26.8, "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Rain",         "SHIP_COST": 340.48, "DELAY_DAYS": 4,  "LATE": "Yes"},
        {"ORD_ID": "ORD-025001", "CATEGORY": "Office Supplies",        "SHIP_METHOD": "Standard",      "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 1680.4, "WEIGHT_KG": 1.31, "ORDER_VALUE": 58.98,   "PROMISED_DAYS": 5,  "WAREHOUSE_HRS": 9.6,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Clear",        "SHIP_COST": 49.80,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-026001", "CATEGORY": "Automotive",             "SHIP_METHOD": "International", "CARRIER": "EagleCourier",        "DISTANCE_KM": 807.5,  "WEIGHT_KG": 6.96, "ORDER_VALUE": 495.72,  "PROMISED_DAYS": 10, "WAREHOUSE_HRS": 12.8, "PKG_SIZE": "Large",    "PRIORITY": "Low",    "WEATHER": "Snow",         "SHIP_COST": 83.49,  "DELAY_DAYS": 8,  "LATE": "Yes"},
        {"ORD_ID": "ORD-027001", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "Standard",      "CARRIER": "GlobalExpress",       "DISTANCE_KM": 3027.0, "WEIGHT_KG": 0.82, "ORDER_VALUE": 44.59,   "PROMISED_DAYS": 6,  "WAREHOUSE_HRS": 10.1, "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 90.02,  "DELAY_DAYS": 2,  "LATE": "Yes"},
        {"ORD_ID": "ORD-028001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "Standard",      "CARRIER": "SwiftShip",           "DISTANCE_KM": 2356.8, "WEIGHT_KG": 0.59, "ORDER_VALUE": 680.08,  "PROMISED_DAYS": 6,  "WAREHOUSE_HRS": 21.9, "PKG_SIZE": "Medium",   "PRIORITY": "Normal", "WEATHER": "Snow",         "SHIP_COST": 62.27,  "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-029001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "Standard",      "CARRIER": "SpeedyCargo",         "DISTANCE_KM": 711.0,  "WEIGHT_KG": 3.14, "ORDER_VALUE": 156.11,  "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 20.1, "PKG_SIZE": "Large",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 34.51,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-030001", "CATEGORY": "Fashion",                "SHIP_METHOD": "International", "CARRIER": "PrimeDelivery",       "DISTANCE_KM": 3580.4, "WEIGHT_KG": 0.10, "ORDER_VALUE": 155.10,  "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 19.5, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Cloudy",       "SHIP_COST": 154.82, "DELAY_DAYS": 2,  "LATE": "Yes"},
        {"ORD_ID": "ORD-031001", "CATEGORY": "Toys",                   "SHIP_METHOD": "Express",       "CARRIER": "ParcelPro",           "DISTANCE_KM": 212.7,  "WEIGHT_KG": 0.54, "ORDER_VALUE": 10.00,   "PROMISED_DAYS": 2,  "WAREHOUSE_HRS": 14.7, "PKG_SIZE": "Large",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 25.43,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-032001", "CATEGORY": "Books",                  "SHIP_METHOD": "Express",       "CARRIER": "EagleCourier",        "DISTANCE_KM": 1747.6, "WEIGHT_KG": 0.40, "ORDER_VALUE": 15.01,   "PROMISED_DAYS": 3,  "WAREHOUSE_HRS": 18.1, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Rain",         "SHIP_COST": 57.12,  "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-033001", "CATEGORY": "Beauty",                 "SHIP_METHOD": "Economy",       "CARRIER": "FastTrack Logistics", "DISTANCE_KM": 793.4,  "WEIGHT_KG": 0.53, "ORDER_VALUE": 148.38,  "PROMISED_DAYS": 8,  "WAREHOUSE_HRS": 4.9,  "PKG_SIZE": "Medium",   "PRIORITY": "Urgent", "WEATHER": "Storm",        "SHIP_COST": 19.92,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-034001", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "International", "CARRIER": "ParcelPro",           "DISTANCE_KM": 3489.7, "WEIGHT_KG": 0.54, "ORDER_VALUE": 10.20,   "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 11.7, "PKG_SIZE": "Large",    "PRIORITY": "High",   "WEATHER": "Cloudy",       "SHIP_COST": 287.52, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-035001", "CATEGORY": "Toys",                   "SHIP_METHOD": "Standard",      "CARRIER": "GlobalExpress",       "DISTANCE_KM": 45.5,   "WEIGHT_KG": 0.77, "ORDER_VALUE": 59.88,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 2.9,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Clear",        "SHIP_COST": 7.79,   "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-036001", "CATEGORY": "Sports & Fitness",       "SHIP_METHOD": "International", "CARRIER": "GlobalExpress",       "DISTANCE_KM": 1477.8, "WEIGHT_KG": 3.86, "ORDER_VALUE": 83.76,   "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 29.4, "PKG_SIZE": "Large",    "PRIORITY": "Normal", "WEATHER": "Rain",         "SHIP_COST": 104.84, "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-037001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "Standard",      "CARRIER": "SwiftShip",           "DISTANCE_KM": 1793.4, "WEIGHT_KG": 4.36, "ORDER_VALUE": 131.64,  "PROMISED_DAYS": 5,  "WAREHOUSE_HRS": 16.3, "PKG_SIZE": "Large",    "PRIORITY": "Low",    "WEATHER": "Snow",         "SHIP_COST": 66.88,  "DELAY_DAYS": 3,  "LATE": "Yes"},
        {"ORD_ID": "ORD-038001", "CATEGORY": "Electronics",            "SHIP_METHOD": "International", "CARRIER": "SpeedyCargo",         "DISTANCE_KM": 1893.2, "WEIGHT_KG": 4.74, "ORDER_VALUE": 417.56,  "PROMISED_DAYS": 11, "WAREHOUSE_HRS": 6.9,  "PKG_SIZE": "Large",    "PRIORITY": "High",   "WEATHER": "Storm",        "SHIP_COST": 136.36, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-039001", "CATEGORY": "Electronics",            "SHIP_METHOD": "International", "CARRIER": "GlobalExpress",       "DISTANCE_KM": 3421.5, "WEIGHT_KG": 0.10, "ORDER_VALUE": 739.15,  "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 19.4, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 172.76, "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-040001", "CATEGORY": "Home & Kitchen",         "SHIP_METHOD": "Economy",       "CARRIER": "SpeedyCargo",         "DISTANCE_KM": 1193.1, "WEIGHT_KG": 6.97, "ORDER_VALUE": 15.62,   "PROMISED_DAYS": 9,  "WAREHOUSE_HRS": 5.4,  "PKG_SIZE": "Large",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 38.28,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-041001", "CATEGORY": "Pet Supplies",           "SHIP_METHOD": "Standard",      "CARRIER": "PrimeDelivery",       "DISTANCE_KM": 6958.6, "WEIGHT_KG": 1.84, "ORDER_VALUE": 14.00,   "PROMISED_DAYS": 8,  "WAREHOUSE_HRS": 2.8,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Rain",         "SHIP_COST": 224.88, "DELAY_DAYS": 4,  "LATE": "Yes"},
        {"ORD_ID": "ORD-042001", "CATEGORY": "Pet Supplies",           "SHIP_METHOD": "Express",       "CARRIER": "SwiftShip",           "DISTANCE_KM": 3482.3, "WEIGHT_KG": 1.97, "ORDER_VALUE": 78.61,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 5.9,  "PKG_SIZE": "Medium",   "PRIORITY": "High",   "WEATHER": "Cloudy",       "SHIP_COST": 143.34, "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-043001", "CATEGORY": "Electronics",            "SHIP_METHOD": "Standard",      "CARRIER": "ParcelPro",           "DISTANCE_KM": 2778.9, "WEIGHT_KG": 5.17, "ORDER_VALUE": 334.67,  "PROMISED_DAYS": 6,  "WAREHOUSE_HRS": 10.8, "PKG_SIZE": "Large",    "PRIORITY": "High",   "WEATHER": "Cloudy",       "SHIP_COST": 116.28, "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-044001", "CATEGORY": "Health & Personal Care", "SHIP_METHOD": "Economy",       "CARRIER": "GlobalExpress",       "DISTANCE_KM": 759.1,  "WEIGHT_KG": 0.99, "ORDER_VALUE": 38.88,   "PROMISED_DAYS": 8,  "WAREHOUSE_HRS": 5.5,  "PKG_SIZE": "Large",    "PRIORITY": "High",   "WEATHER": "Rain",         "SHIP_COST": 23.03,  "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-045001", "CATEGORY": "Fashion",                "SHIP_METHOD": "International", "CARRIER": "PrimeDelivery",       "DISTANCE_KM": 3948.6, "WEIGHT_KG": 0.58, "ORDER_VALUE": 74.75,   "PROMISED_DAYS": 12, "WAREHOUSE_HRS": 11.1, "PKG_SIZE": "Large",    "PRIORITY": "Low",    "WEATHER": "Cloudy",       "SHIP_COST": 325.55, "DELAY_DAYS": 4,  "LATE": "Yes"},
        {"ORD_ID": "ORD-046001", "CATEGORY": "Fashion",                "SHIP_METHOD": "Express",       "CARRIER": "ParcelPro",           "DISTANCE_KM": 5717.7, "WEIGHT_KG": 1.36, "ORDER_VALUE": 162.95,  "PROMISED_DAYS": 5,  "WAREHOUSE_HRS": 7.4,  "PKG_SIZE": "Large",    "PRIORITY": "Urgent", "WEATHER": "Rain",         "SHIP_COST": 326.74, "DELAY_DAYS": 1,  "LATE": "Yes"},
        {"ORD_ID": "ORD-047001", "CATEGORY": "Beauty",                 "SHIP_METHOD": "Standard",      "CARRIER": "BlueRoute",           "DISTANCE_KM": 179.1,  "WEIGHT_KG": 0.37, "ORDER_VALUE": 67.76,   "PROMISED_DAYS": 4,  "WAREHOUSE_HRS": 21.8, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 9.13,   "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-048001", "CATEGORY": "Electronics",            "SHIP_METHOD": "Economy",       "CARRIER": "SwiftShip",           "DISTANCE_KM": 121.5,  "WEIGHT_KG": 0.10, "ORDER_VALUE": 1988.01, "PROMISED_DAYS": 8,  "WAREHOUSE_HRS": 19.5, "PKG_SIZE": "Small",    "PRIORITY": "Normal", "WEATHER": "Clear",        "SHIP_COST": 6.68,   "DELAY_DAYS": 0,  "LATE": "No"},
        {"ORD_ID": "ORD-049001", "CATEGORY": "Toys",                   "SHIP_METHOD": "Economy",       "CARRIER": "GlobalExpress",       "DISTANCE_KM": 1352.5, "WEIGHT_KG": 0.55, "ORDER_VALUE": 93.86,   "PROMISED_DAYS": 9,  "WAREHOUSE_HRS": 5.3,  "PKG_SIZE": "Medium",   "PRIORITY": "Urgent", "WEATHER": "Clear",        "SHIP_COST": 24.34,  "DELAY_DAYS": 0,  "LATE": "No"},
        # --- 5 test rows: predict DELAY_DAYS (one per shipping method) ---
        *[{**r, "DELAY_DAYS": "[PREDICT]"} for r in _REGRESSION_TEST_ROWS],
        # --- 5 test rows: predict LATE (3×Yes, 2×No) ---
        *[{**r, "LATE": "[PREDICT]"} for r in _CLASSIFICATION_TEST_ROWS],
    ],
}

# ---------------------------------------------------------------------------
# Feature columns + agent state
# ---------------------------------------------------------------------------
# SHIP_COST is a known feature for all rows; it was the former regression target
# and is now a predictor for DELAY_DAYS.
FEATURE_COLS = ["CATEGORY", "DISTANCE_KM", "WEIGHT_KG", "ORDER_VALUE", "PROMISED_DAYS",
                "WAREHOUSE_HRS", "SHIP_METHOD", "CARRIER", "PKG_SIZE", "PRIORITY", "WEATHER",
                "SHIP_COST"]


class AgentState(TypedDict):
    analysis_result: Optional[str]
    tabpfn_result: Optional[str]


# ---------------------------------------------------------------------------
# Tool: Prior Labs TabPFN-3.5 Plus via SAP AI Core GenAI Hub
# ---------------------------------------------------------------------------
def call_tabpfn(payload: dict) -> str:
    """Send prediction requests to TabPFN-3.5 Plus deployed on SAP AI Core.

    For each target column, splits rows into training (known values) and test
    ([PREDICT] placeholder), then POSTs to the AI Core deployment endpoint.
    Returns all predictions as a JSON string.
    """
    try:
        token = _get_auth_token()
        base_url = os.environ["AICORE_BASE_URL"].rstrip("/")  # guard against double-slash in URL
        url = f"{base_url}/v2/inference/deployments/{TABPFN_DEPLOYMENT_ID}/predict"
        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": os.environ.get("AICORE_RESOURCE_GROUP", "default"),
            "Content-Type": "application/json",
        }

        rows = payload["rows"]
        target_columns = payload["prediction_config"]["target_columns"]
        results: dict = {"predictions": {}}

        for target in target_columns:
            col_name    = target["name"]
            placeholder = target["prediction_placeholder"]
            task_type   = target["task_type"]

            # .get() instead of direct access because LATE-only test rows have no DELAY_DAYS key
            train_rows = [r for r in rows if r.get(col_name) is not None and r.get(col_name) != placeholder]
            test_rows  = [r for r in rows if r.get(col_name) == placeholder]

            if not test_rows:
                results["predictions"][col_name] = []
                continue

            train_df = pd.DataFrame(train_rows)
            test_df  = pd.DataFrame(test_rows)

            x_train = {col: train_df[col].tolist() for col in FEATURE_COLS}
            x_test  = {col: test_df[col].tolist() for col in FEATURE_COLS}

            if task_type == "regression":
                request_payload = {
                    "task_config": {
                        "task": "regression",
                        "predict_params": {"output_type": "mean"},
                    },
                    "x_train": x_train,
                    "y_train": train_df[col_name].astype(float).tolist(),  # plain list for regression
                    "x_test": x_test,
                }
            else:
                request_payload = {
                    "task_config": {
                        "task": "classification",
                        "tabpfn_config": {
                            "n_estimators": 8,
                            "inference_precision": "auto",
                        },
                        "predict_params": {"output_type": "preds"},
                    },
                    "x_train": x_train,
                    "y_train": {col_name: train_df[col_name].tolist()},  # classification requires a dict keyed by column name
                    "x_test": x_test,
                }

            print(f"\n--- TabPFN request [{col_name}] ---")
            print(f"URL: {url}")
            print(json.dumps(request_payload, indent=2))

            response = requests.post(url, headers=headers, json=request_payload, timeout=300)  # TabPFN can take up to ~60s for larger contexts

            print(f"HTTP {response.status_code}")
            print(f"Response body: {response.text[:500]}")  # truncated to keep terminal output readable

            response.raise_for_status()
            api_result = response.json()

            predictions = api_result["prediction"]
            results["predictions"][col_name] = [
                {"ORD_ID": row["ORD_ID"], "SHIP_METHOD": row["SHIP_METHOD"], col_name: p}
                for row, p in zip(test_rows, predictions)
            ]

        return json.dumps(results, indent=2)

    except requests.RequestException as e:
        return f"Error calling TabPFN: {str(e)}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
system_prompt = """You are a senior e-commerce logistics analyst.
You have received TabPFN-3.5 Plus predictions for two missing fields in a shipment dataset:
- DELAY_DAYS (integer): the predicted number of days an order arrives after its promised delivery date.
- LATE (Yes/No): whether each order is predicted to arrive after its promised delivery date.

Your role is to summarise the predictions in a clear operations report:
- List the predicted delivery delay (in days) for each order, grouped by shipping method.
- List the predicted on-time / late status for each order.
- Highlight any patterns (e.g. which shipping methods, carriers, or weather conditions correlate with longer delays).

You rely exclusively on the model predictions provided — never estimate or adjust values yourself."""


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------
def logistics_node(state: AgentState) -> dict:
    print("\nLogistics Agent starting...")

    tabpfn_result = call_tabpfn(payload)

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here are the Prior Labs TabPFN-3.5 Plus predictions for the missing delivery fields. Write a concise logistics operations summary:\n\n{tabpfn_result}"),
    ])

    analysis_result = response.content
    print("Analysis complete")

    return {
        "analysis_result": analysis_result,
        "tabpfn_result": tabpfn_result,  # stored in state so evaluate_predictions can run without a second API call
    }


# ---------------------------------------------------------------------------
# Evaluation: predicted vs ground truth
# ---------------------------------------------------------------------------
def evaluate_predictions(tabpfn_result_json: str) -> None:
    if not tabpfn_result_json or tabpfn_result_json.startswith("Error"):
        print(f"\nSkipping evaluation — TabPFN returned an error:\n{tabpfn_result_json}")
        return

    results = json.loads(tabpfn_result_json)

    print("\n" + "=" * 65)
    print("Prediction vs Ground Truth")
    print("=" * 65)

    # Regression: DELAY_DAYS
    delay_preds = results["predictions"].get("DELAY_DAYS", [])
    if delay_preds:
        print("\nDELAY_DAYS  (regression, days)")
        print(f"  {'Order ID':<14} {'Method':<15} {'Predicted':>10} {'Actual':>10} {'Abs Err':>9}")
        print("  " + "-" * 64)
        abs_errors = []
        for p in delay_preds:
            ord_id    = p["ORD_ID"]
            predicted = float(p["DELAY_DAYS"])
            actual    = GROUND_TRUTH.get(ord_id, {}).get("DELAY_DAYS")
            if actual is None:
                continue
            err = abs(predicted - actual)
            abs_errors.append(err)
            print(f"  {ord_id:<14} {p['SHIP_METHOD']:<15} {predicted:>10.2f} {actual:>10} {err:>9.2f}")
        if abs_errors:
            print(f"\n  MAE: {sum(abs_errors) / len(abs_errors):.2f} days  |  Max error: {max(abs_errors):.2f} days")

    # Classification: LATE
    late_preds = results["predictions"].get("LATE", [])
    if late_preds:
        print("\nLATE  (classification, Yes/No)")
        print(f"  {'Order ID':<14} {'Method':<15} {'Predicted':<12} {'Actual':<12} {'':>4}")
        print("  " + "-" * 58)
        correct = 0
        for p in late_preds:
            ord_id    = p["ORD_ID"]
            predicted = str(p["LATE"])
            actual    = GROUND_TRUTH.get(ord_id, {}).get("LATE", "?")
            hit = predicted == actual
            if hit:
                correct += 1
            mark = "OK" if hit else "--"
            print(f"  {ord_id:<14} {p['SHIP_METHOD']:<15} {predicted:<12} {actual:<12} {mark:>4}")
        print(f"\n  Accuracy: {correct}/{len(late_preds)}  ({correct / len(late_preds) * 100:.0f}%)")


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("logistics", logistics_node)
    workflow.add_edge(START, "logistics")
    workflow.add_edge("logistics", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    result = app.invoke({
        "analysis_result": None,
        "tabpfn_result": None,
    })

    print("\n" + "=" * 50)
    print("E-commerce Logistics Report:")
    print("=" * 50)
    print(result["analysis_result"])

    evaluate_predictions(result["tabpfn_result"])


if __name__ == "__main__":
    main()

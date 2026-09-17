"""
Stolen Goods Loss Appraiser - LangGraph Agent with Prior Labs TabPFN-3.5 Plus
Based on SAP CodeJam exercises 02 & 03 (Python-LangGraph), using
TabPFN-3.5 Plus deployed on SAP AI Core GenAI Hub.

Requirements:
    pip install langgraph langchain-litellm langchain-core requests pandas python-dotenv

Config:
    .env  - AI Core credentials (AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
            AICORE_BASE_URL, AICORE_RESOURCE_GROUP)
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

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TABPFN_DEPLOYMENT_ID = os.environ["TABPFN_DEPLOYMENT_ID"]

# Initialize the LLM via LiteLLM pointing to SAP Generative AI Hub
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
# Payload – same structure as exercise 03 payload.py
# ---------------------------------------------------------------------------
payload = {
    "prediction_config": {
        "target_columns": [
            {
                "name": "INSURANCE_VALUE",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "regression",
            },
            {
                "name": "ITEM_CATEGORY",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "classification",
            },
        ]
    },
    "index_column": "ITEM_ID",
    "rows": [
        {"ITEM_ID": "ART_001", "ITEM_NAME": "Water Lilies - Series 1",         "ARTIST": "Claude Monet",       "ACQUISITION_DATE": "1987-03-15", "INSURANCE_VALUE": 45000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "200x180cm",    "CONDITION_SCORE": 9,  "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 8},
        {"ITEM_ID": "ART_002", "ITEM_NAME": "Japanese Bridge at Giverny",       "ARTIST": "Claude Monet",       "ACQUISITION_DATE": "1995-06-22", "INSURANCE_VALUE": 42000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "92x73cm",      "CONDITION_SCORE": 8,  "RARITY_SCORE": 8,  "PROVENANCE_CLARITY": 9},
        {"ITEM_ID": "ART_003", "ITEM_NAME": "Irises",                           "ARTIST": "Vincent van Gogh",   "ACQUISITION_DATE": "2001-11-08", "INSURANCE_VALUE": "[PREDICT]", "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "71x93cm",      "CONDITION_SCORE": 7,  "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 8},
        {"ITEM_ID": "ART_004", "ITEM_NAME": "Starry Night Over the Rhone",      "ARTIST": "Vincent van Gogh",   "ACQUISITION_DATE": "1998-09-14", "INSURANCE_VALUE": 48000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "73x92cm",      "CONDITION_SCORE": 8,  "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 9},
        {"ITEM_ID": "ART_005", "ITEM_NAME": "The Birth of Venus",               "ARTIST": "Sandro Botticelli",  "ACQUISITION_DATE": "1992-04-30", "INSURANCE_VALUE": 55000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "172x278cm",    "CONDITION_SCORE": 6,  "RARITY_SCORE": 10, "PROVENANCE_CLARITY": 10},
        {"ITEM_ID": "ART_006", "ITEM_NAME": "Primavera",                        "ARTIST": "Sandro Botticelli",  "ACQUISITION_DATE": "1989-02-19", "INSURANCE_VALUE": 52000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "203x314cm",    "CONDITION_SCORE": 7,  "RARITY_SCORE": 10, "PROVENANCE_CLARITY": 10},
        {"ITEM_ID": "ART_007", "ITEM_NAME": "Girl with a Pearl Earring",        "ARTIST": "Johannes Vermeer",   "ACQUISITION_DATE": "2003-07-11", "INSURANCE_VALUE": "[PREDICT]", "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "44x39cm",      "CONDITION_SCORE": 8,  "RARITY_SCORE": 10, "PROVENANCE_CLARITY": 9},
        {"ITEM_ID": "ART_008", "ITEM_NAME": "The Music Lesson",                 "ARTIST": "Johannes Vermeer",   "ACQUISITION_DATE": "1994-05-20", "INSURANCE_VALUE": 38000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "64x73cm",      "CONDITION_SCORE": 8,  "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 9},
        {"ITEM_ID": "ART_009", "ITEM_NAME": "The Persistence of Memory",        "ARTIST": "Salvador Dalí",      "ACQUISITION_DATE": "2005-03-10", "INSURANCE_VALUE": 35000000, "ITEM_CATEGORY": "[PREDICT]",  "DIMENSIONS": "24x33cm",      "CONDITION_SCORE": 9,  "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 10},
        {"ITEM_ID": "ART_010", "ITEM_NAME": "Metamorphosis of Narcissus",       "ARTIST": "Salvador Dalí",      "ACQUISITION_DATE": "1996-08-12", "INSURANCE_VALUE": 32000000, "ITEM_CATEGORY": "Painting",   "DIMENSIONS": "51x78cm",      "CONDITION_SCORE": 8,  "RARITY_SCORE": 8,  "PROVENANCE_CLARITY": 8},
        {"ITEM_ID": "ART_011", "ITEM_NAME": "The Bronze Dancer",                "ARTIST": "Auguste Rodin",      "ACQUISITION_DATE": "1991-07-22", "INSURANCE_VALUE": 8500000,  "ITEM_CATEGORY": "Sculpture",  "DIMENSIONS": "Height: 1.8m", "CONDITION_SCORE": 9,  "RARITY_SCORE": 7,  "PROVENANCE_CLARITY": 8},
        {"ITEM_ID": "ART_012", "ITEM_NAME": "The Thinker",                      "ARTIST": "Auguste Rodin",      "ACQUISITION_DATE": "2000-11-05", "INSURANCE_VALUE": "[PREDICT]", "ITEM_CATEGORY": "Sculpture",  "DIMENSIONS": "Height: 1.9m", "CONDITION_SCORE": 9,  "RARITY_SCORE": 7,  "PROVENANCE_CLARITY": 9},
        {"ITEM_ID": "ART_013", "ITEM_NAME": "Hope Diamond Replica - Royal Cut", "ARTIST": "Unknown Jeweler",    "ACQUISITION_DATE": "1988-02-19", "INSURANCE_VALUE": 12000000, "ITEM_CATEGORY": "Jewelry",    "DIMENSIONS": "Width: 15cm",  "CONDITION_SCORE": 10, "RARITY_SCORE": 10, "PROVENANCE_CLARITY": 7},
        {"ITEM_ID": "ART_014", "ITEM_NAME": "Cartier Ruby Necklace - 1920s",    "ARTIST": "Cartier",            "ACQUISITION_DATE": "2002-09-11", "INSURANCE_VALUE": 9500000,  "ITEM_CATEGORY": "Jewelry",    "DIMENSIONS": "Length: 45cm", "CONDITION_SCORE": 9,  "RARITY_SCORE": 8,  "PROVENANCE_CLARITY": 9},
    ],
    "data_schema": {
        "ITEM_ID":            {"dtype": "string"},
        "ITEM_NAME":          {"dtype": "string"},
        "ARTIST":             {"dtype": "string"},
        "ACQUISITION_DATE":   {"dtype": "date"},
        "INSURANCE_VALUE":    {"dtype": "numeric"},
        "ITEM_CATEGORY":      {"dtype": "string", "categories": ["Painting", "Sculpture", "Jewelry"]},
        "DIMENSIONS":         {"dtype": "string"},
        "CONDITION_SCORE":    {"dtype": "numeric", "range": [1, 10], "description": "1=Poor to 10=Pristine"},
        "RARITY_SCORE":       {"dtype": "numeric", "range": [1, 10], "description": "1=Common to 10=Extremely Rare"},
        "PROVENANCE_CLARITY": {"dtype": "numeric", "range": [1, 10], "description": "1=Unknown to 10=Perfect Documentation"},
    },
}

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
FEATURE_COLS = ["CONDITION_SCORE", "RARITY_SCORE", "PROVENANCE_CLARITY"]


class AgentState(TypedDict):
    appraisal_result: Optional[str]
    messages: list


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
        base_url = os.environ["AICORE_BASE_URL"].rstrip("/")
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

            train_rows = [r for r in rows if r[col_name] != placeholder]
            test_rows  = [r for r in rows if r[col_name] == placeholder]

            if not test_rows:
                results["predictions"][col_name] = []
                continue

            train_df = pd.DataFrame(train_rows)
            test_df  = pd.DataFrame(test_rows)

            # Column-oriented dict format — TabPFN handles string labels natively
            x_train = {col: train_df[col].tolist() for col in FEATURE_COLS}
            x_test  = {col: test_df[col].tolist() for col in FEATURE_COLS}

            if task_type == "regression":
                request_payload = {
                    "task_config": {
                        "task": "regression",
                        "predict_params": {"output_type": "mean"},
                    },
                    "x_train": x_train,
                    "y_train": train_df[col_name].astype(float).tolist(),
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
                    "y_train": {col_name: train_df[col_name].tolist()},
                    "x_test": x_test,
                }

            print(f"\n--- TabPFN request [{col_name}] ---")
            print(f"URL: {url}")
            print(json.dumps(request_payload, indent=2))

            response = requests.post(url, headers=headers, json=request_payload, timeout=300)

            print(f"HTTP {response.status_code}")
            print(f"Response body: {response.text[:500]}")

            response.raise_for_status()
            api_result = response.json()

            predictions = api_result["prediction"]
            results["predictions"][col_name] = [
                {"ITEM_ID": row["ITEM_ID"], "ITEM_NAME": row["ITEM_NAME"], col_name: p}
                for row, p in zip(test_rows, predictions)
            ]

        return json.dumps(results, indent=2)

    except Exception as e:
        return f"Error calling TabPFN: {str(e)}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
system_prompt = """You are a certified Fine Art & Valuables Insurance Claims Specialist at a leading specialty insurer.
A policyholder has filed a theft claim covering items from their insured collection. Your role is to review the AI-assisted valuations produced by Prior Labs TabPFN-3.5 Plus and prepare a formal claims appraisal report that:
- States the predicted insurance replacement value for each unvalued item in the claim.
- Confirms the predicted category of any item whose classification was missing from the policy schedule.
- Summarises total estimated claim exposure across the portfolio.

You rely exclusively on the model predictions provided — never estimate or adjust values yourself.
Write in the professional, factual tone expected in a formal insurance claims document."""


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------
def appraiser_node(state: AgentState) -> dict:
    print("\nAppraiser Agent starting...")

    tabpfn_result = call_tabpfn(payload)

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here are the Prior Labs TabPFN-3.5 Plus predictions for the items in this insurance claim. Write a professional appraisal summary:\n\n{tabpfn_result}"),
    ])

    appraisal_result = response.content
    print("Appraisal complete")

    return {
        "appraisal_result": appraisal_result,
        "messages": state["messages"] + [{"role": "assistant", "content": appraisal_result}],
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("appraiser", appraiser_node)
    workflow.add_edge(START, "appraiser")
    workflow.add_edge("appraiser", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    result = app.invoke({
        "appraisal_result": None,
        "messages": [],
    })

    print("\n" + "=" * 50)
    print("Insurance Appraiser Report:")
    print("=" * 50)
    print(result["appraisal_result"])


if __name__ == "__main__":
    main()

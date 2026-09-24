from pathlib import Path
import certifi
import os
import json

import pandas as pd
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain.agents import create_agent
from langchain.tools import tool
from hana_ml import dataframe
from hana_ml.model_storage import ModelStorage

# Load credentials from the .env file two levels up (repo root)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Point the TLS stack at the certifi bundle so HANA Cloud connections succeed
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Identifies the PAL model saved in HANA Model Storage
MODEL_NAME = "my_hana_ai_model"
MODEL_VERSION = 14
KEY_COLUMN = "BOOKING_DATE"


def _get_connection():
    """Open a HANA Cloud connection using credentials from the environment."""
    return dataframe.ConnectionContext(
        address=os.environ["HANA_HOST"],
        port=int(os.environ["HANA_PORT"]),
        user=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"],
        sslValidateCertificate=False,
        encrypt=True,
    )


@tool
def run_forecast(dates: list[str]) -> str:
    """Run a time-series forecast for the given dates using the saved HANA PAL model.

    Args:
        dates: List of dates to forecast in YYYY-MM-DD format (e.g. ["2024-10-01", "2024-10-02"]).
               The model was trained on daily sales data, so provide one entry per day.
    """
    if not dates:
        return json.dumps({"error": "No dates provided."})

    cc = _get_connection()
    try:
        # Upload the requested dates as a temporary HANA table for the PAL model to read
        df = pd.DataFrame({KEY_COLUMN: pd.to_datetime(dates)})
        predict_hdf = dataframe.create_dataframe_from_pandas(
            cc, df, "#FORECAST_INPUT", force=True, drop_exist_tab=True
        )

        # Load the saved PAL model from Model Storage and run prediction
        ms = ModelStorage(connection_context=cc)
        pal_model = ms.load_model(MODEL_NAME, MODEL_VERSION)
        pal_model.predict(data=predict_hdf, key=KEY_COLUMN)

        # Read results back from the output table PAL wrote to
        result_table = pal_model._predict_output_table_names[0]
        rows = cc.table(result_table).collect().to_dict(orient="records")

        return json.dumps({
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "dates_requested": len(dates),
            "result_table": result_table,
            "row_count": len(rows),
            "predictions": rows
        }, default=str)
    finally:
        cc.close()


SYSTEM_PROMPT = """You are a time-series forecasting assistant connected to SAP HANA Cloud.
You have one tool: run_forecast. It accepts a list of dates (YYYY-MM-DD) and returns predicted daily sales values.
The model was trained on daily data, so always generate one date per day for the requested period.

When the user asks for a forecast (e.g. "forecast for October", "predict next week", "run prediction for Q4"):
1. Determine the exact date range from the user's request. Use year 2024 if no year is specified.
2. Generate every calendar day in that range as a YYYY-MM-DD string.
3. Call run_forecast immediately with that full list — do not ask for confirmation.

If the tool raises an error, report it clearly. Do not fabricate forecast results.
If it succeeds, present the predictions in a readable table or summary."""

llm = ChatLiteLLM(model="sap/anthropic--claude-4.6-sonnet", temperature=0)

# create_agent wires the LLM, tools, and system prompt into a runnable agent
forecast_agent = create_agent(
    model=llm,
    tools=[run_forecast],
    system_prompt=SYSTEM_PROMPT
)

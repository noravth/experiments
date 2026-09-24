import json
import os

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_executor import ForecastExecutor


app_url = (
    lambda d: f"https://{d.get('application_uris', [])[0]}"
    if d.get("application_uris")
    else None
)(json.loads(os.environ.get("VCAP_APPLICATION", "{}"))) or "http://localhost:8080"

agent_card = AgentCard(
    name="Forecast Agent",
    description="Time-series forecasting agent using SAP HANA PAL models, exposed as an A2A server",
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/markdown"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            url=app_url,
        )
    ],
    skills=[
        AgentSkill(
            id="forecast",
            name="Run Sales Forecast",
            description="Runs a time-series forecast for a given date range using a saved HANA PAL model",
            tags=["forecast", "sales", "time-series", "hana", "pal"],
        )
    ],
)

request_handler = DefaultRequestHandler(
    agent_executor=ForecastExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

routes = []
routes.extend(create_agent_card_routes(agent_card))
routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))


async def health(request: Request):
    return JSONResponse({"status": "ok"})


routes.append(Route("/health", health))

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

import asyncio

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Part,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from langchain_core.messages import HumanMessage

from agent_langchain import forecast_agent


class ForecastExecutor(AgentExecutor):
    """Bridges an incoming A2A task to the LangGraph forecast agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        user_input = context.get_user_input()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: forecast_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            ),
        )

        final_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                final_text = msg.content
                break
        if not final_text:
            final_text = "Forecast completed but no response was generated."

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=Artifact(
                    artifact_id="forecast_result",
                    name="forecast_result",
                    parts=[Part(text=final_text)],
                ),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            )
        )

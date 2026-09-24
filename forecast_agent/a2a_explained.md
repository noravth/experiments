# A2A Explained — How this Agent Server Works Under the Hood

## The A2A Protocol

**A2A (Agent-to-Agent)** is a protocol where agents expose themselves as HTTP servers. A caller sends a **task** (a message/request), and the server processes it asynchronously, emitting **events** as the work progresses. The caller gets a stream of state transitions: submitted → working → completed, plus any output artifacts.

The transport is **JSON-RPC over HTTP**. The server also publishes a static "agent card" at `/.well-known/agent.json` that describes what the agent can do — like an OpenAPI spec for agents.

---

## The Three-Layer Stack in `server.py`

```
HTTP Request
    │
    ▼
Starlette app  (routes defined in server.py)
    │
    ├── GET /.well-known/agent.json  → AgentCard  (describes the agent)
    │
    └── POST /  (JSON-RPC)
            │
            ▼
        DefaultRequestHandler  (a2a-sdk built-in)
            │  - deserializes the JSON-RPC envelope
            │  - creates/looks up a Task in InMemoryTaskStore
            │  - hands off to your executor
            ▼
        ForecastExecutor.execute()  ← your business logic lives here
            │
            ▼
        EventQueue  → events stream back through the handler to the caller
```

### 1. `AgentCard` — the agent's identity

```python
agent_card = AgentCard(
    name="Forecast Agent",
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=app_url)],
    skills=[AgentSkill(id="forecast", ...)]
)
```

This is static metadata served at `GET /.well-known/agent.json`. Callers (Joule, other agents) read it to discover what the agent does and how to call it.

### 2. `DefaultRequestHandler` — the protocol machinery

```python
request_handler = DefaultRequestHandler(
    agent_executor=ForecastExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)
```

The SDK's built-in handler does all the JSON-RPC plumbing — parsing `tasks/send`, managing task lifecycle, routing events back. You don't touch this. You only inject your executor and a task store.

`InMemoryTaskStore` keeps task state in-process (fine for single-instance CF deployments; swap for a Redis-backed store if you scale out).

### 3. Routes — wiring it all to HTTP

```python
routes.extend(create_agent_card_routes(agent_card))
routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
```

---

## What `AgentExecutor` Does Under the Hood

`ForecastExecutor` inherits from `AgentExecutor` and implements two async methods:

| Method | When called | What it must do |
|---|---|---|
| `execute(context, event_queue)` | Every incoming task | Emit events in order, run logic, emit result |
| `cancel(context, event_queue)` | Client cancels the task | Emit `TASK_STATE_CANCELED` |

### The event sequence inside `execute()`

```python
# 1. Acknowledge receipt
await event_queue.enqueue_event(
    Task(id=context.task_id, context_id=context.context_id,
         status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED))
)

# 2. Signal work has started
await event_queue.enqueue_event(
    TaskStatusUpdateEvent(..., status=TaskStatus(state=TaskState.TASK_STATE_WORKING))
)

# 3. Run the actual agent (blocking sync call, offloaded to a thread pool)
result = await loop.run_in_executor(None, lambda: forecast_agent.invoke(...))

# 4. Emit the result as an Artifact
await event_queue.enqueue_event(
    TaskArtifactUpdateEvent(..., artifact=Artifact(parts=[Part(text=final_text)]))
)

# 5. Signal completion
await event_queue.enqueue_event(
    TaskStatusUpdateEvent(..., status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED))
)
```

**Why `run_in_executor`?** The LangChain agent is synchronous. Wrapping it in `run_in_executor(None, ...)` offloads it to a thread pool so it doesn't block the async event loop. If your agent is already async, you can `await` it directly.

### `context: RequestContext` — what comes in from the caller

| Attribute | Value |
|---|---|
| `context.task_id` | Unique ID for this task — included in every event so the client can match responses |
| `context.context_id` | Conversation/session ID (for multi-turn interactions) |
| `context.get_user_input()` | The text message the caller sent |

---

## The Full Data Flow for One Request

```
Caller (Joule / another agent)
  │  POST /  {"jsonrpc":"2.0","method":"tasks/send","params":{"message":"forecast Q4..."}}
  ▼
DefaultRequestHandler
  - creates a Task in InMemoryTaskStore
  - calls ForecastExecutor.execute(context, event_queue)
  ▼
ForecastExecutor.execute()
  - enqueues SUBMITTED event
  - enqueues WORKING event
  - calls forecast_agent.invoke({"messages": [HumanMessage(user_input)]})
      └─ LangChain/LangGraph agent runs tools, calls HANA PAL, etc.
  - extracts last non-tool AI message as final_text
  - enqueues TaskArtifactUpdateEvent with result
  - enqueues COMPLETED event
  ▼
DefaultRequestHandler collects all events and returns a single JSON-RPC response
  (streaming=False means no live stream — the full result is returned once complete)
  ▼
Caller receives the artifact text
```

---

## Key Takeaways

- **`server.py`** is mostly boilerplate — `AgentCard`, `DefaultRequestHandler`, routes, and Starlette app wiring. It stays the same across agents.
- **`agent_executor.py`** is where business logic lives. Subclass `AgentExecutor`, implement the 4-event `execute()` pattern (SUBMITTED → WORKING → artifact → COMPLETED), plug in your agent invocation between step 2 and the artifact emit.
- The `AgentCard` is what makes the agent discoverable — think of it as the agent's public API contract.
- `InMemoryTaskStore` is fine for CF single-instance deployments. For multi-instance scale-out, replace it with a shared store.

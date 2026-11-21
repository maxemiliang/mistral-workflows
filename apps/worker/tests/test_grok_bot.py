"""The grok-bot workflow sends the persona plus conversation to the hosted model and returns its reply."""

import asyncio
from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from mistralai.workflows.testing import create_test_worker

from worker.workflows.grok_bot import GROK_PERSONA, GrokBotWorkflow

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)

received_requests: list[dict[str, Any]] = []


@workflows.activity(name="grok_chat")
async def fake_grok_chat(request: Any) -> dict[str, Any]:
    """Stand-in for the hosted Mistral chat call: records the request, returns a canned reply."""
    dumped = request.model_dump() if hasattr(request, "model_dump") else dict(request)
    received_requests.append(dumped)
    return {
        "reply": "Rayleigh scattering: the sky is blue because air molecules scatter short wavelengths.",
        "model": dumped["model"],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


async def test_grok_bot_answers_with_persona_and_conversation(temporal_env) -> None:  # noqa: ANN001
    async with create_test_worker(
        temporal_env,
        workflows=[GrokBotWorkflow],
        activities=[fake_grok_chat],
    ):
        handle = await temporal_env.client.start_workflow(
            "grok-bot",
            {
                "messages": [
                    {"role": "user", "content": "Why is the sky blue?"},
                    {"role": "assistant", "content": "Short answer: scattering."},
                    {"role": "user", "content": "Tell me more, and be witty about it."},
                ]
            },
            id="test-grok-bot",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)

    assert result["reply"].startswith("Rayleigh scattering")
    assert result["model"] == "mistral-large-latest"
    assert result["usage"]["total_tokens"] == 120

    assert len(received_requests) == 1
    request = received_requests[0]
    assert request["model"] == "mistral-large-latest"
    assert request["reasoning_effort"] == "low"
    assert request["max_tokens"] == 2048
    messages = request["messages"]
    assert messages[0]["role"] == "system"
    assert GROK_PERSONA.splitlines()[0] in messages[0]["content"]
    assert "Current date:" in messages[0]["content"]
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "Tell me more, and be witty about it."


async def test_grok_bot_passes_model_and_sampling_overrides(temporal_env) -> None:  # noqa: ANN001
    received_requests.clear()
    async with create_test_worker(
        temporal_env,
        workflows=[GrokBotWorkflow],
        activities=[fake_grok_chat],
    ):
        handle = await temporal_env.client.start_workflow(
            "grok-bot",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "mistral-small-latest",
                "temperature": 0.2,
                "max_tokens": 64,
                "reasoning_effort": "high",
            },
            id="test-grok-bot-overrides",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)

    assert result["model"] == "mistral-small-latest"
    request = received_requests[0]
    assert request["model"] == "mistral-small-latest"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 64
    assert request["reasoning_effort"] == "high"

"""grok-bot: plain persona chat path and the connector agent path."""

import asyncio
from datetime import timedelta
from typing import Any

import pytest
import mistralai.workflows as workflows
from mistralai.workflows.testing import create_test_worker

from worker.workflows.grok_bot import GROK_PERSONA, GrokBotWorkflow

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)

received_chats: list[dict[str, Any]] = []
received_agents: list[dict[str, Any]] = []


@workflows.activity(name="grok_chat")
async def fake_grok_chat(request: Any) -> dict[str, Any]:
    """Stand-in for the hosted chat completion: records the request, returns a canned reply."""
    dumped = request.model_dump() if hasattr(request, "model_dump") else dict(request)
    received_chats.append(dumped)
    return {
        "reply": "Rayleigh scattering: air molecules scatter short wavelengths more.",
        "model": dumped["model"],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


@workflows.activity(name="grok_agent_chat")
async def fake_grok_agent_chat(params: Any) -> dict[str, Any]:
    """Stand-in for the connector agent loop."""
    dumped = params.model_dump() if hasattr(params, "model_dump") else dict(params)
    received_agents.append(dumped)
    return {
        "reply": "Found it: github.com/maxemiliang/mistral-workflows (via GitHub connector).",
        "model": dumped["model"],
        "usage": {},
        "connectors_used": dumped["connector_names"],
    }


async def _run(temporal_env, input_data: dict[str, Any], execution_id: str) -> Any:  # noqa: ANN001
    async with create_test_worker(
        temporal_env,
        workflows=[GrokBotWorkflow],
        activities=[fake_grok_chat, fake_grok_agent_chat],
    ):
        handle = await temporal_env.client.start_workflow(
            "grok-bot",
            input_data,
            id=execution_id,
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        return await asyncio.wait_for(handle.result(), timeout=15)


async def test_plain_chat_uses_bot_persona_and_conversation(temporal_env) -> None:  # noqa: ANN001
    received_chats.clear()
    received_agents.clear()
    result = await _run(
        temporal_env,
        {
            "messages": [
                {"role": "user", "content": "Why is the sky blue?"},
                {"role": "assistant", "content": "Short answer: scattering."},
                {"role": "user", "content": "Tell me more, and be witty about it."},
            ]
        },
        "test-grok-bot-plain",
    )
    assert result["reply"].startswith("Rayleigh scattering")
    assert result["model"] == "mistral-large-latest"
    assert result["usage"]["total_tokens"] == 120
    assert result["connectors_used"] == []
    assert received_agents == []

    assert len(received_chats) == 1
    request = received_chats[0]
    messages = request["messages"]
    assert messages[0]["role"] == "system"
    assert GROK_PERSONA.splitlines()[0] in messages[0]["content"]
    assert "Current date:" in messages[0]["content"]
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "Tell me more, and be witty about it."
    assert request["reasoning_effort"] == "low"
    assert request["max_tokens"] == 2048


async def test_custom_bot_config_is_passed_through(temporal_env) -> None:  # noqa: ANN001
    received_chats.clear()
    result = await _run(
        temporal_env,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "bot": {
                "name": "pirate-bot",
                "system_prompt": "You are a pirate. Arrr.",
                "model": "mistral-small-latest",
                "temperature": 0.2,
                "max_tokens": 64,
                "reasoning_effort": "high",
            },
        },
        "test-grok-bot-custom",
    )
    assert result["model"] == "mistral-small-latest"
    request = received_chats[0]
    assert request["model"] == "mistral-small-latest"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 64
    assert request["reasoning_effort"] == "high"
    assert "You are a pirate. Arrr." in request["messages"][0]["content"]


async def test_connectors_take_the_agent_path(temporal_env) -> None:  # noqa: ANN001
    received_chats.clear()
    received_agents.clear()
    result = await _run(
        temporal_env,
        {
            "messages": [
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "find my mistral-workflows repo on GitHub"},
            ],
            "bot": {"name": "grok-bot", "connectors": ["github_app", "gmail"]},
        },
        "test-grok-bot-connectors",
    )
    assert result["connectors_used"] == ["github_app", "gmail"]
    assert "github.com" in result["reply"]
    assert received_chats == []

    assert len(received_agents) == 1
    call = received_agents[0]
    assert call["connector_names"] == ["github_app", "gmail"]
    assert call["user_message"] == "find my mistral-workflows repo on GitHub"
    assert "user: earlier question" in call["history_text"]
    assert "assistant: earlier answer" in call["history_text"]
    assert GROK_PERSONA.splitlines()[0] in call["system_prompt"]


async def test_unknown_connector_fails_fast(temporal_env) -> None:  # noqa: ANN001
    with pytest.raises(Exception) as excinfo:
        await _run(
            temporal_env,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "bot": {"connectors": ["not_a_connector"]},
            },
            "test-grok-bot-unknown-connector",
        )
    chain = []
    err: BaseException | None = excinfo.value
    while err is not None:
        chain.append(str(err))
        err = err.__cause__ or err.__context__
    assert any("Unknown connectors" in part for part in chain), chain

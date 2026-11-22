"""A Grok-style chatbot served by Mistral's hosted API, with optional connectors.

The ``grok-bot`` workflow takes a conversation plus a bot configuration (persona,
model, sampling, connector names) and returns the bot's reply.

Two execution paths:
- No connectors: a single ``mistralai_chat_complete`` call with the bot's persona
  as system message.
- With connectors: a durable ``Agent``/``Runner`` loop whose connector slots let
  the model call the triggering user's configured connectors (GitHub, Gmail, ...).

The workflow is ``on_behalf_of`` so connector calls resolve the triggering user's
credentials; the managed deployment must be hardened.
"""

from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    Agent,
    AssistantMessage,
    ChatCompletionRequest,
    LocalSession,
    Runner,
    SystemMessage,
    UserMessage,
    mistralai_chat_complete,
)
from mistralai.workflows.plugins.mistralai.connectors import (
    connector,
    uses_connectors,
)

DEFAULT_MODEL = "mistral-large-latest"

#: Connectors the GUI can attach to a bot. Names must match connectors
#: configured in the workspace (see the Connectors guide: never guess them).
CONNECTOR_SLOTS = {
    "github_app": connector("github_app"),
    "gmail": connector("gmail"),
    "google_drive_mcp": connector("google_drive_mcp"),
    "notion": connector("notion"),
    "google_calendar": connector("google_calendar"),
}

GROK_PERSONA = """You are a Grok-style chatbot running on Mistral's hosted API: sharp, witty, and allergic to corporate fluff.

Voice and attitude:
- Direct and honest. Say the thing. Truth over comfort, but never cruel for its own sake.
- Witty and a little irreverent. A joke lands when it serves the answer, not instead of it.
- No filler, no "As an AI language model", no excessive hedging. If you are unsure, say so in one clause and move on.
- Concise by default; expand when the question deserves depth.
- First person, casual, human. Short sentences. Light sarcasm is fine.

Behavior:
- Answer the actual question first, then add context.
- If the question is ambiguous, pick the most likely reading, answer it, and note the alternative in one line.
- Push back when the premise is wrong, and give the reason.
- When connectors are attached, use them when they help answer; say which source you used.
- Never claim to be Grok or to be made by xAI; you are a Mistral-powered alternative inspired by that vibe.
"""


class ChatMessage(BaseModel):
    """One message of the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class BotConfig(BaseModel):
    """A bot definition, sent by the client with every request."""

    name: str = "grok-bot"
    system_prompt: str = GROK_PERSONA
    model: str = DEFAULT_MODEL
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    reasoning_effort: str | None = Field(
        default="low",
        description="Reasoning effort passed to the model; null lets the model default apply. "
        "mistral-large-latest supports low/medium/high.",
    )
    connectors: list[str] = Field(
        default_factory=list,
        description="Connector names to attach (subset of the workflow's declared slots).",
    )


class GrokBotInput(BaseModel):
    """Input of the ``grok-bot`` workflow."""

    messages: list[ChatMessage] = Field(
        min_length=1,
        description="Conversation so far, oldest first; the last message is usually the user's prompt.",
    )
    bot: BotConfig = Field(default_factory=BotConfig)


class GrokBotOutput(BaseModel):
    """Output of the ``grok-bot`` workflow."""

    reply: str
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)
    connectors_used: list[str] = Field(default_factory=list)


def _to_request_message(message: ChatMessage) -> Any:
    if message.role == "user":
        return UserMessage(content=message.content)
    if message.role == "assistant":
        return AssistantMessage(content=message.content)
    return SystemMessage(content=message.content)


@workflows.activity(name="grok_chat", start_to_close_timeout=timedelta(seconds=180))
async def grok_chat(request: ChatCompletionRequest) -> GrokBotOutput:
    """Call the hosted Mistral chat API and normalize the response."""
    response = await mistralai_chat_complete(request)
    content = response.choices[0].message.content
    if isinstance(content, str):
        reply = content
    else:
        reply = "".join(getattr(chunk, "text", "") for chunk in content)
    usage = response.usage.model_dump() if response.usage is not None else {}
    return GrokBotOutput(reply=reply, model=response.model, usage=usage)


class AgentChatParams(BaseModel):
    """Single-argument payload for the agent activity (activities take one param)."""

    system_prompt: str
    history_text: str
    user_message: str
    model: str
    temperature: float
    max_tokens: int
    reasoning_effort: str | None
    connector_names: list[str]


@workflows.activity(
    name="grok_agent_chat", start_to_close_timeout=timedelta(seconds=600)
)
async def grok_agent_chat(params: AgentChatParams) -> GrokBotOutput:
    """Run a durable agent loop with connector tools attached."""
    from mistralai.client.models import TextChunk

    slots = [CONNECTOR_SLOTS[name] for name in params.connector_names]
    instructions = (
        f"{params.system_prompt}\n\nConversation so far:\n{params.history_text}\n\n"
        "Respond to the last user message. Use the attached connector tools when they help."
    )
    completion_args: dict[str, Any] = {
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
    }
    if params.reasoning_effort is not None:
        completion_args["reasoning_effort"] = params.reasoning_effort
    agent = Agent(
        name="grok-bot",
        model=params.model,
        instructions=instructions,
        connectors=slots,
        completion_args=completion_args,
    )
    outputs = await Runner.run(
        agent=agent, inputs=params.user_message, session=LocalSession(), max_turns=8
    )
    reply = "\n".join(
        output.text for output in outputs if isinstance(output, TextChunk)
    )
    return GrokBotOutput(
        reply=reply, model=params.model, connectors_used=params.connector_names
    )


@workflows.workflow.define(
    name="grok-bot",
    workflow_display_name="Grok Bot (Mistral-powered)",
    workflow_description=(
        "A Grok-style chatbot on Mistral's hosted API. Input is a conversation plus a bot "
        "config (persona, model, sampling, connectors); output is the bot's reply plus model "
        "and token usage. Runs on behalf of the triggering user so connectors resolve their "
        "credentials."
    ),
    on_behalf_of=True,
)
@uses_connectors(*CONNECTOR_SLOTS.values())
class GrokBotWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: GrokBotInput) -> GrokBotOutput:
        now = workflows.workflow.now()
        bot = input.bot
        unknown = [name for name in bot.connectors if name not in CONNECTOR_SLOTS]
        if unknown:
            raise workflows.WorkflowError(
                f"Unknown connectors: {unknown}. Known: {sorted(CONNECTOR_SLOTS)}",
                type="invalid_request_error",
                non_retryable=True,
            )

        if not bot.connectors:
            request = ChatCompletionRequest(
                model=bot.model,
                messages=[
                    SystemMessage(
                        content=f"{bot.system_prompt}\nCurrent date: {now:%Y-%m-%d}."
                    ),
                    *(_to_request_message(message) for message in input.messages),
                ],
                temperature=bot.temperature,
                max_tokens=bot.max_tokens,
                # Keep reasoning light by default so small max_tokens budgets are not
                # burned entirely on a reasoning trace with an empty reply.
                reasoning_effort=bot.reasoning_effort,
            )
            return await grok_chat(request)

        history_text = "\n".join(f"{m.role}: {m.content}" for m in input.messages[:-1])
        last = input.messages[-1]
        return await grok_agent_chat(
            AgentChatParams(
                system_prompt=f"{bot.system_prompt}\nCurrent date: {now:%Y-%m-%d}.",
                history_text=history_text or "(no earlier messages)",
                user_message=last.content,
                model=bot.model,
                temperature=bot.temperature,
                max_tokens=bot.max_tokens,
                reasoning_effort=bot.reasoning_effort,
                connector_names=bot.connectors,
            )
        )

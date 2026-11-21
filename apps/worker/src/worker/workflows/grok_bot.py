"""A Grok-style chatbot served by Mistral's hosted chat API.

The ``grok-bot`` workflow takes a conversation and returns the bot's reply. The
personality lives in :data:`GROK_PERSONA`; the model call runs inside the
``grok_chat`` activity through the Mistral AI plugin's ``mistralai_chat_complete``,
which talks to the hosted ``/v1/chat/completions`` API with the worker's identity.
"""

from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    SystemMessage,
    UserMessage,
    mistralai_chat_complete,
)

DEFAULT_MODEL = "mistral-large-latest"

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
- Never claim to be Grok or to be made by xAI; you are a Mistral-powered alternative inspired by that vibe.
"""


class ChatMessage(BaseModel):
    """One message of the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class GrokBotInput(BaseModel):
    """Input of the ``grok-bot`` workflow."""

    messages: list[ChatMessage] = Field(
        min_length=1,
        description="Conversation so far, oldest first; the last message is usually the user's prompt.",
    )
    model: str = Field(
        default=DEFAULT_MODEL, description="Hosted Mistral model to chat with."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)


class GrokBotOutput(BaseModel):
    """Output of the ``grok-bot`` workflow."""

    reply: str
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)


def _to_request_message(message: ChatMessage) -> ChatCompletionRequestMessage:
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


@workflows.workflow.define(
    name="grok-bot",
    workflow_display_name="Grok Bot (Mistral-powered)",
    workflow_description=(
        "A Grok-style chatbot on Mistral's hosted API: witty, direct, truth-seeking replies. "
        "Input is a conversation; output is the bot's reply plus model and token usage."
    ),
)
class GrokBotWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: GrokBotInput) -> GrokBotOutput:
        now = workflows.workflow.now()
        request = ChatCompletionRequest(
            model=input.model,
            messages=[
                SystemMessage(content=f"{GROK_PERSONA}\nCurrent date: {now:%Y-%m-%d}."),
                *(_to_request_message(message) for message in input.messages),
            ],
            temperature=input.temperature,
            max_tokens=input.max_tokens,
            # A chatbot wants a snappy answer, not a reasoning trace: with small
            # max_tokens the model can burn the whole budget on reasoning and
            # return an empty reply.
            reasoning_effort="none",
        )
        return await grok_chat(request)

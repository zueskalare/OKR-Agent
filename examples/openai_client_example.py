#!/usr/bin/env python
"""
OpenAI Client Examples for AtomAgents

Demonstrates the standard OpenAI client library against the AtomAgents
backend. The PRIMARY surface is the Responses API (client.responses.create,
POST /v1/responses); /v1/chat/completions remains supported as a legacy
surface (one example kept at the bottom).

Usage:
    1. Start the backend: ./venv/bin/python -m backend.openai_backend --port 8002
    2. Run: python examples/openai_client_example.py
"""

from openai import OpenAI
import base64
from PIL import Image
import io

# Configure the client to use the AtomAgents backend
client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="dummy",  # Not actually used but required
)

# The session_dir extension keys conversation continuity: the backend keeps
# graph state per session, so requests only carry the NEWEST user message.
SESSION = {"session_dir": "session-example-responses"}


def example_text_response():
    """Example: simple text request via the Responses API."""
    print("🔤 Text Response Example")
    print("-" * 50)

    try:
        response = client.responses.create(
            model="atomagent",
            input="Hello! Can you help me with molecular dynamics simulations?",
            extra_body=SESSION,
        )
        print("User: Hello! Can you help me with molecular dynamics simulations?")
        print(f"Assistant ({response.status}): {response.output_text}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def example_streaming_response():
    """Example: streaming — semantic events instead of chat chunks."""
    print("🌊 Streaming Response Example")
    print("-" * 50)

    try:
        stream = client.responses.create(
            model="atomagent",
            input="Explain the basics of LAMMPS molecular dynamics software.",
            stream=True,
            extra_body=SESSION,
        )
        print("User: Explain the basics of LAMMPS molecular dynamics software.")
        print("Assistant: ", end="", flush=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type in ("response.completed", "response.incomplete"):
                print(f"\n[terminal: {event.type}]")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def example_agent_metadata():
    """Example: read atom_* agent metadata off output items.

    Each output item is one agent message; the AtomAgents extension
    `metadata` (atom_agent/atom_stage/atom_type) survives SDK parsing via
    item.model_extra — something the legacy chat surface never exposed
    through typed SDK models.
    """
    print("🏷️ Agent Metadata Example")
    print("-" * 50)

    try:
        response = client.responses.create(
            model="atomagent",
            input="hi",
            extra_body={"session_dir": "session-example-metadata"},
        )
        for item in response.output:
            md = (item.model_extra or {}).get("metadata", {})
            text = item.content[0].text if item.content else ""
            print(f"[{md.get('atom_agent')} / {md.get('atom_stage')}] {text[:80]}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def example_multimodal_response():
    """Example: image input (input_image part with a bare data-URL)."""
    print("🖼️ Multimodal Response Example")
    print("-" * 50)

    try:
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = client.responses.create(
            model="atomagent",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What color is this image?"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{img_base64}",
                        },
                    ],
                }
            ],
            extra_body=SESSION,
        )
        print("User: What color is this image? [Red 100x100 square]")
        print(f"Assistant: {response.output_text[:200]}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def example_session_continuation():
    """Example: multi-turn continuity via session_dir (NOT message replay).

    The backend rejects previous_response_id — continuity lives in the
    session's checkpointed graph state, so each turn sends only the new
    message with the same session_dir.
    """
    print("💭 Session Continuation Example")
    print("-" * 50)

    try:
        r1 = client.responses.create(
            model="atomagent",
            input="What is molecular dynamics?",
            extra_body={"session_dir": "session-example-continuity"},
        )
        print("User: What is molecular dynamics?")
        print(f"Assistant: {r1.output_text[:100]}...")
        r2 = client.responses.create(
            model="atomagent",
            input="Can you give me a simple example?",
            extra_body={"session_dir": "session-example-continuity"},
        )
        print("User: Can you give me a simple example?")
        print(f"Assistant: {r2.output_text[:100]}...")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def example_list_models():
    """Example: list available models."""
    print("📋 List Models Example")
    print("-" * 50)

    try:
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            print(f"  - {model.id} (owned by: {model.owned_by})")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def example_legacy_chat_completions():
    """Example: the legacy /v1/chat/completions surface (still supported)."""
    print("♻️ Legacy Chat Completions Example")
    print("-" * 50)

    try:
        response = client.chat.completions.create(
            model="atomagent",
            messages=[{"role": "user", "content": "One sentence: what is LAMMPS?"}],
            extra_body={"session_dir": "session-example-legacy"},
        )
        print(f"Assistant: {response.choices[0].message.content[:200]}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def main():
    """Run all examples."""
    print("🤖 AtomAgents OpenAI Client Examples (Responses API)")
    print("=" * 50)

    try:
        client.models.list()
        print("✅ Connected to AtomAgents OpenAI API")
        print()
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Start the backend: ./venv/bin/python -m backend.openai_backend")
        return

    example_list_models()
    example_text_response()
    example_streaming_response()
    example_agent_metadata()
    example_multimodal_response()
    example_session_continuation()
    example_legacy_chat_completions()

    print("🎉 All examples completed!")


if __name__ == "__main__":
    main()

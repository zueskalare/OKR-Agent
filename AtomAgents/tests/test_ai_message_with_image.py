import pytest

try:
    # langchain_core is preferred in recent versions for message types
    from langchain_core.messages import AIMessage
except ImportError:
    # fall back to older langchain.schema
    from langchain.schema import AIMessage

def test_ai_message_with_image_content():
    # Example: base64-encoded PNG (not a real image here, just a dummy)
    img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAUA"
    content_parts = [
        {"type": "text", "text": "This is an image:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + img_b64
            }
        }
    ]
    msg = AIMessage(content=content_parts)

    # The message content should be a list of dicts (text+image)
    assert isinstance(msg.content, list)
    assert any(part.get("type") == "image_url" for part in msg.content)

    # Optionally: check the first part is text and the second is image_url
    assert msg.content[0]["type"] == "text"
    assert msg.content[1]["type"] == "image_url"
    assert msg.content[1]["image_url"]["url"].startswith("data:image/png")

if __name__ == "__main__":
    pytest.main([__file__])
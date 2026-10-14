"""Regressions from ultrareview pass 2 (interface layer)."""

import base64


def test_latest_user_message_has_no_role_prefix():
    from backend.openai_backend import _extract_latest_user_message

    text, images = _extract_latest_user_message(
        [{"role": "user", "content": "Explain deformation gradient"}]
    )
    assert text == "Explain deformation gradient"  # no "user: " pollution
    assert images == []


def test_latest_user_message_hoists_system_content():
    from backend.openai_backend import _extract_latest_user_message

    text, _ = _extract_latest_user_message(
        [
            {"role": "system", "content": "You are a materials-science tutor."},
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second turn"},
        ]
    )
    assert "materials-science tutor" in text
    assert text.endswith("second turn")
    assert "first turn" not in text  # only the NEWEST user turn
    assert "user:" not in text and "reply" not in text


def test_latest_user_message_extracts_images():
    from backend.openai_backend import _extract_latest_user_message

    b64 = base64.b64encode(b"pngbytes").decode()
    text, images = _extract_latest_user_message(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]
    )
    assert text == "see image"
    assert images == [b64]

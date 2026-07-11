from pathlib import Path

from PIL import Image

from src.schema import ImageAsset
from src.vision.openai_vision import OpenAIVisionCaptioner


class _FakeMessage:
    content = "这是一张头颅CT医学影像。"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self):
        self.payload = None

    def create(self, **kwargs):
        self.payload = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_openai_vision_captioner_sends_text_and_image_url(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    image = ImageAsset("img1", "book.pdf", 5, image_path, kind="page")
    client = _FakeClient()

    captioner = OpenAIVisionCaptioner(
        api_key="key",
        api_base="https://example.test/v1",
        model="qwen-vl-plus",
        client=client,
    )
    caption = captioner.caption(image, prompt="请描述。")

    content = client.chat.completions.payload["messages"][0]["content"]
    assert caption.caption == "这是一张头颅CT医学影像。"
    assert content[0] == {"type": "text", "text": "请描述。"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

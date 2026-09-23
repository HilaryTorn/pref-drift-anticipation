from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# pylint: disable=C0103
class LMStyle(Enum):
    VLLMBase = "VLLMBase"
    VLLMAsync = "VLLMAsync"  # vllm via local server
    SGLangAsync = "SGLangAsync"  # sglang via local server
    OpenAIChat = "OpenAIChat"
    Claude = "Claude"
    Grok = "Grok"
    Gemini = "Gemini"
    GeminiThinking = "GeminiThinking"
    DeepSeek = "DeepSeek"
    Mistral = "Mistral"
    Cohere = "Cohere"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    release_date: datetime | None  # XXX Should we use timezone.utc?
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_repr": self.model_repr,
            "model_style": self.model_style.value,
            "release_date": int(self.release_date.timestamp() * 1000),
            "link": self.link,
        }


LanguageModelList: list[LanguageModel] = [
    LanguageModel(
        "VLLMBase",
        "VLLMBase",
        LMStyle.VLLMBase,
        datetime(2023, 9, 1),
        link="General/[local-path]",
    ),
    LanguageModel(
        "VLLMAsync",
        "VLLMAsync",
        LMStyle.VLLMAsync,
        datetime(2023, 9, 1),
        link="General/[local-path]",
    ),
    LanguageModel(
        "SGLangAsync",
        "SGLangAsync",
        LMStyle.SGLangAsync,
        datetime(2023, 9, 1),
        link="General/[local-path]",
    ),
    LanguageModel(
        "OpenAIChat",
        "OpenAIChat",
        LMStyle.OpenAIChat,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "Claude",
        "Claude",
        LMStyle.Claude,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "Grok",
        "Grok",
        LMStyle.Grok,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "Gemini",
        "Gemini",
        LMStyle.Gemini,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "GeminiThinking",
        "GeminiThinking",
        LMStyle.GeminiThinking,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "DeepSeek",
        "DeepSeek",
        LMStyle.DeepSeek,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "Mistral",
        "Mistral",
        LMStyle.Mistral,
        datetime(2024, 6, 30),
        link="",
    ),
    LanguageModel(
        "Cohere",
        "Cohere",
        LMStyle.Cohere,
        datetime(2024, 6, 30),
        link="",
    ),
]

LanguageModelStore: dict[str, LanguageModel] = {
    lm.model_name: lm for lm in LanguageModelList
}

if __name__ == "__main__":
    print(list(LanguageModelStore.keys()))

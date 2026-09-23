from lcb_runner.lm_styles import LMStyle, LanguageModel

from lcb_runner.lm_styles.oai_runner import OpenAIRunner
from lcb_runner.lm_styles.claude_runner import ClaudeRunner
from lcb_runner.lm_styles.cohere_runner import CohereRunner
from lcb_runner.lm_styles.deepseek_runner import DeepSeekRunner
from lcb_runner.lm_styles.gemini_runner import GeminiRunner
from lcb_runner.lm_styles.grok_runner import GrokRunner
from lcb_runner.lm_styles.mistral_runner import MistralRunner
from lcb_runner.lm_styles.vllm_runner import VLLMRunner
from lcb_runner.lm_styles.vllm_async_runner import VllmAsyncRunner


def build_runner(args, model: LanguageModel):

    if model.model_style == LMStyle.OpenAIChat:
        return OpenAIRunner(args, model)

    if model.model_style == LMStyle.Claude:
        return ClaudeRunner(args, model)

    if model.model_style == LMStyle.Cohere:
        return CohereRunner(args, model)

    if model.model_style == LMStyle.DeepSeek:
        return DeepSeekRunner(args, model)

    if model.model_style in (LMStyle.Gemini, LMStyle.GeminiThinking):
        return GeminiRunner(args, model)

    if model.model_style == LMStyle.Grok:
        return GrokRunner(args, model)

    if model.model_style == LMStyle.Mistral:
        return MistralRunner(args, model)

    if model.model_style == LMStyle.VLLMBase:
        return VLLMRunner(args, model)

    if model.model_style in (LMStyle.VLLMAsync, LMStyle.SGLangAsync):
        return VllmAsyncRunner(args, model)

    raise NotImplementedError(
        f"Runner for language model style {model.model_style} not implemented yet"
    )

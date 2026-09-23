import os
import json
import random
import string
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Dict, Union, Optional
import asyncio
from tqdm.asyncio import tqdm_asyncio

from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:  # vLLM is CUDA-only; keep this module importable on CPU/Mac for the
    LLM = SamplingParams = LoRARequest = None  # LiteLLM/HuggingFace agents. vLLMAgent fails only if instantiated.
import torch  # Import torch to detect GPUs
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor
)
try:
    from litellm import acompletion as litellm_acompletion
except ImportError:  # optional; only LiteLLMAgent needs it. Lets HuggingFace-only
    litellm_acompletion = None  # (CPU/Mac) runs skip the litellm install.


load_dotenv()
# =================== Utils ===================
def retry(times=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times - 1:  # Last attempt
                        raise
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
        return wrapper
    return decorator

class LLMAgent(ABC):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, retry_times: int = 3, accepts_system_message: bool = True):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_outputs = "Sorry, I can not satisfy that request."
        self.retry_times = retry_times
        self.accepts_system_message = accepts_system_message

    @abstractmethod
    def _completions(self, messages) -> str:
        raise NotImplementedError
        
    def _completions_batch(self, messages) -> List[str]:
        raise NotImplementedError
    
    async def _async_completions(self, messages) -> str:
        raise NotImplementedError

    
    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError
    
    async def completions_stream(self, messages: List[Dict]) -> str:
        try:
            response = self._completions_stream(messages)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
    
    def completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = self._completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
        
    def completions_batch(self, messages: List[Dict], **kwargs) -> List[str]:
        try:
            response = self._completions_batch(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
        
    async def async_completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = await self._async_completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
        
class vLLMAgent(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=True, tokenizer_path=None, max_logprobs: int = 20, lora_path: Optional[str] = None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        # Hard cap on top-k logprobs the engine will return; controls the cap
        # used by ``choice_probs``. Default 20 matches the OpenAI cap; for
        # A/B forced choice the top-2 is plenty, but bump higher if you'd
        # rather widen the safety margin.
        self.max_logprobs = max_logprobs

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        additional_kwargs = {}
        if "deepseek" in model.lower():
            additional_kwargs["max_model_len"] = 8192
            additional_kwargs["dtype"] = "float16"
            additional_kwargs["enforce_eager"] = True

        # Native (in-process) vLLM LoRA, mirroring `vllm serve --enable-lora --lora-modules
        # name=path` from docs/serving-vllm-aws.md but without standing up a server. `lora_path`
        # is a *local* directory holding one PEFT adapter checkpoint (adapter_config.json +
        # adapter_model.safetensors) -- e.g. one subfolder/checkpoint pulled out of a
        # `*-phase-1-sft-adapters` Hub repo via `snapshot_download(..., allow_patterns=...)`.
        # `max_lora_rank` is read from the adapter's own config so it isn't silently truncated.
        self.lora_request = None
        if lora_path is not None:
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise ValueError(
                    f"lora_path {lora_path!r} has no adapter_config.json; expected a local "
                    "PEFT adapter checkpoint directory."
                )
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            additional_kwargs["enable_lora"] = True
            additional_kwargs["max_lora_rank"] = adapter_config.get("r", 64)
            additional_kwargs["max_loras"] = 1
            self.lora_request = LoRARequest("adapter", 1, lora_path)

        # Initialize vllm
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
            max_logprobs=max_logprobs,
            **additional_kwargs
        )

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert a list of messages to a single prompt string using the tokenizer's apply_chat_template.
        """

        # Use the tokenizer's apply_chat_template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)

        result_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            result_texts.append(generated_text.strip())

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1) -> List[str]:
        return self._completions(messages_list, batch_size)

    def choice_probs(
        self,
        messages_list: List[List[Dict]],
        choices: List[str] = ['A', 'B'],
        logprobs: Optional[int] = None,
    ) -> List[Optional[float]]:
        """
        Forced-choice via a single logprobs call per message, batched natively
        through ``self.llm.generate`` (one continuous-batched forward pass for
        all prompts). Returns P(choices[0]) per message:

            P_A = sum(exp(lp)) over A-variant tokens
                / (sum(exp(lp)) over A-variants + sum(exp(lp)) over B-variants)

        where variants are matched by stripped, case-insensitive token text
        (so " A", "A", '"A"' etc. all count as A). ``logprobs`` is the number
        of top tokens vLLM returns per position; defaults to
        ``self.max_logprobs`` (set at agent construction; 20 by default). For
        A/B forced choice the answer tokens are virtually always in the top
        few, so this is plenty; bump ``vLLMAgent(..., max_logprobs=...)`` at
        construction if you want a larger safety margin.

        Returns:
            list aligned with ``messages_list``. An entry is ``None`` when
            neither choice appears in the returned top-logprobs.

        Notes:
            * Uses temperature=1.0 so the returned distribution matches the
              temp-1.0 hard-sampling distribution.
            * vLLM raises ``VLLMValidationError`` if ``logprobs`` exceeds the
              engine's ``max_logprobs`` cap (set at engine construction).
        """
        import math
        if logprobs is None:
            logprobs = self.max_logprobs
        assert len(choices) == 2, "choices must be a list of two options"
        a_norm = choices[0].strip().strip('"\'').lower()
        b_norm = choices[1].strip().strip('"\'').lower()

        prompts = [self._messages_to_prompt(m) for m in messages_list]
        sp = SamplingParams(temperature=1.0, max_tokens=1, logprobs=logprobs)
        outputs = self.llm.generate(prompts, sp, lora_request=self.lora_request)

        results: List[Optional[float]] = []
        for out in outputs:
            try:
                first_step = out.outputs[0].logprobs[0]
            except (IndexError, AttributeError, TypeError):
                results.append(None)
                continue
            a_mass = 0.0
            b_mass = 0.0
            for _tid, lp in first_step.items():
                tok = getattr(lp, 'decoded_token', None)
                if tok is None:
                    continue
                norm = tok.strip().strip('"\'').lower()
                if norm == a_norm:
                    a_mass += math.exp(lp.logprob)
                elif norm == b_norm:
                    b_mass += math.exp(lp.logprob)
            denom = a_mass + b_mass
            results.append(a_mass / denom if denom > 0 else None)
        return results

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._messages_to_prompt(messages)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        # Since VLLM does not support asynchronous operations, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


FEW_SHOT_PROMPT = """Which city is the capital of France?

Option A: Paris
Option B: Rome

Answer: A

---

Which planet is known as the Red Planet?

Option A: Mars
Option B: Jupiter

Answer: A

---

Which is the largest mammal on Earth?

Option A: Elephant
Option B: Blue Whale

Answer: B

---

What is the chemical symbol for water?

Option A: H2O
Option B: CO2

Answer: A

---

Which shape has three sides?

Option A: Triangle
Option B: Square

Answer: A

---

"""

FEW_SHOT_PROMPT = ""

class vLLMAgentBaseModel(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=False, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        
        # Initialize vllm
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count()  # Use all available GPUs
        )

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        """
        Format messages into strings that the model can process.
        We prepend a hard-coded 5-shot prompt and then append the user's single message.
        """
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            # We expect only one user message, but we'll handle any number just in case
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            # Prepend the 5-shot prompt, then the user's message
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = self._format_messages(messages_list)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        result_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            result_texts.append(generated_text.strip())

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1) -> List[str]:
        return self._completions(messages_list, batch_size)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._format_messages([messages])[0]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        # Since VLLM does not support asynchronous operations, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


class HuggingFaceAgentLogitsPrediction(LLMAgent):

    def __init__(
        self,
        model="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=2048,
        temperature=0.0,
        cache_dir='/data/public_models',
        trust_remote_code=False,
        accepts_system_message=False
    ):
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.accepts_system_message = False  # Hard-coded for base models with no system messages

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        # Set padding token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"  # Automatically distribute across GPUs
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
            ).to(self.device)

        self.llm.eval()  # Set to evaluation mode

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        """
        Format messages into strings that the model can process.
        We prepend a hard-coded 5-shot prompt and then append the user's single message.
        """
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            # We expect only one user message, but we'll handle any number just in case
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            # Prepend the 5-shot prompt, then the user's message
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(
        self,
        messages: Union[List[Dict], List[List[Dict]]],
        batch_size: int = 1,
        options: List[str] = ['A', 'B']
    ) -> List[Dict[str, float]]:
        """
        Get completion logits for specific options, returning a probability distribution
        over those options that sums to 1.0.
        """
        formatted_messages = self._format_messages(messages)
        results = []

        # Convert option strings (e.g. "A", "B") to the correct token IDs for " A", " B", etc.
        option_tokens = [" " + opt for opt in options]
        option_ids = self.tokenizer.encode(option_tokens, add_special_tokens=False)

        # Process in batches
        for i in range(0, len(formatted_messages), batch_size):
            batch_messages = formatted_messages[i:i + batch_size]

            # Tokenize inputs
            inputs = self.tokenizer(
                batch_messages,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=False
            ).to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.llm(**inputs)
                # We only need the last token's logits for each sequence
                logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]

            # Process each sequence in the batch
            for logits_seq in logits:
                # Create a masked logits array that is -1e4 everywhere except
                # for the chosen option token IDs.  -1e4 is safely in range for float16.
                masked_logits = torch.full_like(logits_seq, -1e4)
                for option_id in option_ids:
                    if option_id < logits_seq.shape[0]:
                        masked_logits[option_id] = logits_seq[option_id]

                # Compute the softmax distribution over the entire vocabulary,
                # then isolate just the options and renormalize so they sum to 1.
                full_dist = F.softmax(masked_logits, dim=0)
                subset_dist = full_dist[option_ids]
                subset_dist = subset_dist / subset_dist.sum()  # Force sum to 1

                # Build a dictionary of {option_letter: probability}
                distribution_dict = {
                    options[idx]: float(subset_dist[idx]) for idx in range(len(options))
                }
                results.append(distribution_dict)

        return results


class HuggingFaceAgent(LLMAgent):
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, batch_size=512, accepts_system_message=True, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model_name = model
        self.batch_size = batch_size
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            padding_side='left'  # Important: Set padding to left side
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        adapter_config_path = os.path.join(model, "adapter_config.json") if isinstance(model, str) else ""
        if adapter_config_path and os.path.exists(adapter_config_path):
            if tokenizer_path is None:
                raise ValueError(
                    f"{model} looks like a PEFT adapter checkpoint, but tokenizer_path/base model "
                    "was not provided in config.yaml."
                )
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                tokenizer_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        self.model.eval()
        
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a prompt using the model's chat template."""
        output = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return output
    
    def _completions(self, messages: List[Dict]) -> str:
        """Handle single completion."""
        return self._completions_batch([messages])[0]
    
    def _completions_batch(self, messages_list: List[List[Dict]], **kwargs) -> List[str]:
        """Handle batch completions with left padding."""
        from accelerate.utils import find_executable_batch_size
        from tqdm import tqdm
        
        # Format all messages into prompts using chat template
        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        all_outputs = []
        
        @find_executable_batch_size(starting_batch_size=self.batch_size)
        def _process_batch(batch_size):
            nonlocal all_outputs
            print(f"\nProcessing with batch size: {batch_size}", flush=True)
            
            # Process in batches
            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize with padding
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2048  # Adjust based on model context window
                ).to(self.model.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else 1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode outputs
                for j, output in enumerate(outputs):
                    # Find where the prompt ends
                    prompt_length = len(inputs["input_ids"][j])
                    # Only decode the new tokens
                    decoded = self.tokenizer.decode(
                        output[prompt_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    all_outputs.append(decoded.strip())
        
        # Find and use the largest working batch size
        _process_batch()
        return all_outputs
    
    async def _async_completions(self, messages: List[Dict]) -> str:
        """Async completion just calls sync version since HF doesn't have async API."""
        return self._completions(messages)
    
    async def _completions_stream(self, messages: List[Dict]) -> str:
        """Streaming not implemented for HF models."""
        raise NotImplementedError("Streaming not implemented for HuggingFace models")


class LiteLLMAgent:
    # Class-level default so `_reasoning_extra_body` can read `self.enable_thinking`
    # even when neither the constructor nor a subclass sets it. None = do not send the
    # chat_template_kwargs.enable_thinking key (leave the server template default).
    enable_thinking = None

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        accepts_system_message: bool = True,
        max_retries: int = 5,
        base_timeout: float = 5.0,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        use_jitter: bool = True,
        reasoning_max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ):
        self.model = model
        # OpenAI-compatible endpoint URL (e.g. a self-hosted vLLM server). None means
        # LiteLLM routes to the provider's default endpoint from the model prefix.
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.accepts_system_message = accepts_system_message

        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_jitter = use_jitter
        # Cap on native reasoning/thinking tokens so a reasoning model commits an
        # answer inside `max_tokens` instead of exhausting the budget mid-thought
        # (finish_reason='length' -> content=None -> dropped elicitation). None = no
        # cap. Sent as vLLM's top-level `thinking_token_budget`: the served endpoint
        # must run a vLLM version with the ThinkingTokenBudgetLogitsProcessor (after
        # PRs #20859/#37112) and `--reasoning-parser qwen3`, or the field is silently
        # ignored. See docs/serving-vllm-aws.md. Logprobs forced-choice
        # (async_choice_probs, max_tokens=1) never reasons, so it is applied only on
        # the sampled-generation path here.
        self.reasoning_max_tokens = reasoning_max_tokens
        # Pins Qwen's native thinking/non-thinking mode via the served chat template
        # instead of trusting its per-size default (which varies -- see
        # rl_training/prompts.py). Sent as `chat_template_kwargs.enable_thinking` in the
        # request body (the exact switch the RL runner uses). None = do not send the key,
        # leaving the server default in place; True/False force the mode. Requires the
        # endpoint to run `--reasoning-parser qwen3`. Applied on BOTH the sampled path and
        # the logprobs forced-choice path -- on the latter, enable_thinking=False keeps the
        # scored first token the answer label rather than an opening `<think>`.
        #
        # Guarded (not an unconditional assignment) so a subclass that sets
        # self.enable_thinking BEFORE calling super().__init__() -- e.g. the RL runner's
        # RLNativeThinkingEndpointAgent -- is not clobbered back to the None default. The
        # class-level default below keeps the attribute defined when nobody passes it.
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking

    def _reasoning_extra_body(self) -> Optional[Dict]:
        """Request-body fragment for native reasoning control, or None when nothing to send.

        Two independent levers, both no-ops unless the server runs `--reasoning-parser qwen3`:

        * `thinking_token_budget` (TOP-LEVEL, vLLM's native mechanism,
          https://docs.vllm.ai/en/latest/features/reasoning_outputs/): when the reasoning-token
          count reaches it, vLLM forces the reasoning-end string (`</think>` for Qwen3) so the model
          must produce its answer. NOT `chat_template_kwargs.thinking_budget` -- that is a different
          provider's convention (Featherless), not vLLM's.
        * `chat_template_kwargs.enable_thinking` -- pins thinking on/off in the chat template.

        Returns None when neither is configured, so callers that pass this through unchanged keep
        their prior no-extra-body behavior."""
        body: Dict = {}
        if self.reasoning_max_tokens:
            body["thinking_token_budget"] = self.reasoning_max_tokens
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return body or None

    async def async_completions(
        self,
        messages: List[List[Dict]],
        verbose: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Returns a list of LLM responses, in order.
        Uses a semaphore to limit concurrency, and tqdm_asyncio for progress.
        """

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            """
            Attempts to process a single message up to `max_retries` times.
            On generic exceptions, sleeps with exponential backoff and optional jitter.
            On timeout, doubles the request timeout without sleeping.
            """
            message = messages[message_idx]

            current_timeout = self.base_timeout
            retry_delay = self.base_delay
            response = None

            for attempt in range(self.max_retries):
                # Acquire the semaphore before making the LLM call
                async with semaphore:
                    # if verbose:
                    #     print(
                    #         f"[Attempt {attempt+1}/{self.max_retries}] "
                    #         f"Message index {message_idx}, timeout={current_timeout:.1f}s"
                    #     )

                    try:
                        completion_res = await litellm_acompletion(
                            model=self.model,
                            messages=message,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            timeout=current_timeout,
                            api_base=self.base_url,
                            # LiteLLM's hosted_vllm route splats extra_body as a mapping, so passing
                            # None raises "'NoneType' object is not a mapping" -- every attempt fails,
                            # retries exhaust, and the row comes back an EMPTY STRING with the row
                            # count intact. That reads as the model refusing to answer rather than as
                            # plumbing, which is the same misdiagnosis serving-vllm-aws.md:305 records.
                            # _reasoning_extra_body() returns None whenever neither enable_thinking nor
                            # reasoning_max_tokens is set, so omit the key entirely in that case.
                            **({"extra_body": _xb} if (_xb := self._reasoning_extra_body()) else {}),
                        )
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1

                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}. Timed out after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None  # no more retries
                            if verbose:
                                print(f"Max retries (timeouts) reached for message index {message_idx}.")
                        else:
                            current_timeout *= 2.0

                        continue  # next attempt

                    except Exception as e:
                        counts["errors"] += 1

                        if verbose:
                            print(
                                f"[Error] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                            if verbose:
                                print(f"Max retries (errors) reached for message index {message_idx}.")
                        else:
                            # Sleep with exponential backoff
                            sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            if verbose:
                                print(f"Sleeping {sleep_for:.1f}s before retry (error backoff)...")
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)

                        continue  # next attempt

                    # Success: parse the response. content can be None when a
                    # reasoning model exhausts max_tokens on reasoning before emitting
                    # an answer (finish_reason='length') or on a refusal — guard the
                    # .strip() so one such response doesn't crash the whole run.
                    msg = completion_res.choices[0].message
                    content = msg.content
                    # With `--reasoning-parser qwen3`, vLLM splits the native <think> trace into a
                    # separate reasoning field and leaves `content` as just the post-</think> answer.
                    # (LiteLLM exposes it as reasoning_content; vLLM's raw field is reasoning.) Rejoin
                    # them so raw_response keeps the FULL trace -- needed to measure think length and
                    # whether </think> closed, and to see whether a thinking-budget cut is biting -- while
                    # the parser still reads the "Answer: X" off the last line. When the model exhausts
                    # its budget mid-think, content is empty and the answer (if any) sits at the end of
                    # the reasoning field, so fall back to it alone. No reasoning field (parser off) ->
                    # content already holds the inline <think>..</think>, so this is a no-op.
                    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                    content = content.strip() if content else None
                    reasoning = reasoning.strip() if reasoning else None
                    if reasoning and content:
                        response = f"<think>{reasoning}</think>\n{content}"
                    elif content:
                        response = content
                    else:
                        response = reasoning  # may be None
                    break  # done with retries

            results[message_idx] = response

        # Create a task for each message
        tasks = [process_message(i) for i in range(len(messages))]

        # Use tqdm_asyncio to track progress as tasks finish
        # You can also set `leave=False` or other tqdm arguments as needed
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="LLM calls"):
            await coro

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")
            print(f"Number of generic errors: {counts['errors']}")

        return [results[i] for i in range(len(messages))]

    async def async_choice_probs(
        self,
        messages: List[List[Dict]],
        choices: List[str] = ['A', 'B'],
        verbose: bool = True,
    ) -> List[Optional[float]]:
        """
        Forced-choice via a single logprobs call per message (HTTP/OpenAI-
        compatible path). Mirrors ``async_completions``'s concurrency and
        retry structure but requests ``max_tokens=1, temperature=1.0,
        logprobs=True, top_logprobs=20`` (top_logprobs is capped at 20 by the
        OpenAI spec). Returns P(choices[0]) per message, or ``None`` when
        neither choice appears in the returned top-logprobs.

        For local models where you control the inference server, prefer
        ``vLLMAgent.choice_probs`` -- it is faster (no HTTP overhead, single
        batched forward pass) and supports a much larger top-k.
        """
        import math
        assert len(choices) == 2, "choices must be a list of two options"
        a_norm = choices[0].strip().strip('"\'').lower()
        b_norm = choices[1].strip().strip('"\'').lower()

        def _compute_p_a(completion_res) -> Optional[float]:
            try:
                choice0 = completion_res.choices[0]
                lp = getattr(choice0, 'logprobs', None) or (
                    choice0.get('logprobs') if isinstance(choice0, dict) else None
                )
                content = getattr(lp, 'content', None) if lp is not None else None
                if content is None and isinstance(lp, dict):
                    content = lp.get('content')
                first = content[0] if content else None
                top = getattr(first, 'top_logprobs', None) if first is not None else None
                if top is None and isinstance(first, dict):
                    top = first.get('top_logprobs')
            except (AttributeError, IndexError, TypeError):
                return None
            if not top:
                return None
            a_mass = 0.0
            b_mass = 0.0
            for entry in top:
                if isinstance(entry, dict):
                    tok = entry.get('token'); lgp = entry.get('logprob')
                else:
                    tok = getattr(entry, 'token', None); lgp = getattr(entry, 'logprob', None)
                if tok is None or lgp is None:
                    continue
                norm = tok.strip().strip('"\'').lower()
                if norm == a_norm:
                    a_mass += math.exp(lgp)
                elif norm == b_norm:
                    b_mass += math.exp(lgp)
            denom = a_mass + b_mass
            return a_mass / denom if denom > 0 else None

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        results: Dict[int, Optional[float]] = {}

        async def process_message(idx: int):
            current_timeout = self.base_timeout
            retry_delay = self.base_delay
            for attempt in range(self.max_retries):
                async with semaphore:
                    try:
                        completion_res = await litellm_acompletion(
                            model=self.model,
                            messages=messages[idx],
                            max_tokens=1,
                            temperature=1.0,
                            logprobs=True,
                            top_logprobs=20,
                            timeout=current_timeout,
                            api_base=self.base_url,
                            # Forced-choice scores the FIRST generated token. With
                            # enable_thinking unpinned and a thinking-default template, that
                            # token is an opening `<think>`, not the answer label -- so pass
                            # enable_thinking=False here to keep the scored token the answer.
                            # thinking_token_budget is a harmless no-op at max_tokens=1.
                            # Omitted entirely when None -- see the note on the sampled path above.
                            **({"extra_body": _xb} if (_xb := self._reasoning_extra_body()) else {}),
                        )
                        results[idx] = _compute_p_a(completion_res)
                        return
                    except asyncio.TimeoutError:
                        current_timeout = min(current_timeout * 2.0, 60.0)
                    except Exception as exc:
                        if verbose:
                            print(
                                f"[Error] async_choice_probs attempt "
                                f"{attempt+1}/{self.max_retries} msg {idx}: {exc}"
                            )
                        await asyncio.sleep(min(retry_delay, self.max_delay))
                        retry_delay *= 2
            results[idx] = None

        tasks = [process_message(i) for i in range(len(messages))]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks),
                                              desc="logprob calls"):
            await coro
        return [results[i] for i in range(len(messages))]


class VLLMEndpointAgent(LiteLLMAgent):
    """Our own self-hosted vLLM OpenAI-compatible endpoint (e.g. on AWS).

    Reuses LiteLLMAgent wholesale (concurrency, retry/backoff, timeouts, sampling
    and forced-choice paths); it only fixes the LiteLLM provider route. LiteLLM's
    native ``hosted_vllm`` provider speaks the OpenAI wire protocol to ``base_url``,
    so ``served_model`` is just the name passed to ``vllm serve --served-model-name``
    (the ``hosted_vllm/`` prefix is an implementation detail hidden here).
    """
    def __init__(self, served_model: str, base_url: str, **kwargs):
        super().__init__(model=f"hosted_vllm/{served_model}", base_url=base_url, **kwargs)

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    pass


from lcb_runner.lm_styles.base_runner import BaseRunner
from lcb_runner.runner.parser import ConfigLCB
from typing import List, Dict


class VLLMRunner(BaseRunner):
    def __init__(self, args: ConfigLCB, model):
        super().__init__(args, model)

        self.args = args

        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )

        self.llm = LLM(
            model=model_tokenizer_path,
            tokenizer=model_tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_seq_length,
            dtype=args.dtype,
            # enforce_eager=True,
            disable_custom_all_reduce=True,
            enable_prefix_caching=args.enable_prefix_caching,
            trust_remote_code=args.trust_remote_code,
            max_num_seqs=args.batch_size,  # Limit batch size
            enable_reasoning=args.cot_code_execution,
        )

        self.sampling_params = SamplingParams(
            n=args.n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=args.stop,
        )

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: List[List[Dict[str, str]]]) -> list[list[str]]:

        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            hprompt = str(prompt)  # make prompt hashable
            if self.args.use_cache and hprompt in self.cache:
                if len(self.cache[hprompt]) == self.args.n:
                    outputs[prompt_index] = self.cache[hprompt]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)

        if remaining_prompts:

            vllm_outputs = self.llm.chat(remaining_prompts, self.sampling_params)

            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                    remaining_indices, remaining_prompts, vllm_outputs
                ):
                    hprompt = str(remaining_prompt)  # make prompt hashable
                    self.cache[hprompt] = [o.text for o in vllm_output.outputs]
                    outputs[index] = [o.text for o in vllm_output.outputs]
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
        return outputs

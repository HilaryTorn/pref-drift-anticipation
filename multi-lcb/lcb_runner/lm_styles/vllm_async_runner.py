from tqdm.auto import tqdm

try:
    from openai import AsyncOpenAI, BadRequestError, APIConnectionError, APITimeoutError
except ImportError as e:
    pass

from httpx import ReadTimeout

from copy import deepcopy
from lcb_runner.lm_styles.base_runner import BaseRunner
from lcb_runner.runner.parser import ConfigLCB
from typing import List, Dict
import asyncio
import re
import os


class VllmAsyncRunner(BaseRunner):
    """Compatible both with Sglang and VLLM."""

    base_url = "http://localhost:5000/v1"
    api_key = "NO_KEY"

    if os.environ["OPENAI_BASE_URL"]:
        base_url = os.environ["OPENAI_BASE_URL"]

    if os.environ["OPENAI_KEY"]:
        api_key = os.environ["OPENAI_KEY"]

    client = None

    def __init__(self, args: ConfigLCB, model):
        super().__init__(args, model)

        print(f"Using server url: {self.base_url}")

        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=args.gen_timeout
        )

        self.queue_depth: int = args.batch_size
        model_name = args.local_model_path

        self.client_kwargs: dict[str | str] = {
            "model": model_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": 1,
            # add reasoning
            "extra_body": {"chat_template_kwargs": args.chat_template_kwargs},
        }

        # check that model_name is correct
        lst_sever_models = self.get_server_models()
        assert model_name in lst_sever_models, ValueError(
            f"Wrong model name for VLLM/SGLang server. Available models: {lst_sever_models}"
        )

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        raise NotImplementedError

    @staticmethod
    async def _get_models_list(client: AsyncOpenAI) -> List[str]:
        out = await client.models.list()
        models = out.data
        return [x.id for x in models]

    def get_server_models(self):
        """List of available models."""

        try:
            coro = self._get_models_list(self.client)
            availab_models = asyncio.run(coro)
        except APIConnectionError as e:
            e.message = "can't connect to the server"
            raise e

        return availab_models

    async def async_batch(self, prompts: List[List[Dict]]) -> List[List[str]]:

        async def _fetch_answer(chat: List[Dict]) -> str:
            # Worker
            client_kwargs = deepcopy(self.client_kwargs)

            cnt_reruns = 0
            max_reruns = 3
            restart = True
            while restart:
                cnt_reruns += 1
                restart = False
                try:
                    response = await self.client.chat.completions.create(
                        messages=chat,
                        **client_kwargs,
                    )
                except BadRequestError as e:
                    if (
                        "Requested token count exceeds the model's maximum context length"
                        in e.message
                    ):
                        # restart request with smaller 'max_tokens' client config, common issue with long answers from reasoning models
                        r = re.search(
                            "maximum context length of ([0-9]*) tokens.*? ([0-9]*) tokens from the input messages",
                            e.message,
                        )
                        context_len = int(r.group(1))
                        request_len = int(r.group(2))
                        new_max_tokens = max(context_len - request_len - 1, 0)
                        client_kwargs["max_tokens"] = new_max_tokens
                        restart = True if cnt_reruns < max_reruns else False
                        if restart:
                            print(
                                f"Requested 'max_token' count exceeds the model's maximum context length. Reducing 'max_tokens' to {new_max_tokens}. Restarting the request."
                            )
                        else:
                            raise e
                    else:
                        raise e
                except (ReadTimeout, APITimeoutError) as e:
                    restart = True if cnt_reruns < max_reruns else False
                    if restart:
                        print(f"{type(e)} - API Timeout, restarting the request.")
                    else:
                        raise e

            content = response.choices[0].message.content
            return content

        n_prompts = len(prompts)
        queue = []
        tasks = set()
        tasks_ids = dict()
        outputs = [[] for _ in range(len(prompts))]
        flag_run = True

        cur_idx = 0
        pr_bar = tqdm(total=n_prompts * self.args.n)

        while flag_run:

            while cur_idx < n_prompts and len(queue) < self.queue_depth:
                for _ in range(self.args.n):
                    queue.append(cur_idx)
                cur_idx += 1

            while len(queue) > 0 and len(tasks) < self.queue_depth:
                idx = queue.pop(0)
                sample = prompts[idx]
                task = asyncio.create_task(_fetch_answer(sample))
                tasks.add(task)
                tasks_ids[task] = idx

            if len(tasks) == 0:
                flag_run = False
                break

            done, tasks = await asyncio.wait(
                tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                idx = tasks_ids.pop(task)
                if task.cancelled():
                    raise NotImplementedError

                ex = task.exception()
                if not ex:
                    content = task.result()
                    outputs[idx].append(content)
                    pr_bar.update()
                elif isinstance(ex, APIConnectionError):
                    # server is down
                    raise ex
                else:
                    print(
                        f"Task id {idx} finished with exception [{type(ex)}, {ex}]. Skipping the failed request."
                    )

        pr_bar.close()
        return outputs

    def run_batch(self, prompts: List[List[Dict[str, str]]]) -> List[List[str]]:

        coro = self.async_batch(prompts)
        outputs = asyncio.run(coro)
        return outputs

import os
from time import sleep

import openai
from openai import OpenAI

from lcb_runner.lm_styles.base_runner import BaseRunner


class GrokRunner(BaseRunner):

    def __init__(self, args, model):
        super().__init__(args, model)

        self.client = OpenAI(
            api_key=os.environ.get("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        model_name = args.model.split("_")[0]
        self.client_kwargs: dict[str | str] = {
            "model": model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "n": 1,
            # "timeout": args.openai_timeout,
            # "stop": args.stop, --> stop is only used for base models currently
        }
        if "_" in args.model:
            self.client_kwargs["reasoning_effort"] = args.model.split("_")[1]

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        def _fetch_response(counter):
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response.choices[0].message.content
                return content
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as e:
                counter = counter - 1
                if counter == 0:
                    raise e
                print("Exception: ", repr(e))
                print(prompt[0]["content"])
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return _fetch_response(counter)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(_fetch_response(10))
        except Exception as e:
            raise e
        return outputs

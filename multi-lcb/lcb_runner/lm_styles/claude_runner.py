import os
from time import sleep

try:
    from anthropic import Anthropic
except ImportError as e:
    pass

from lcb_runner.lm_styles.base_runner import BaseRunner


class ClaudeRunner(BaseRunner):

    def __init__(self, args, model):
        super().__init__(args, model)

        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"), timeout=1200)

        if "Thinking" in model.model_style.value:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_tokens": 32000,
                "thinking": {"type": "enabled", "budget_tokens": 24000},
                "stream": False,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
            }

    def _run_single(self, prompt: tuple[str, str]) -> list[str]:

        def _fetch_response(counter: int) -> str:
            """Fetch response from the server and restarts request if needed.

            Args:
                counter (int): count retries.

            Returns:
                content: msg content
            """
            try:
                response = self.client.messages.create(
                    system=prompt[0],
                    messages=prompt[1],
                    **self.client_kwargs,
                )
                content = "\n".join(
                    [
                        getattr(x, "text", getattr(x, "thinking", "\nREDACTED\n"))
                        for x in response.content
                    ]
                )
                return content
            except Exception as e:
                print("Exception: ", repr(e), "Sleeping for 20 seconds...")
                sleep(20 * (11 - counter))
                counter = counter - 1
                if counter == 0:
                    print(f"Failed to run model for {prompt}!")
                    print("Exception: ", repr(e))
                    raise e
                return _fetch_response(counter)

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(_fetch_response(10))
        except Exception as e:
            raise e

        return outputs

import os
from time import sleep

try:
    from cohere import ClientV2
except ImportError as e:
    pass

from lcb_runner.lm_styles.base_runner import BaseRunner


class CohereRunner(BaseRunner):

    def __init__(self, args, model):
        super().__init__(args, model)

        self.client = ClientV2(os.getenv("COHERE_API_KEY"))

        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "p": args.top_p,
        }

    def _run_single(self, prompt: tuple[dict[str, str], str]) -> list[str]:
        def _fetch_response(counter):
            try:
                response = self.client.chat(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response.message.content[0].text
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

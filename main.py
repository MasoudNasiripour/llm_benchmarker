from typing import Any, List

from llm_benchmarker import FarsiBench, MMLUBench
from llm_benchmarker import BenchManager


GENERATOR_FUNC_KEY = "gen_func"
CHAT_TEMPLATE_FUNC = "prompt_formatter_func"

def message_format_func_farsibench(system_prompt: str, usr_prompt: List[str], ) -> Any:
    return usr_prompt

def generation_farsibench(messages) -> List[str]:
    return [mes for mes in messages]

def message_format_func_mmlu(system_prompt: str, usr_prompt: List[str], ) -> Any:
    return usr_prompt

def generation_mmlu(messages) -> List[str]:
    return [str(1) for _ in messages]


def main():
    requested_benchmarks = {
        FarsiBench: {
            GENERATOR_FUNC_KEY: generation_farsibench,
            CHAT_TEMPLATE_FUNC: message_format_func_farsibench
        },

        MMLUBench: {
            GENERATOR_FUNC_KEY: generation_mmlu,
            CHAT_TEMPLATE_FUNC: message_format_func_mmlu
        }
    }
    

    runner = BenchManager(requested_benchmarks)
    print(runner.run())


if __name__ == "__main__":
    main()

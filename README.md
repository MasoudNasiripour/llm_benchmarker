# 📦 llm benchmarker

A package for benchmarking LLM models. Validating models with different benchmarks sometimes needs a lot of time.
This package provides different benchmark in one unified system. 

### 🚀 Installation

For installing the package you can go with one of the below methods:
1. Using ```pypi```:
``` properties
pip install llm_benchmarker
```

2. From ```Source```:
clone the repo in your system. then run the below command to install the dependency for the project:

```properties
pip install -r requirements.txt
```

3. Using ```Docker```:
``` properties
docker pull masoudnasiripour/llm_benchmarker
```

### 📖 Usage
For using this package, you need to make an instance of ```BenchManager``` in ```from benchmarker import BenchManager```.
After that you need to define a dictionary. with Benchmark types as keys and another dictionary as value. \
In second dictionary you must provide two function references, one of them is a generaion function and another one is a chat formatted function.
```python
from typing import Any
from benchmarker import FarsiBench, MMLUBench, BenchManager


GENERATOR_FUNC_KEY = "gen_func"
CHAT_TEMPLATE_FUNC = "prompt_formatter_func"

def message_format_func(system_prompt: str, usr_prompt: list[str], ) -> Any:
    return usr_prompt

def generation(messages) -> list[str]:
    return [mes for mes in messages]

def message_format_func_mmlu(system_prompt: str, usr_prompt: list[str], ) -> Any:
    return usr_prompt

def generation_mmlu(messages) -> list[str]:
    return [str(1) for _ in messages]


# How we use these two
formatted = message_format_func("My System prompt",
                    ["use message one", "use message two", "use message three"])


if __name__ == "__main__":

    model_conf_per_bench = {
        FarsiBench : {
            GENERATOR_FUNC_KEY: generation,
            CHAT_TEMPLATE_FUNC: message_format_func
        },
        MMLUBench: {
            GENERATOR_FUNC_KEY: generation_mmlu,
            CHAT_TEMPLATE_FUNC: message_format_func_mmlu
        },

    }


    benchmarks = BenchManager(benchmark_model_conf=model_conf_per_bench)
    print(benchmarks.run())
```
Output:
```properties
{
    'PersianQA': {
        'f1_score': 0.12886794355824202,
        'exact_match': 0.0,
        'bleu': 0.0,
        'precisions': [0.0, 0.0, 0.0, 0.0],
        'brevity_penalty': 1.0,
        'length_ratio': 223.6290322580645,
        'translation_length': 207975,
        'reference_length': 930,
        'rouge1': np.float64(0.0),
        'rouge2': np.float64(0.0),
        'rougeL': np.float64(0.0),
        'rougeLsum': np.float64(0.0)},
    'MMLU': {
        'accuracy': 0.24654607605754167
    }
}
```

### Benchmark Lists
| Benchmark | Path | Metrics
|--- | --- | --- |
| PersianQA | ```llm_benchmarker.evals.multiling.FarsiBench``` | ```bleu```, ```rouge```, ```f1```, ```exact-match``` |
| MMLUBench | ```llm_benchmarker.evals.lang.MMLUBench``` | ```accuracy``` |

### Add Benchmark
To adding benchmark you need to:
1. Define a class that inherited from ```BaseBench``` in ```from llm_benchmarker.evals.base import BaseBench```

2. Implement ```compute()``` function with considering it's return type(```dict```).

3. return a ```string``` value from ```shared_key``` function and set a object variable calle ```benchmark_name``` to it.

4. Implement an slot. You need to go in ```data/readers/_<yourbenchmark-category-name>.py``` and implement a function that can load your dataset locally and decorate it with ```slot``` decorator that take an argument called ```shared_key```. This argument must be the value you returned in step 3.

5. then you must go in ```config.py``` and define you'r benchmark in a dictionary called ```DATASETS_PER_BENCH```.
6. ```(WARNING)``` If you are want to put you slot function in other places you must add it's directory path into ```SLOT_DIR_PATH```. llm benchmarker looks python modules defined in these directories for finding ```slot``` functions. 


### 🤝 Contributing
Pull requests are welcome!\
For major changes, please open an issue first to discuss what you would like to change.
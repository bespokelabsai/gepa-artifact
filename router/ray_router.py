from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

llm_config1 = LLMConfig(
    model_loading_config=dict(
        model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        model_source="Qwen/Qwen3-30B-A3B-Instruct-2507",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=2, max_replicas=2,
        )
    ),
    engine_kwargs={
        "tensor_parallel_size": 2,
    },
    accelerator_type="A100",
)

app = build_openai_app({"llm_configs": [llm_config1]})
serve.run(app, blocking=True)
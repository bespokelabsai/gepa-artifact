from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

llm_config1 = LLMConfig(
    model_loading_config=dict(
        model_id="Qwen/Qwen3-8B",
        model_source="Qwen/Qwen3-8B",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=4, max_replicas=4,
        )
    ),
    accelerator_type="A100",
)

app = build_openai_app({"llm_configs": [llm_config1]})
serve.run(app, blocking=True)
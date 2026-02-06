from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


llm_config1 = LLMConfig(
    model_loading_config=dict(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_source="Qwen/Qwen3-4B-Instruct-2507",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=2, max_replicas=2,
        )
    ),
    accelerator_type="A100",
)

app = build_openai_app({"llm_configs": [llm_config1]})
serve.run(app, blocking=True)
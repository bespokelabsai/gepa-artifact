import pickle
# import dspy

# with open("experiment_runs_data/experiment_runs/seed_0/LiveBenchMathBench_CoT_GEPA_gpt-41-mini/gepa_state.bin", "rb") as f:
with open("router/gepa_checkpoints/run_42/gepa_state.bin", "rb") as f:
    gepa_state = pickle.load(f)

import pdb; pdb.set_trace()
for key, value in gepa_state.items():
    print(key, value)

# with open("experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B/prog_candidates/2/program.pkl", "rb") as f:
#     program = pickle.load(f)
#     import pdb; pdb.set_trace()
#     print(program)
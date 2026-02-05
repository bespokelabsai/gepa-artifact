# Router Training Pipeline

This document describes the complete training pipeline for prompt routers, including both MLP-based and LM-based (Qwen) approaches.

## Overview

The training pipeline consists of three main stages:

1. **Data Preparation**: Generate training data from GEPA experiment results
2. **Model Training**: Train either MLP or Qwen-based routers
3. **Evaluation**: Assess router performance and compare with baselines

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GEPA Experiment Results                   │
│                  (experiment_runs_data/)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preparation (Step 1)                       │
│          construct_training_data.py                          │
│                                                              │
│  Input:  gepa_state.bin, prog_candidates/                   │
│  Output: training_data.json                                  │
│                                                              │
│  Format: One example per task with all candidates           │
│  {                                                           │
│    "task_idx": 0,                                           │
│    "claim": "...",  # or question, prompt, problem, etc.    │
│    "candidates": [                                          │
│      {                                                       │
│        "candidate_idx": 0,                                  │
│        "candidate_system_prompt": "...",                    │
│        "reward": 1.0,                                       │
│        "in_pareto_frontier": true                           │
│      }, ...                                                  │
│    ]                                                         │
│  }                                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  MLP Training   │         │  Qwen SFT       │
│    (Step 2a)    │         │   (Step 2b)     │
└─────────────────┘         └─────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  MLP Checkpoint │         │ Compiled DSPy   │
│   best_*.pt     │         │  Predictor      │
└─────────────────┘         └─────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation (Step 3)                         │
│               evaluate_router.py                             │
│                                                              │
│  - Compare against baselines (Random, Oracle)                │
│  - Compute top-k accuracy, regret, Pareto selection rate    │
│  - Generate comparison reports                               │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Data Preparation

### Generate Training Data

```bash
python construct_training_data.py \
    --experiment_dir ../experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B \
    --output_file training_data.json \
    --seed 0 \
    --benchmark_name hoverBench  # Optional, auto-detected
```

**What it does:**
- Loads GEPA optimization results (Pareto frontiers, candidate scores)
- Loads the validation set for the appropriate benchmark
- Loads all system prompt candidates
- Creates one training example per task, including ALL candidates and their rewards
- Saves structured data suitable for both MLP and LM training

**Output:**
- `training_data.json`: Main training data
- `candidate_metadata.json`: Metadata about candidates (without full prompts)
- `dataset_summary.json`: Statistics about the dataset

**Key Parameters:**
- `--experiment_dir`: Path to GEPA experiment directory
- `--output_file`: Where to save training data
- `--benchmark_name`: Which benchmark (auto-detected from dir name if not specified)
- `--seed`: Random seed for reproducibility

### Data Format Details

Each training example contains:
- **Task information**: The input (claim, question, etc.) and task index
- **All candidates**: List of all system prompt candidates with:
  - `candidate_idx`: Unique identifier
  - `candidate_system_prompt`: The full prompt text
  - `reward`: Ground truth performance (0.0 or 1.0 typically)
  - `in_pareto_frontier`: Whether this candidate is Pareto-optimal
- **Pareto frontier**: List of candidate indices on the Pareto frontier

## Step 2a: MLP Router Training

### Train the Neural Network Router

```bash
python train_mlp_router.py \
    --training_data training_data.json \
    --backbone sentence-transformers/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --num_epochs 10 \
    --lr 2e-5 \
    --val_split 0.2 \
    --loss_type bce \
    --output_dir router/checkpoints \
    --seed 42
```

**Architecture:**
- **Encoder**: Pre-trained transformer (e.g., MiniLM, BERT, RoBERTa)
- **Input**: Tokenized (task_input, candidate_prompt) pairs
- **Pooling**: CLS token or pooler output
- **Head**: Linear layer with dropout
- **Output**: Single scalar score per pair

**Training Process:**
1. **Data Expansion**: Each task with N candidates → N training pairs
2. **Tokenization**: Concatenate task input + candidate prompt
3. **Forward Pass**: Encoder → Pooling → Linear → Score
4. **Loss**: Binary cross-entropy (treats reward as probability)
5. **Optimization**: AdamW with linear warmup schedule

**Key Hyperparameters:**
- `--backbone`: Pre-trained model (affects embedding quality)
- `--max_length`: Sequence length (512 typical, increase if prompts are long)
- `--batch_size`: 16-64 depending on GPU memory
- `--lr`: 1e-5 to 5e-5 (lower for larger models)
- `--dropout`: 0.1-0.3 (regularization)
- `--loss_type`: `bce` (binary) or `mse` (regression)
- `--val_split`: 0.1-0.3 (task-level split to avoid leakage)

**Validation:**
- Evaluates on held-out tasks
- Computes top-1, top-3, top-5 accuracy
- Tracks reward prediction error (MSE, MAE)
- Saves best checkpoint based on top-1 accuracy

**Output:**
- `checkpoints/best_checkpoint.pt`: Best model
- `checkpoints/checkpoint_epoch_N.pt`: Regular checkpoints
- `checkpoints/training_history.json`: Training curves
- `checkpoints/training_args.json`: Hyperparameters used

### Using the Trained MLP Router

```python
from mlp_router import MLPRouter

router = MLPRouter(
    checkpoint_path='router/checkpoints/best_checkpoint.pt',
    backbone_name='sentence-transformers/all-MiniLM-L6-v2',
    benchmark_name='hoverBench'
)

# Select best candidate
selected_idx = router.select_candidate(task_input, candidates)

# Get probability distribution
distribution = router.get_probability_distribution(task_input, candidates)

# Get top-k candidates
top_k = router.get_top_k_candidates(task_input, candidates, k=3)
```

## Step 2b: Qwen Router SFT (Supervised Fine-Tuning)

### Fine-tune Language Model with DSPy

```bash
# Set API key
export OPENAI_API_KEY=your_key_here

python train_qwen_sft.py \
    --training_data training_data.json \
    --model openai/gpt-3.5-turbo \
    --optimizer bootstrap \
    --max_bootstrapped_demos 4 \
    --max_labeled_demos 8 \
    --use_reasoning \
    --max_train_samples 100 \
    --val_split 0.2 \
    --output_dir router/qwen_checkpoints \
    --seed 42
```

**Architecture:**
- **Base Model**: Any DSPy-compatible LM (GPT, Claude, Qwen, etc.)
- **Input**: Task description + formatted candidate list
- **Reasoning**: Optional chain-of-thought before selection
- **Output**: Selected candidate index

**Training Process (DSPy Optimization):**

1. **Data Formatting**:
   - Convert each task to a DSPy Example
   - Format candidates with indices and (truncated) prompts
   - Label = best performing candidate index

2. **Optimization Methods**:

   a) **BootstrapFewShot** (Recommended for quick training):
   ```python
   # Generates few-shot examples by bootstrapping
   # Uses successful predictions as demonstrations
   optimizer = dspy.BootstrapFewShot(
       metric=metric_correct_selection,
       max_bootstrapped_demos=4,
       max_labeled_demos=8
   )
   ```

   b) **COPRO** (Prompt optimization):
   ```python
   # Optimizes instruction prompts
   optimizer = dspy.COPRO(
       metric=metric_correct_selection,
       breadth=10,
       depth=3
   )
   ```

   c) **MIPROv2** (Advanced optimization):
   ```python
   # Combines prompt and demonstration optimization
   optimizer = dspy.MIPROv2(
       metric=metric_correct_selection,
       num_candidates=10,
       init_temperature=1.0
   )
   ```

3. **Compilation**:
   - Optimizer searches for best prompts/demonstrations
   - Evaluates using the metric function
   - Returns optimized predictor

**Key Parameters:**
- `--model`: LM to use (e.g., `openai/gpt-3.5-turbo`, `qwen/qwen-2.5-72b-instruct`)
- `--optimizer`: Optimization strategy (`bootstrap`, `copro`, `miprov2`)
- `--max_bootstrapped_demos`: Number of few-shot examples
- `--use_reasoning`: Enable chain-of-thought
- `--max_candidates_display`: Limit candidates shown (for token efficiency)
- `--max_prompt_length`: Truncate long prompts

**Validation:**
- Evaluates before and after optimization
- Reports accuracy improvement
- Saves compiled predictor

**Output:**
- `qwen_checkpoints/compiled_predictor_*.json`: Optimized DSPy program
- `qwen_checkpoints/training_config_*.json`: Training metadata

### Using the Trained Qwen Router

```python
import dspy
from qwen_router import QwenRouter

# Setup DSPy
lm = dspy.LM('openai/gpt-3.5-turbo', api_key='...')
dspy.configure(lm=lm)

# Load compiled predictor (optional)
# predictor = dspy.load('qwen_checkpoints/compiled_predictor_bootstrap.json')

router = QwenRouter(
    lm=lm,
    benchmark_name='hoverBench',
    use_reasoning=True
)

# Select best candidate
selected_idx = router.select_candidate(task_input, candidates)
```

## Step 3: Evaluation

### Evaluate a Single Router

```bash
python evaluate_router.py \
    --data_path training_data.json \
    --router random \
    --output_file results/random_results.json
```

### Compare Multiple Routers

```bash
# Compare baselines
python evaluate_router.py \
    --data_path training_data.json \
    --router all \
    --output_file results/baseline_comparison.json
```

### Evaluate Trained Routers

```python
from evaluate_router import RouterEvaluator
from mlp_router import MLPRouter
from qwen_router import QwenRouter
from router_base import RandomRouter, OracleRouter

# Load evaluator
evaluator = RouterEvaluator('training_data.json')

# Create routers
mlp_router = MLPRouter('checkpoints/best_checkpoint.pt')
qwen_router = QwenRouter(use_reasoning=True)
random_router = RandomRouter(seed=42)
oracle_router = OracleRouter()

# Compare
results = evaluator.compare_routers(
    [random_router, mlp_router, qwen_router, oracle_router],
    top_k=[1, 3, 5],
    output_file='results/full_comparison.json'
)
```

### Evaluation Metrics

**Accuracy Metrics:**
- `top_1_accuracy`: Exact match rate (selected best candidate)
- `top_3_accuracy`: Best candidate in top-3 predictions
- `top_5_accuracy`: Best candidate in top-5 predictions

**Performance Metrics:**
- `avg_selected_reward`: Average reward of selected candidates
- `avg_best_reward`: Average best reward available (upper bound)
- `reward_gap`: Difference between best and selected
- `normalized_regret`: Reward gap / best reward

**Quality Metrics:**
- `pareto_selection_rate`: % of selections from Pareto frontier

## Training Tips

### MLP Router

**For Better Performance:**
1. **Use a stronger backbone**: `roberta-large` > `all-MiniLM-L6-v2`
2. **Increase max_length**: If prompts are long (>256 tokens)
3. **More training data**: At least 50-100 tasks, more is better
4. **Hyperparameter tuning**: Try different learning rates, dropout rates

**For Faster Training:**
1. **Smaller backbone**: `all-MiniLM-L6-v2` is fast
2. **Reduce max_length**: 256 or 384 instead of 512
3. **Increase batch_size**: If you have GPU memory
4. **Fewer epochs**: 5-7 epochs often sufficient

**Common Issues:**
- **Overfitting**: Increase dropout, reduce epochs, add more data
- **Poor validation acc**: Check data quality, try different backbone
- **OOM errors**: Reduce batch_size or max_length

### Qwen Router

**For Better Performance:**
1. **Use reasoning**: `--use_reasoning` enables chain-of-thought
2. **More demonstrations**: Increase `--max_bootstrapped_demos`
3. **Better model**: Use GPT-4 or Qwen-72B instead of GPT-3.5
4. **More training data**: Bootstrap works better with more examples

**For Faster/Cheaper Training:**
1. **Limit training samples**: `--max_train_samples 50`
2. **Use smaller model**: GPT-3.5-turbo is cheap
3. **Reduce demos**: `--max_bootstrapped_demos 2`
4. **Truncate prompts**: `--max_prompt_length 100`

**Common Issues:**
- **API rate limits**: Add delays, use batch processing
- **Poor improvement**: Try different optimizer or more data
- **Parsing errors**: Improve prompt formatting, add examples

## Complete End-to-End Example

```bash
# 1. Prepare data
python construct_training_data.py \
    --experiment_dir ../experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B \
    --output_file training_data.json

# 2a. Train MLP router
python train_mlp_router.py \
    --training_data training_data.json \
    --batch_size 32 \
    --num_epochs 10 \
    --output_dir checkpoints/mlp

# 2b. Train Qwen router
export OPENAI_API_KEY=your_key
python train_qwen_sft.py \
    --training_data training_data.json \
    --model openai/gpt-3.5-turbo \
    --optimizer bootstrap \
    --max_train_samples 100 \
    --output_dir checkpoints/qwen

# 3. Evaluate and compare
python -c "
from evaluate_router import RouterEvaluator
from mlp_router import MLPRouter
from qwen_router import QwenRouter
from router_base import RandomRouter, OracleRouter
import dspy

# Setup
evaluator = RouterEvaluator('training_data.json')

# Load routers
mlp = MLPRouter('checkpoints/mlp/best_checkpoint.pt')

lm = dspy.LM('openai/gpt-3.5-turbo', api_key='...')
dspy.configure(lm=lm)
qwen = QwenRouter(lm=lm)

# Compare
evaluator.compare_routers(
    [RandomRouter(), mlp, qwen, OracleRouter()],
    output_file='comparison.json'
)
"
```

## Files Overview

### Core Files
- `construct_training_data.py`: Data preparation
- `train_mlp_router.py`: MLP training pipeline
- `train_qwen_sft.py`: Qwen SFT pipeline
- `evaluate_router.py`: Evaluation framework

### Router Implementations
- `router_base.py`: Base Router class + baselines (Random, Oracle)
- `mlp_router.py`: MLP-based router
- `qwen_router.py`: LM-based router (DSPy)

### Model Definitions
- `model.py`: Neural network architectures (PromptRouterCrossEncoder)

### Output Directories
- `checkpoints/`: MLP model checkpoints
- `qwen_checkpoints/`: Compiled DSPy predictors
- `results/`: Evaluation results

## Next Steps

After training and evaluation:

1. **Deploy the best router**: Use in production routing pipeline
2. **Continuous improvement**: Retrain with new GEPA experiments
3. **A/B testing**: Compare router performance in real scenarios
4. **Ensemble methods**: Combine MLP and LM routers
5. **Adaptive routing**: Use confidence scores for fallback strategies

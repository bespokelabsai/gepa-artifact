# Prompt Router Training Dataset

This directory contains the training dataset for a prompt router that determines which system prompt candidate to use for a given task.

## Files

- **`training_data.json`**: Main training dataset (300 examples)
- **`candidate_metadata.json`**: Metadata about the 22 system prompt candidates
- **`dataset_summary.json`**: Summary statistics about the dataset
- **`construct_training_data.py`**: Script used to generate this dataset

## Dataset Structure

Each training example in `training_data.json` has the following structure:

```json
{
  "task_idx": 0,
  "claim": "...",  // The claim to verify (task input)
  "label": 1,      // Ground truth label (1=supported, 0=not supported)
  "supporting_facts": [...],  // Supporting facts for the claim
  "pareto_frontier": [1, 2, 5, ...],  // Candidate indices on Pareto frontier for this task
  "selected_candidate": 1,  // Best candidate selected for this task
  "selected_candidate_score": true,  // Score of selected candidate on this task
  "all_candidate_scores": {...}  // Scores of all Pareto frontier candidates
}
```

## Key Information

- **Number of tasks**: 300 (validation set from HoVer benchmark)
- **Number of candidates**: 22 (system prompt candidates from GEPA optimization)
- **Average Pareto frontier size**: 15.23 candidates per task
- **Selection strategy**: For each task, the best-performing candidate from its Pareto frontier is selected

## Pareto Frontier

For each task, the Pareto frontier contains the candidate indices that achieve the best trade-offs across multiple objectives. These represent the "non-dominated" solutions where improving one metric would require degrading another.

## Usage

### Regenerating the Dataset

To regenerate the dataset with different settings:

```bash
python router/construct_training_data.py \
  --experiment_dir experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B \
  --output_file router/training_data.json \
  --seed 0 \
  --strategy best_from_pareto
```

### Strategy Options

- **`best_from_pareto`** (default): Select the single best-performing candidate from the Pareto frontier for each task (300 examples)
- **`all_pareto`**: Create one training example per Pareto frontier candidate (more examples, includes all valid candidates)

### Training a Router

The training data can be used to train a classifier that:
1. Takes a task (claim + context) as input
2. Predicts which system prompt candidate to use (0-21)

Example approaches:
- **Classification**: Train a model to predict `selected_candidate` given the `claim`
- **Learning-to-rank**: Rank all candidates in `pareto_frontier` based on their scores
- **Multi-task learning**: Predict both the best candidate and whether the claim is supported

## System Prompt Candidates

The actual system prompts are stored as pickle files in:
```
experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B/prog_candidates/{0-21}/program.pkl
```

Each candidate directory contains:
- `program.pkl`: Serialized DSPy program with the system prompt
- `metadata.json`: Version information

## Training the Router

### Quick Start

Train with default settings:

```bash
python router/train.py
```

This will:
- Load the training data from `router/training_data.json`
- Split into train/val (80/20)
- Train a cross-encoder model for 5 epochs
- Save checkpoints to `router/checkpoints/`

### Training Arguments

```bash
python router/train.py \
  --backbone sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 8 \
  --num_negatives 3 \
  --num_epochs 5 \
  --lr 2e-5 \
  --max_length 512 \
  --output_dir router/checkpoints \
  --val_split 0.2
```

**Key arguments:**
- `--backbone`: Pretrained model to use (default: sentence-transformers/all-MiniLM-L6-v2)
- `--batch_size`: Training batch size
- `--num_negatives`: Number of negative candidates per positive example
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--max_length`: Maximum sequence length for tokenization
- `--val_split`: Validation split ratio

### Model Architecture

The router uses a **cross-encoder** architecture:
1. **Input**: Concatenated (query, system_prompt) pairs
2. **Encoder**: Transformer backbone (e.g., BERT, MiniLM)
3. **Output**: Scalar compatibility score

During training:
- Positive pair: (query, best_candidate_prompt)
- Negative pairs: (query, other_candidate_prompts)
- Loss: Binary cross-entropy

During inference:
- Score all (query, candidate) pairs
- Select candidate with highest score

## Inference

### Load Trained Model

```python
from router.inference import PromptRouter

router = PromptRouter(
    checkpoint_path='router/checkpoints/best_checkpoint.pt',
    prog_candidates_dir='experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B/prog_candidates'
)

# Route a single query
query = "John de Mol Jr. is the Dutch media tycoon..."
best_candidate = router.route(query, top_k=1)[0]
print(f"Best candidate: {best_candidate}")

# Get top-3 with scores
top_3 = router.route(query, top_k=3, return_scores=True)
for candidate, score in top_3:
    print(f"Candidate {candidate}: {score:.4f}")
```

### Command Line Inference

Single query:
```bash
python router/inference.py \
  --checkpoint router/checkpoints/best_checkpoint.pt \
  --query "Your query here" \
  --top_k 3
```

Batch evaluation:
```bash
python router/inference.py \
  --checkpoint router/checkpoints/best_checkpoint.pt \
  --test_data router/training_data.json \
  --top_k 3 \
  --output router/predictions.json
```

## Evaluation Metrics

The router is evaluated using:
- **Top-1 Accuracy**: Percentage of times the model picks the exact best candidate
- **Top-K Accuracy**: Percentage of times the best candidate is in the top-K predictions

Note: Only tasks where the best candidate succeeded (score=True/1) are evaluated.

## Files

After training, you'll have:
- `checkpoints/best_checkpoint.pt`: Best model checkpoint
- `checkpoints/checkpoint_epoch_N.pt`: Epoch checkpoints
- `checkpoints/training_args.json`: Training configuration
- `checkpoints/training_history.json`: Training metrics per epoch

## Notes

- The scores in the dataset are boolean (True/False) or 0, indicating whether the candidate succeeded on the task
- All tasks are 3-hop reasoning tasks from the HoVer (Hover) benchmark
- The dataset uses the same train/val split and preprocessing as the original GEPA experiments (seed=0)
- Training typically takes 10-30 minutes on a GPU depending on the backbone model

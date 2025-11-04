# Running Inference with Pre-trained Models

This guide explains how to run inference using pre-trained model weights, which can be downloaded from the [this release](https://github.com/noegroup/ScoreMD/releases/tag/1.0.0).

## Overview

Our model supports two inference modes:

1. **Independent Sampling (IID)**: Generate independent samples using the classical generative diffusion process (denoising diffusion).
2. **Molecular Dynamics Simulation**: Use the score-force relation to extract forces for continuous MD simulations. The relation is: $\nabla_x \log p(x, t=0) = - \nabla_x \frac{U(x)}{k_BT} \approx \nabla_x \text{NNET}(x, t=0)$.

## Configuration Parameters

### Independent Sampling (IID) Parameters

- **`evaluation.num_iid_samples`**: Number of independent samples to generate
- **`evaluation.inference_bs`**: Batch size for inference (controls memory usage)

### Molecular Dynamics Simulation Parameters

- **`evaluation.num_langevin_samples`**: Total number of Langevin samples to generate
- **`evaluation.num_langevin_intermediate_steps`**: Number of Langevin steps performed before storing one sample (default: 10). Example: With 100 samples and 10 intermediate steps, you get 1000 total Langevin steps but only store every 10th sample.
- **`evaluation.num_parallel_langevin_samples`**: Number of parallel Langevin simulations to run simultaneously
- **`evaluation.langevin_dt`**: Time step for the Langevin simulation in picoseconds. If not specified, the timestep is automatically determined by the dataset
- **`evaluation.eval_t`**: Time at which the model is evaluated. Ideally use `0`, but to avoid numerical issues we typically use $10^{-5}$ for evaluation. This was changed when evaluating the "Two For One" model (see paper for details)

### Common Parameters

- **`evaluation.seed`**: Random seed for reproducibility (default: 1). Can be set for both sampling methods. 
- **`wandb.enabled`**: Whether to use Weights and Biases for logging (default: True).
- **`wandb.name`**: Name of the run for Weights and Biases.
- **`evaluation.only_store_results`**: If this is true, we only store the results with minimal evaluation. Especially for the large runs, this can save a lot of time and memory.

## Getting the Model Weights
You can either use the pre-trained models we provide or train your own models.

* Pre-trained model weights are available in the [this release](https://github.com/noegroup/ScoreMD/releases/tag/1.0.0). Simply download the weights and unzip them into the `models` directory.
* You can also use your own trained models by specifying the path to your model weights in the `load_dir` parameter. See [TRAIN.md](TRAIN.md) for more details.

## Running Inference
Inference runs automatically after training, but you can also run it separately using pre-trained weights. The process is similar to training (see [TRAIN.md](TRAIN.md)), but you add the `load_dir` parameter to specify where to load the model weights from. For you not having to specify all the parameters, you can use the existing config file by adding the `--config-path` and `--config-name` flags. Further, you will need to ensure that the model does not continue training by setting `continue_from=null`.

### Example: Alanine Dipeptide
```bash
python train.py \
  --config-path models/aldp/both/.hydra \
  --config-name config.yaml \
  load_dir=models/aldp/both \
  continue_from=null \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=120000 \
  evaluation.eval_t=1e-5 \
  evaluation.only_store_results=False \
  wandb.enabled=True \
  wandb.name=aldp-both-fp-0.001-inference
```

This command will:
- Load the pre-trained model from `models/aldp/both`
- Run Langevin dynamics simulation with 120,000 samples
- Generate 120,000 independent samples
- Generate results and save them to the `outputs/` directory

### Example: Chignolin
We can use almost the same command as for alanine dipeptide, but we need to change the config path to `models/chignolin/both/.hydra` and the model weights to `models/chignolin/both`.

```bash
python train.py \
  --config-path models/chignolin/both/.hydra \
  --config-name config.yaml \
  load_dir=models/chignolin/both \
  continue_from=null \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=120000 \
  evaluation.eval_t=1e-5 \
  evaluation.only_store_results=False \
  wandb.enabled=True \
  wandb.name=chignolin-both-fp-0.002-inference
```

### Example: Alanine Dipeptide - Full Specification
You can also specify all the parameters manually. If you change the parameters a lot, it might be easier to copy and paste the parameters. The following command is equivalent to the previous alanine dipeptide inference command.

```bash
python train.py dataset=aldp \
  load_dir=models/aldp/both \
  dataset.limit_samples=50_000 \
  dataset.validation=False \
  training_schedule=vp_three_models \
  +architecture=transformer/score_score_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=1000 \
  training_schedule.epochs.1=2000 \
  training_schedule.epochs.2=7000 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=100000 \
  +training_schedule/augment=random_rotations \
  dataset.coarse_graining_level=full \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=120000 \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.losses.2.loss.beta=0.001 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  evaluation.only_store_results=False \
  wandb.enabled=True \
  +wandb.name=aldp-both-fp-0.001-inference
```

## Running Multiple Inference Runs

Using Hydra's multirun feature, you can easily run multiple inference runs with different seeds or configurations by adding the `-m` flag.

```bash
python train.py \
  --config-path models/aldp/both/.hydra \
  --config-name config.yaml \
  -m seed=0,1,2 \
  load_dir=models/aldp/both \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=100000 \
  evaluation.eval_t=1e-5 \
  evaluation.only_store_results=False \
  wandb.enabled=True \
  wandb.name=aldp-both-fp-0.001-inference
```

This will run inference three times with seeds 0, 1, and 2. Results will be saved in the `multirun/` directory, with separate subdirectories for each run.

## Evaluating on Test Sets

For minipeptides (two amino acids), you can evaluate on the test set by:
1. Setting the validation path to point to your test set using `dataset.val_path`
2. Using `load_dir` to load a pre-trained model
3. Optionally limiting evaluation to specific peptides using `evaluation.limit_inference_peptides`

### Example: Minipeptides Test Set Evaluation

```bash
python train.py \
  --config-path models/minipeptides/diffusion/.hydra \
  --config-name config.yaml \
  load_dir=models/minipeptides/diffusion \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  evaluation.eval_t=1e-5 \
  "dataset.val_path=./storage/minipeptides/test.npy" \
  "evaluation.limit_inference_peptides=[KS, KG, AT, LW, KQ, NY, IM, TD, HT, NF, RL, ET, AC, RV, HP, AP]" \
  evaluation.num_parallel_langevin_samples=10 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=100000 \
  evaluation.only_store_results=False \
  wandb.enabled=True \
  wandb.name=minipeptide-test
```

This will:
- Evaluate on the test set located at `./storage/minipeptides/test.npy` (you can get this [here](https://github.com/noegroup/ScoreMD/releases/tag/1.0.0))
- Generate both IID samples (100,000) and Langevin samples (120,000) for comparison
- Limit evaluation to the specified peptides
- Save results to the `outputs/` directory

## Output Location

All inference results are saved in the `outputs/` directory (or `multirun/` for multirun experiments). Each run creates a timestamped subdirectory containing:
- Generated samples (`.npy` files)
- Evaluation metrics (`.json` files)
- Visualization plots (`.png` files)
- Trajectory animations (if applicable)

For more details on the evaluation metrics and plotting capabilities, see the [evaluation README](evaluation/README.md).

# Troubleshooting

**Issue**: Evaluation takes too long or runs out of memory
- **Solution**: Fokker-Planck error evaluation can be computationally expensive. You can reduce `evaluation.num_fp_timepoints`, and even set it to 0 to skip this evaluation.
- You can also decrease batch sizes, or set `evaluation.only_store_results=True` to skip computations.

**Issue**: Out of memory errors
- **Solution**: Reduce `evaluation.inference_bs`, `evaluation.num_iid_samples` and `evaluation.num_parallel_langevin_samples`
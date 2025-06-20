<h1 align="center">Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models</h1>
<p align="center">
A framework for training diffusion models with stable, self-consistent scores near the data distribution.
</p>

<p align="center">
<img src="images/main.png" alt="Visualization of the main idea of this paper."/>
</p>

# 🚀 Getting Started

All dependencies are managed with [pixi](https://github.com/prefix-dev/pixi), ensuring fully reproducible environments.
To set up the environment, simply run:

```bash
pixi install --frozen
```

Prefer using your own toolchain like conda?
Check out our `pyproject.toml` and install the dependencies with your preferred tool.
In that case, replace `pixi run` with standard Python execution, for example:

```bash
python train.py
```

## Training & Evaluation

We use Hydra for configuration management. You can override any config via command-line arguments or config files.

### Example: Toy Systems

Train on example toy systems using the provided configurations:

```bash
pixi run train dataset=double_well +architecture=mlp/small_score
pixi run train dataset=double_well_2d +architecture=mlp/small_score
```

Outputs will be saved to the `outputs/` directory.

# 📄 Reproducing Paper Results
Below, we list the exact commands used to generate the main results from the paper.
Note that in this version of the repository, we use slightly different names for the parameters, e.g., we use $\beta$ instead of $\alpha$ for the regularization strength.

## Müller-Brown Potential

### Baseline Model

```bash
pixi run train dataset=mueller_brown +architecture=mlp/small_potential \
  training_schedule.epochs=180 \
  training_schedule.losses.0.time_weighting.midpoint=0.5
```

### Mixture of Experts

```bash
pixi run train dataset=mueller_brown +architecture=mlp/small_mixture_fair \
  weighting_function=ranged \
  training_schedule=vp_three_models \
  training_schedule.epochs.0=30 \
  training_schedule.epochs.1=30 \
  training_schedule.epochs.2=120 \
  training_schedule.losses.2.time_weighting.midpoint=0.05
```

### Fokker-Planck Regularization

```bash
pixi run train dataset=mueller_brown +architecture=mlp/small_potential \
  training_schedule.epochs=180 \
  training_schedule.losses.0.loss.div_est=2 \
  training_schedule.losses.0.loss.beta=0.0005 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  training_schedule.losses.0.time_weighting.midpoint=0.5
```

### Both (Mixture + Fokker-Planck)

```bash
pixi run train dataset=mueller_brown +architecture=mlp/small_mixture_fair \
  weighting_function=ranged \
  training_schedule=vp_three_models \
  training_schedule.losses.2.loss.div_est=2 \
  training_schedule.losses.2.loss.beta=0.0005 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.epochs.0=30 \
  training_schedule.epochs.1=30 \
  training_schedule.epochs.2=120 \
  training_schedule.losses.2.time_weighting.midpoint=0.05
```

## Alanine Dipeptide

### Baseline Model
```bash
pixi run train dataset=aldp \
  dataset.limit_samples=50_000 \
  dataset.validation=False \
  +architecture=transformer/potential \
  training_schedule.epochs.0=10000 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=1000 \
  training_schedule.losses.0.loss.alpha=0 \
  training_schedule.losses.0.loss.beta=0 \
  +training_schedule/augment=random_rotations \
  dataset.coarse_graining_level=full \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  wandb.enabled=True \
  +wandb.name=aldp-baseline
```

### Mixture
```bash
pixi run train dataset=aldp \
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
  training_schedule.losses.2.loss.alpha=0 \
  +training_schedule/augment=random_rotations \
  dataset.coarse_graining_level=full \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.losses.2.loss.beta=0 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  wandb.enabled=True \
  +wandb.name=aldp-mixture
```

### Fokker-Planck
```bash
pixi run train dataset=aldp \
  dataset.limit_samples=50_000 \
  dataset.validation=False \
  +architecture=transformer/potential \
  training_schedule.epochs.0=10000 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=1000 \
  training_schedule.losses.0.loss.alpha=0 \
  training_schedule.losses.0.loss.beta=0.0005 \
  +training_schedule/augment=random_rotations \
  dataset.coarse_graining_level=full \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  wandb.enabled=True \
  +wandb.name=aldp-fp-0.0005
```

### Both (Mixture + Fokker-Planck)
```bash
pixi run train dataset=aldp \
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
  training_schedule.losses.2.loss.alpha=0 \
  +training_schedule/augment=random_rotations \
  dataset.coarse_graining_level=full \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.losses.2.loss.beta=0.0001 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  wandb.enabled=True \
  +wandb.name=aldp-both-fp-0.0001
```

## Minipeptides
### Baseline
```bash
pixi run train dataset=minipeptides \
  +architecture=transformer/large_potential \
  training_schedule.epochs=120 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  "evaluation.limit_inference_peptides=[KA, RP]" \
  training_schedule.losses.0.loss.beta=0.000 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=minipeptide-baseline
```

### Mixture
```bash
pixi run train dataset=minipeptides \
  training_schedule=vp_three_models \
  +architecture=transformer/score_score_large_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=10 \
  training_schedule.epochs.1=20 \
  training_schedule.epochs.2=100 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10000 \
  +training_schedule/augment=random_rotations \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  "evaluation.limit_inference_peptides=[KA, RP]" \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.losses.2.loss.beta=0.000 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  wandb.enabled=True \
  +wandb.name=minipeptide-mixture
```

### Fokker-Planck
```bash
pixi run train dataset=minipeptides \
  +architecture=transformer/large_potential \
  training_schedule.epochs=120 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  "evaluation.limit_inference_peptides=[KA, RP]" \
  training_schedule.losses.0.loss.beta=0.0005 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  wandb.enabled=True \
  +wandb.name=minipeptide-fp-0.0005
```

### Both (Mixture + Fokker-Planck)
```bash
pixi run train dataset=minipeptides \
  training_schedule=vp_three_models \
  +architecture=transformer/score_score_large_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=10 \
  training_schedule.epochs.1=20 \
  training_schedule.epochs.2=100 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10000 \
  +training_schedule/augment=random_rotations \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  "evaluation.limit_inference_peptides=[KA, RP]" \
  evaluation.eval_t=1e-5 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.losses.2.loss.beta=0.0001 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  wandb.enabled=True \
  +wandb.name=minipeptide-both-fp-0.0001
```

## Example Inference on Testste
For the minipeptides, the training script can also be used for evaluation on the test set. For that, set the validation set to the test set and use `load_dir`.

```bash
pixi run train dataset=minipeptides \
  +architecture=transformer/large_potential \
  training_schedule.epochs=120 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.0.loss.beta=0.000 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  "dataset.val_path=./storage/minipeptides/test.npy" \
  "evaluation.limit_inference_peptides=[KS, KG, AT, LW, KQ, NY, IM, TD, HT, NF, RL, ET, AC, RV, HP, AP]" \
  evaluation.num_parallel_langevin_samples=10 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=100000 \
  load_dir=./outputs/minipeptides/DATE/TIME
  wandb.enabled=True \
  +wandb.name=minipeptide-test
```

# Contribution
Feel free to open an issue if you encounter any problems or have questions.


# Training the Models from the Paper
Below, we list the exact commands used to generate the main results from the paper.
Note that in this repository, we use slightly different names for the parameters. Most notably, we use $\beta$ instead of $\alpha$ for the regularization strength.

You can also find the full hydra configs for the models we trained in the .hydra folder of the resepective model weights. Refer to the [README.md](README.md) on where to find these models.

In case you want to train a model the following two arguments might be useful: `load_dir` can be used to load a trained model, and `continue_from` will do the same but work in the same output directory as specified.

## Müller-Brown Potential

### Baseline Model

```bash
python train.py dataset=mueller_brown +architecture=mlp/small_potential \
  training_schedule.epochs=180 \
  training_schedule.losses.0.time_weighting.midpoint=0.5
```

### Mixture of Experts

```bash
python train.py dataset=mueller_brown +architecture=mlp/small_mixture_fair \
  weighting_function=ranged \
  training_schedule=vp_three_models \
  training_schedule.epochs.0=30 \
  training_schedule.epochs.1=30 \
  training_schedule.epochs.2=120 \
  training_schedule.losses.2.time_weighting.midpoint=0.05
```

### Fokker-Planck Regularization

```bash
python train.py dataset=mueller_brown +architecture=mlp/small_potential \
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
python train.py dataset=mueller_brown +architecture=mlp/small_mixture_fair \
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
python train.py dataset=aldp \
  dataset.limit_samples=50_000 \
  dataset.validation=False \
  +architecture=transformer/potential \
  training_schedule.epochs.0=10000 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=1000 \
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
python train.py dataset=aldp \
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
python train.py dataset=aldp \
  dataset.limit_samples=50_000 \
  dataset.validation=False \
  +architecture=transformer/potential \
  training_schedule.epochs.0=10000 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=1000 \
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
python train.py dataset=aldp \
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
  evaluation.eval_t=1e-5 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  training_schedule.losses.2.loss.beta=0.001 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  wandb.enabled=True \
  +wandb.name=aldp-both-fp-0.001
```

## Chignolin
### Baseline Model
```bash
python train.py dataset=chignolin \
  +architecture=transformer/large_potential \
  training_schedule.epochs=2000 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=1000000 \
  evaluation.num_iid_samples=427794 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.0.loss.beta=0 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=chignolin-baseline
```

### Mixture
```bash
python train.py dataset=chignolin \
  training_schedule=vp_three_models \
  +architecture=transformer/score_score_large_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=100 \
  training_schedule.epochs.1=200 \
  training_schedule.epochs.2=1700 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=1000000 \
  evaluation.num_iid_samples=427794 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  training_schedule.losses.2.loss.beta=0 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=chignolin-mixture
```

### Fokker-Planck
```bash
python train.py dataset=chignolin \
  +architecture=transformer/large_potential \
  training_schedule.epochs=2000 \
  optimizer.learning_rate=0.0001 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=1000000 \
  evaluation.num_iid_samples=427794 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.0.loss.beta=0.002 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=chignolin-fp-0.002
```

### Both (Mixture + Fokker-Planck)
```bash
python train.py dataset=chignolin \
  training_schedule=vp_three_models \
  +architecture=transformer/score_score_large_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=100 \
  training_schedule.epochs.1=200 \
  training_schedule.epochs.2=1700 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=1000000 \
  evaluation.num_iid_samples=427794 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  training_schedule.losses.2.loss.beta=0.002 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=chignolin-both-fp-0.002
```

## BBA

### Baseline Model
```bash
python train.py dataset=bba \
  +architecture=transformer/large_potential \
  training_schedule.epochs=1500
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=100000 \
  evaluation.num_iid_samples=1300156 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.0.loss.beta=0 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=bba-baseline
```

### Mixture
```bash
python train.py dataset=bba \
  training_schedule=vp_three_models \
  +architecture=transformer/large_score_large_score_large_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=200 \
  training_schedule.epochs.1=400 \
  training_schedule.epochs.2=1200 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=100000 \
  evaluation.num_iid_samples=1300156 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  training_schedule.losses.2.loss.beta=0 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=bba-mixture
```

### Fokker-Planck
```bash
python train.py dataset=bba \
  +architecture=transformer/large_potential \
  training_schedule.epochs=1500
  training_schedule.BS=1024 \
  optimizer.learning_rate=0.0001 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=100000 \
  evaluation.num_iid_samples=1300156 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.0.loss.beta=0.01 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=bba-fp-0.01
```

### Both (Mixture + Fokker-Planck)
```bash
python train.py dataset=bba \
  training_schedule=vp_three_models \
  +architecture=transformer/large_score_large_score_large_potential \
  weighting_function=ranged \
  training_schedule.epochs.0=200 \
  training_schedule.epochs.1=400 \
  training_schedule.epochs.2=1200 \
  training_schedule.BS=1024 \
  optimizer.learning_rate=0.0001 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.num_langevin_samples=100000 \
  evaluation.num_iid_samples=1300156 \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.2.time_weighting.midpoint=0.03 \
  training_schedule.losses.2.loss.beta=0.02 \
  training_schedule.losses.2.loss.div_est=2 \
  +training_schedule.losses.2.loss.residual_fp=True \
  training_schedule.losses.2.loss.partial_t_approx=True \
  +training_schedule.losses.2.loss.single_gamma=True \
  evaluation.eval_t=1e-5 \
  wandb.enabled=True \
  +wandb.name=bba-both-fp-0.02
```

## Minipeptides
### Baseline
```bash
python train.py dataset=minipeptides \
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
  wandb.enabled=True \
  +wandb.name=minipeptide-baseline
```

### Mixture
```bash
python train.py dataset=minipeptides \
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
python train.py dataset=minipeptides \
  +architecture=transformer/large_potential \
  training_schedule.epochs=120 \
  training_schedule.BS=1024 \
  checkpoint_options.save_interval_steps=10 \
  +training_schedule/augment=random_rotations \
  evaluation.inference_bs=1024 \
  evaluation.fp_inference_bs=256 \
  training_schedule.losses.0.loss.beta=0.005 \
  training_schedule.losses.0.loss.div_est=2 \
  +training_schedule.losses.0.loss.residual_fp=True \
  training_schedule.losses.0.loss.partial_t_approx=True \
  +training_schedule.losses.0.loss.single_gamma=True \
  wandb.enabled=True \
  +wandb.name=minipeptide-fp-0.005
```

### Both (Mixture + Fokker-Planck)
```bash
python train.py dataset=minipeptides \
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

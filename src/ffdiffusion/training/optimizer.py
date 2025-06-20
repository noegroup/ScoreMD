import optax
from optax import GradientTransformation


def get_constant_lr_optimizer(num_steps: int, learning_rate: float, clip: float) -> GradientTransformation:
    return optax.chain(
        optax.clip(max_delta=clip),
        optax.adamw(learning_rate=learning_rate),
    )


def get_cosine_lr_optimizer(
    num_steps: int, learning_rate: float, min_learning_rate: float, clip: float
) -> GradientTransformation:
    return optax.chain(
        optax.clip(max_delta=clip),
        optax.adamw(
            learning_rate=optax.cosine_decay_schedule(learning_rate, num_steps, min_learning_rate / learning_rate)
        ),
    )

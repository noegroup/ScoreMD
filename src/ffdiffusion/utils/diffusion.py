import jax


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def batch_mul_A(a, b):
    return jax.vmap(lambda b: a * b)(b)


def get_score(model, params, *args, **kwargs):
    """
    This function takes the parameters of a model and returns a function that computes the score of the model at a
    specific (x,t) point. The function is returned as a closure, so that the model and its
    parameters are fixed when the function is returned.

    :param model: The model to evaluate
    :param params: The parameters of the model
    :param args: Additional arguments to the model.apply function
    :param kwargs: Additional keyword arguments to the model.apply function
    """

    def call(x, features, t):
        return model.apply(params, x, features, t, *args, **kwargs)

    return call


from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline


from linear_model.config.core import config


price_pipe = Pipeline(
    [
        (
            "Lasso",
            Lasso(
                alpha=config.model_config.alpha,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)

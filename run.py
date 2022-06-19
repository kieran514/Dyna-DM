import wandb.sweeps

from wandb.sweeps.config import tune
from wandb.sweeps.config.hyperopt import hp
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch

tune_config = tune.run(
    "train_dynamic.py",
    search_alg=HyperOptSearch(
        dict(
            disp_lr=hp.choice("disp_lr", [0.001, 0.0005, 0.0001]),
            ego_lr=hp.choice("disp_lr", [0.001, 0.0005, 0.0001]),
            obj_lr=hp.choice("disp_lr", [0.001, 0.0005, 0.0001]),
            weight_decay=hp.choice("weight-decay", [1e-5, 1e-6, 1e-7]),
            dmni=hp.choice("dmni", [3, 5, 10]),
            maxdinst=hp.choice("maxdinst", [0.5, 0.75, 0.9]),
            mindinst=hp.choice("mindinst", [0.0001, 0.01, 1, 5]),
            percentiou=hp.choice("percentiou", [0, 1, 5])),
        metric="val_loss",
        mode="min"),
    num_samples=50,
    )

tune_config.save("sweep-tune-hyperopt.yaml")
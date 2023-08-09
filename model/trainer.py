import copy
from collections import namedtuple
from typing import List

import pandas as pd
import pyro
import torch
from pyro import optim
from pyro.infer import SVI, Predictive, TraceGraph_ELBO
from pyro.infer.autoguide import AutoDelta

from model.hmm_module import VolatilityForecastModule

TrainedModel = namedtuple("TrainedModel", "model guide state_dict")


def _format_tensor(val):
    """
    Converts torch tensor into string
    """
    if len(val.shape) == 0:
        return f"{val.detach():.4f}"

    return f"[{', '.join([_format_tensor(e) for e in val])}]"


def train_model(training_series: pd.Series, epochs: int = 1000) -> TrainedModel:
    """
    Trains a single volatility forecasting model.
    :param volatility_df: Dataframe containing raw volatility data. Index consists of dates and values consist of
        volatilities.
    :param epochs: Number of epochs to train.
    :return: Object containing trained model.
    """
    pyro.clear_param_store()  # Pyro stores trained parameters as global variables. Clear previously trained parameters.

    in_sample_data = torch.tensor(training_series.values, dtype=torch.float32).unsqueeze(0)

    net = VolatilityForecastModule()
    if torch.cuda.is_available():  # Conduct training in GPU if it's available
        net = net.cuda()
    net = net.train()

    guide = AutoDelta(net)

    optimizer = optim.Adam({
        'lr': 1e-3
    })
    svi = SVI(net, guide, optimizer, loss=TraceGraph_ELBO())

    for i in range(epochs):
        elbo = svi.step(in_sample_data)

        #  Print variable values
        params = ', '.join([f"{key} = {_format_tensor(val)}" for key, val in pyro.get_param_store().items()])
        print(f"[Epoch {i}] ELBO: {elbo}, Params: {params}")

    final_model_state_dict = copy.deepcopy(pyro.get_param_store().get_state())

    return TrainedModel(net, guide, final_model_state_dict)


def predict(trained_model: TrainedModel, volatility_series: pd.Series) \
        -> (torch.Tensor, List[int]):
    """
    Generate volatility forecasts using a pre-trained model.
    :param trained_model: Object containing pre-trained model. Should match the output produced by `train_model` method.
    :param volatility_series: Dataframe containing raw volatility data. Index consists of dates and values consist
        of volatilities
    :return: A DataFrame containing sampled forecasts, with the same index and columns as volatility_df
    """
    pyro.get_param_store().set_state(trained_model.state_dict)

    # Set trained model to evaluation mode so it doesn't continue to change model parameters
    trained_model.model.eval()
    trained_model.guide.eval()

    x = torch.tensor(volatility_series.values, dtype=torch.float32).unsqueeze(0)
    predictive = Predictive(
        model=trained_model.model.cpu(),
        guide=trained_model.guide.cpu(),
        num_samples=1,
        return_sites=('predicted_volatility',)
    )
    preds = predictive(x)
    vol_preds = preds['predicted_volatility'].squeeze(0)

    volatility_pred_series = pd.Series(
        vol_preds.T.detach().cpu().numpy(),
        index=volatility_series.index
    )

    return volatility_pred_series

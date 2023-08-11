import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class VolatilityForecastModule(PyroModule):
    def forward(self, x: torch.Tensor, warmup_periods: int = 5):
        """
        Uses Kalman filter to forecast volatilities
        :param x: Volatility data to train the Kalman filter with
        :param warmup_periods: Number of periods to skip analysis on, to allow the model to "warm up" before making
        predictions for analysis.
        """

        n_t = x.shape[1]

        # Set first observed volatility as the forecast for second volatility
        vol_loc = x[0, 0].unsqueeze(-1).unsqueeze(-1)

        init_dist_scale = pyro.sample("init_dist_scale", dist.LogNormal(-2., 3.))
        vol_scale = torch.tensor([[init_dist_scale]]).expand([1, 1, 1])

        transition_matrix = torch.tensor([[1.]])

        transition_scale = pyro.sample("trans_scale", dist.LogNormal(-2., 3.))
        transition_dist = dist.Normal(0., transition_scale.unsqueeze(-1)).to_event(1)

        observation_matrix = torch.tensor([[1.]])

        observation_scale = pyro.sample("obs_scale", dist.LogNormal(-2., 3.))
        observation_dist = dist.Normal(0., observation_scale.unsqueeze(-1)).to_event(1)

        forecasts = []
        for t in range(1, n_t):
            init_dist = dist.MultivariateNormal(vol_loc, vol_scale)

            hmm = dist.GaussianHMM(
                init_dist,
                transition_matrix,
                transition_dist,
                observation_matrix,
                observation_dist,
                duration=1
            )

            next_x = x[0, t].unsqueeze(-1).unsqueeze(-1)
            if t >= warmup_periods:
                pyro.sample(f"obs_{t}", hmm, obs=next_x)

            new_init_dist = hmm.filter(next_x)
            vol_loc = new_init_dist.loc.detach()
            vol_scale = new_init_dist.covariance_matrix.detach()

            forecasts.append((vol_loc).squeeze())

        pyro.deterministic(f"predicted_volatility", torch.stack(forecasts, dim=0))

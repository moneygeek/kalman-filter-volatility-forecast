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

        # Set first observed volatility as the forecast for second volatility
        vol_loc = x[0, 0].unsqueeze(-1).unsqueeze(-1)

        init_dist_scale = pyro.sample("init_dist_scale", dist.LogNormal(-2., 3.))
        vol_scale = torch.tensor([[init_dist_scale]]).expand([1, 1, 1])

        # Expect the next true volatility to be, on average, the same as the previous true volatility
        transition_matrix = torch.tensor([[1.]])

        # The degree of changes expected for true volatilities.
        transition_scale = pyro.sample("trans_scale", dist.LogNormal(-2., 3.))

        transition_dist = dist.Normal(0., transition_scale.unsqueeze(-1)).to_event(1)

        # Expect observed volatility to be centered around true volatility
        observation_matrix = torch.tensor([[1.]])

        # The degree of measurement error expected
        observation_scale = pyro.sample("obs_scale", dist.LogNormal(-2., 3.))

        observation_dist = dist.Normal(0., observation_scale.unsqueeze(-1)).to_event(1)

        forecasts = []
        for t in range(1, x.shape[1]):  # Iterate through each observed volatility one by one
            init_dist = dist.MultivariateNormal(vol_loc, vol_scale)

            # Get distribution indicating where we expect the next volatility to be.
            hmm = dist.GaussianHMM(
                init_dist,
                transition_matrix,
                transition_dist,
                observation_matrix,
                observation_dist,
                duration=1
            )

            next_x = x[0, t].unsqueeze(-1).unsqueeze(-1)

            # Tell the trainer how well the observed volatility fit expectations
            if t >= warmup_periods:
                pyro.sample(f"obs_{t}", hmm, obs=next_x)

            # Update estiamte of where true volatility is given the latest data point
            new_init_dist = hmm.filter(next_x)
            vol_loc = new_init_dist.loc.detach()
            vol_scale = new_init_dist.covariance_matrix.detach()

            # Generate forecast for t+1, as of t
            forecasts.append((vol_loc).squeeze())

        pyro.deterministic(f"predicted_volatility", torch.stack(forecasts, dim=0))

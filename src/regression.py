from __future__ import print_function

__author__ = 'Mark Hamilton'

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler
from tseries.tseries import DeltaTransformer, DoublePipeline, TimeSeriesRegressor
from tseries.tseries.models import LinearRegressionWithUncertainty
from utils import plot_coeffs
from tseries.tseries.utils import mse, names_to_indicies, one_step_prediction
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict
from itertools import product


def run_regression(dependent="tradepubs", delta=True, uncertainty=True, drop_dependent=True):
    data = pd.read_csv("../data/all_cols.csv")
    categorical_cols = ['disruptions', 'domesticpolitics']
    feature_cols = ['year', 'orgs', 'gdp', 'alltexts', 'population', 'disruptions',
                    'taxrevenue', 'councils', 'commerce', 'domesticpolitics', 'merchants', 'recharters', 'daysP']
    if dependent in {'econtexts', 'PubSubset', 'philosophy', 'politics'}:
        pass
    else:
        raise ValueError("not a proper dependent variable")

    Y = data[dependent]

    # Maybe drop dependent variable
    if not drop_dependent:
        feature_cols += [dependent]  # type: pd.DataFrame

    # Drop the cols to omit
    X = data[feature_cols]  # type: pd.DataFrame

    cols = X.columns.values

    if uncertainty:
        model = LinearRegressionWithUncertainty()
    else:
        model = LinearRegression()

    x_steps = [('imputer', Imputer()),
               ('scaler', StandardScaler(with_mean=False)),
               ('reg', TimeSeriesRegressor(model))]
    y_steps = [('yscale', StandardScaler(with_mean=False))]
    if delta:
        x_steps.insert(0, ("xdelta", DeltaTransformer(omit=names_to_indicies(X, categorical_cols))))
        y_steps.insert(0, ("ydelta", DeltaTransformer()))

    pipe = DoublePipeline(x_steps=x_steps, y_steps=y_steps)

    fit_model = pipe.fit(X, Y)
    Y_hat = np.squeeze(fit_model.predict(X))

    def get_ts_metrics(Y, Y_hat):
        Y_delta = DeltaTransformer().transform(Y)
        Y_hat_delta = DeltaTransformer().transform(Y_hat)
        Y_os = one_step_prediction(Y, Y)
        Y_hat_os = one_step_prediction(Y_hat, Y)

        metrics = {"mse_delta": mse(Y_delta, Y_hat_delta),
                   "r2_delta": r2_score(Y_delta, Y_hat_delta),
                   "mse_one_step": mse(Y_os, Y_hat_os),
                   "r2_one_step": r2_score(Y_os, Y_hat_os),
                   "feature_cols": ",".join(feature_cols)}

        fontsize = 18
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(Y_delta, label="$\Delta Y$")
        plt.plot(Y_hat_delta, label="$\Delta \hat{Y}$")
        plt.annotate("$ MSE(\Delta Y, \Delta \hat{Y}) = " + "{}$".format(round(metrics["mse_delta"], 2)),
                     xy=(0.35, 0.92), xycoords='axes fraction', fontsize=16)
        plt.annotate("$ r^{2}(\Delta Y, \Delta \hat{Y}) = " + "{}$".format(round(metrics["r2_delta"], 3)),
                     xy=(0.35, 0.85), xycoords='axes fraction', fontsize=16)
        plt.xlabel("$Time$", fontsize=fontsize)
        plt.legend(loc="upper left")
        plt.subplot(2, 2, 2)
        plt.scatter(Y_delta, Y_hat_delta)
        plt.ylabel("$\Delta \hat{Y}$", fontsize=fontsize)
        plt.xlabel("$\Delta Y$", fontsize=fontsize)
        plt.subplot(2, 2, 3)
        plt.plot(Y_os, label="$Y$")
        plt.plot(Y_hat_os, label="$\hat{Y}_1$")
        plt.annotate("$ MSE(Y, \hat{Y}_1) = " + "{}$".format(round(metrics["mse_one_step"], 2)),
                     xy=(0.35, 0.92), xycoords='axes fraction', fontsize=16)
        plt.annotate("$ r^{2}(Y, \hat{Y}_1) = " + "{}$".format(round(metrics["r2_one_step"], 3)),
                     xy=(0.35, 0.85), xycoords='axes fraction', fontsize=16)
        plt.xlabel("$Time$", fontsize=fontsize)
        plt.legend(loc="upper left")
        plt.subplot(2, 2, 4)
        plt.scatter(Y_os, Y_hat_os)
        plt.ylabel("$\hat{Y}_1$", fontsize=fontsize)
        plt.xlabel("$Y$", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig('../plots/regression/fit_dependent_{}_delta_{}.png'.format(dependent, delta))
        plt.clf()
        return metrics

    plt.subplot(1, 1, 1)
    print("Coefficients for dependent={} delta={}:".format(dependent, delta))
    coeff_df = plot_coeffs(model, cols, cutoff=0.0, sort=True, tick_angle=80)
    plt.tight_layout()
    plt.savefig('../plots/regression/coeffs_dependent_{}_delta_{}.png'.format(dependent, delta))
    plt.clf()

    return get_ts_metrics(Y[1:], Y_hat), coeff_df


param_grid = {
    "delta": [True, False],
    "dependent": ['econtexts', 'PubSubset', 'philosophy', 'politics']
}


def param_product(param_grid):
    param_lists = []
    for key, values in param_grid.iteritems():
        param_lists.append([(key, value) for value in values])
    return [dict(elem) for elem in product(*param_lists)]


param_dicts = param_product(param_grid)

coeff_dfs = []
results = defaultdict(list)

for param_dict in param_dicts:
    metrics, coeff_df = run_regression(**param_dict)
    for param, value in param_dict.items():
        coeff_df[param] = value
    coeff_dfs.append(coeff_df)

    metrics.update(param_dict)
    for k, v in metrics.items():
        results[k].append(v)

result_df = pd.DataFrame(results)
coeff_df = pd.concat(coeff_dfs)  # type: pd.DataFrame

result_df.to_csv("../plots/metrics_summary.csv")
coeff_df.to_csv("../plots/coeff_summary.csv")

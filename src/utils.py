import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model.base import LinearRegression
from sklearn.pipeline import Pipeline

from tseries.tseries import DoublePipeline, TimeSeriesRegressor
from tseries.tseries.models import BayesianLasso
from tseries.tseries.models import LinearRegressionWithUncertainty
import pandas as pd


def bar(values, tick_labels,
        lows=None, highs=None,
        sort_by_value=False, reverse=True, relative_error=True,
        width=0.4, tick_angle=80):
    indicies = np.arange(len(values))

    has_errors = lows is not None and highs is not None
    if has_errors:
        if relative_error:
            ctop = highs
            cbot = lows
        else:
            ctop = np.array(highs) - np.array(values)
            cbot = np.array(values) - np.array(lows)

        if sort_by_value:
            sorted_data = zip(*sorted(zip(values, cbot, ctop, tick_labels), reverse=reverse))
            values, cbot, ctop, tick_labels = sorted_data

        plt.bar(indicies, values, width, yerr=(cbot, ctop), color='r')
    else:
        if sort_by_value:
            sorted_data = zip(*sorted(zip(values, tick_labels), reverse=reverse))
            values, tick_labels = sorted_data

        plt.bar(indicies, values, width, color='r')

    plt.xticks(indicies + width, tick_labels, rotation=tick_angle)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)


def plot_coeffs(regressor, features, cutoff=0, sort=True, tick_angle=80):
    def extract_regressor(regressor):
        if isinstance(regressor, Pipeline):
            return extract_regressor(regressor.steps[-1][1])
        if isinstance(regressor, DoublePipeline):
            return extract_regressor(regressor.x_pipe_.steps[-1][1])
        if isinstance(regressor, TimeSeriesRegressor):
            return extract_regressor(regressor.base_estimator)
        else:
            return regressor

    regressor = extract_regressor(regressor)

    if isinstance(regressor, BayesianLasso):
        model_name = "Bayesian Lasso, b={}".format(regressor.b)
    elif isinstance(regressor, Lasso):
        model_name = "Standard Lasso, $\alpha$={}".format(regressor.alpha)
    elif isinstance(regressor, LinearRegression) \
            or isinstance(regressor, LinearRegressionWithUncertainty):
        model_name = "Linear Regression"
    else:
        raise ValueError("regressor not of proper type")

    coeffs = np.squeeze(regressor.coef_)

    if isinstance(regressor, BayesianLasso) \
            or isinstance(regressor, LinearRegressionWithUncertainty):
        lows, highs = regressor.confidence_intervals()
        lows, highs = np.squeeze(lows), np.squeeze(highs)
        data = zip(coeffs, lows, highs, features)
        if cutoff is not None:
            coeffs, lows, highs, features = zip(*[p for p in data if abs(p[0]) > cutoff])

    else:
        lows, highs = None, None
        data = zip(coeffs, features)
        if cutoff is not None:
            coeffs, features = zip(*[p for p in data if abs(p[0]) > cutoff])


    # add some text for labels, title and axes ticks
    plt.ylabel('Coefficient', fontsize=20)
    plt.title('Coefficients of {}'.format(model_name), fontsize=20)
    bar(coeffs, features, lows, highs, relative_error=False,
        sort_by_value=sort, tick_angle=tick_angle)

    coeff_df = pd.DataFrame(sorted(data, key=lambda t: t[1]),
                            columns=["coeff", "lower_limit", "upper_limit", "feature"])
    return coeff_df


if __name__ == "main":
    bar(np.linspace(1, 0, 5),
        ["foo" + str(i) for i in range(5)],
        .2 * np.linspace(0, 1, 5) ** 2,
        .4 * np.linspace(0, 1, 5) ** 3,
        sort_by_value=True)
    plt.show()

from seaborn import pairplot
from tseries.tseries import DeltaTransformer
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


data = pd.read_csv("../data/all_cols.csv")

data_transformed = DeltaTransformer().transform(data)

axes = scatter_matrix(data, alpha=0.5, figsize=(30, 30))
corr = data.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" % corr[i, j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.savefig("../plots/pairplot/pairplot.png")
plt.clf()

axes = scatter_matrix(data_transformed, alpha=0.5, figsize=(30, 30))
corr = data.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" % corr[i, j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.savefig("../plots/pairplot/pairplot_transformed.png")
plt.clf()

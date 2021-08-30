import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import crystalball
import uproot
import numpy as np
import math

input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

file2 = uproot.open("reco-Delta40-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]



result = pd.concat(dframes)


def function1(x, alpha, n, mu, sigma):
    expArg = -0.5 * pow(abs(alpha), 2.)
    gauss = math.exp(expArg)

    A = pow((n / abs(alpha)), n) * gauss
    B = (n / abs(alpha)) - abs(alpha)
    C = n / ((abs(alpha) * (n - 1.))) * gauss
    D = math.sqrt(math.pi / 2.) * (1. + math.erf((abs(alpha) / math.sqrt(2.))))
    N = 1. / (sigma * (C + D))

    return (N * math.exp(-1 * (pow((x - mu), 2) / (2 * pow(sigma, 2)))))



def function2(x, alpha, n, mu, sigma):
    expArg = -0.5 * pow(abs(alpha), 2.)
    gauss = math.exp(expArg)

    A = pow((n / abs(alpha)), n) * gauss
    B = (n / abs(alpha)) - abs(alpha)
    C = n / ((abs(alpha) * (n - 1.))) * gauss
    D = math.sqrt(math.pi / 2.) * (1. + math.erf((abs(alpha) / math.sqrt(2.))))
    N = 1. / (sigma * (C + D))

    return (N * A * pow((B - (x - mu) / sigma), -n))


def piecewise(x, alpha, n, mu, sigma):
    return np.piecewise(x,[(x - mu) / sigma > -alpha,(x - mu) / sigma >= -alpha ],[function1(x, alpha, n, mu, sigma), function2(x, alpha, n, mu, sigma)])




data1 = df["deent.mom"]
data = result["deent.mom"]

y, bins = np.histogram(data1, bins=200);

# Convert histogram into a classical plot
dx = bins[1]-bins[0]
x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 200)

par1 = [5,2,48,5]
#par1 = [alpha, n, mu, sigma, scale]

#par1 = [5,2,48,5,85000]





plt.hist(data1, bins=100, label='data')
popt,pcov = curve_fit(piecewise, x,y, p0 = [*par1])
plt.plot(x, piecewise(x,*par1), )
plt.plot(x, piecewise(x,*popt), ':r')
plt.show()

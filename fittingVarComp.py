import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import crystalball
import uproot


# defining functions
def newGaussFit(x1, ampl1, mu1, sigma1, ampl2, mu2, sigma2):
    return (ampl1 * np.exp(-(x1 - mu1) ** 2 / 2. / sigma1 ** 2) + (ampl2 * np.exp(-(x1 - mu2) ** 2 / 2. / sigma2 ** 2)))
def Gauss(x, ampl, mu, sigma):
    return ampl * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)


# class def for fixed mu1 and mu2
class GaussClass:

    def __init__(self):
        pass

    def Gaussalt(self, x1, ampl1, sigma1, ampl2, sigma2):
        return (ampl1 * np.exp(-(x1 - self.mu1) ** 2 / 2. / sigma1 ** 2)) + (
                    ampl2 * np.exp(-(x1 - self.mu2) ** 2 / 2. / sigma2 ** 2))


# data
input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten=False)

file2 = uproot.open("reco-Delta50-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]

result = pd.concat(dframes)

# Define your PDF / model ; all the parameters besides the first one will be fitted


data = df2["deent.mom"]

y, bins = np.histogram(data, bins=100);

# Convert histogram into a classical plot
dx = bins[1] - bins[0]
x = np.linspace(bins[0] + dx / 2, bins[-1] - dx / 2, 100)

# Alternative way starts at this point
# Initial guess for the fit
inst = GaussClass()
inst.mu1 = 47.2  # this is the parameter that will stay fixed during the fitting process
inst.mu2 = 40
ampl1 = 6648
mu1 = 48
sigma1 = 4
ampl2 = 200
# mu2 = 45
sigma2 = 0.6

fig = plt.figure()
ax = fig.add_subplot()

# Fit the data to the gaussian pdf
plt.hist(data, bins=100, label='data')
# popt, pcov = curve_fit(inst.Gaussalt, x, y, p0=[ampl1, sigma1, ampl2, sigma2])
popt, pcov = curve_fit(Gauss, x, y, p0=[ampl1, mu1,sigma1])
print(len(popt))
plt.plot(x, Gauss(x, *popt), ':r', label='fit')
print (type(Gauss(x,*popt)))
plt.text(0.7, 0.8, r'amplitude1 = ' + str(round(popt[0], 1)), transform=ax.transAxes)
# plt.text(0.7, 0.75, r'$\mu1 = $' + str(round(popt[1], 1)), transform=ax.transAxes)
plt.text(0.7, 0.75, r'$\sigma1 = $' + str(round(popt[1], 1)), transform=ax.transAxes)
# plt.text(0.7, 0.7, r'amplitude2 = ' + str(round(popt[2], 1)), transform=ax.transAxes)
# # # plt.text(0.7, 0.6, r'$\mu2 = $' + str(round(popt[4], 4)), transform=ax.transAxes)
# plt.text(0.7, 0.65, r'$\sigma2 = $' + str(round(popt[3], 4)), transform=ax.transAxes)

plt.show()

calculated = []
real = []

calculated.append(popt[0])
calculated.append(popt[1])
calculated.append(popt[2])
calculated.append(popt[3])
calculated.append(7210.8)
calculated.append(4.1)
calculated.append(1284.3)
calculated.append(1.0344)
calculated.append(6687.1)
calculated.append(4.5)
calculated.append(3349.2)
calculated.append(1.1396)
calculated.append(9776.9)
calculated.append(3.0)
calculated.append(8549.2)
calculated.append(1.146)
print(popt)
real.append(6648.3)
real.append(47.5)
real.append(86.6)
real.append(34.5)
real.append(1526.1)
real.append(39.4)
real.append(5832.7)
real.append(44.3)
real.append(12832.9)
real.append(49.2)










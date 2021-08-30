import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import crystalball
# import uproot
#
# #defining functions
# def newGaussFit(x1, ampl1, mu1, sigma1, ampl2, mu2, sigma2):
#     return (ampl1 * np.exp(-(x1 - mu1) ** 2 / 2. / sigma1 ** 2) + (ampl2 * np.exp(-(x1 - mu2) ** 2 / 2. / sigma2 ** 2)))
#
# def CrystalBall(beta1,m1,beta2,m2):
#     return crystalball(beta1,m1) + crystalball(beta2,m2)
#
# #class def for fixed mu1 and mu2
# class GaussClass:
#
#     def __init__(self):
#         pass
#
#     def Gaussalt(self, x1, ampl1, sigma1,ampl2, sigma2 ):
#         return (ampl1 * np.exp(-(x1 - self.mu1) ** 2 / 2. / sigma1 ** 2)) + (ampl2 * np.exp(-(x1 - self.mu2) ** 2 / 2. / sigma2 ** 2))
#
# #data
# input_file = uproot.open("trkana.Triggered.root")
# input_tree = input_file["TrkAnaNeg/trkana"]
# df = input_tree.pandas.df(flatten = False)import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import crystalball
import uproot

#defining functions
# def newGaussFit(x1, ampl1, mu1, sigma1, ampl2, mu2, sigma2):
#     return (ampl1 * np.exp(-(x1 - mu1) ** 2 / 2. / sigma1 ** 2) + (ampl2 * np.exp(-(x1 - mu2) ** 2 / 2. / sigma2 ** 2)))

def CrystalBall(x,beta1,m1,loc1,scale1,beta2,m2,loc2,scale2):
    return (crystalball.pdf(x,beta1,m1,loc1,scale1) + crystalball.pdf(x,beta2,m2,loc2,scale2))

#class def for fixed mu1 and mu2
# class GaussClass:
#
#     def __init__(self):
#         pass
#
#     def Gaussalt(self, x1, ampl1, sigma1,ampl2, sigma2 ):
#         return (ampl1 * np.exp(-(x1 - self.mu1) ** 2 / 2. / sigma1 ** 2)) + (ampl2 * np.exp(-(x1 - self.mu2) ** 2 / 2. / sigma2 ** 2))

#data
input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

file2 = uproot.open("reco-Delta45-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]

result = pd.concat(dframes)


# Define your PDF / model ; all the parameters besides the first one will be fitted


data1 = result["deent.mom"]
data = df2["deent.mom"]

y, bins = np.histogram(data, bins=100);

beta1 = 45
m1 = 2

# Convert histogram into a classical plot
dx = bins[1] - bins[0]
x = np.linspace(crystalball.ppf(0.01, beta1, m1),
                crystalball.ppf(0.99, beta1, m1), 100)

#x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 100)

# Alternative way starts at this point
#Initial guess for the fit
# inst = GaussClass()
# inst.mu1 = 47.2    # this is the parameter that will stay fixed during the fitting process
# inst.mu2 = 35
ampl1 = 50
#mu1 = 48
sigma1 = 2
ampl2 = 36
#mu2 = 45
sigma2 = 4



fig = plt.figure()
ax = fig.add_subplot()


# Fit the data to the gaussian pdf
plt.hist(data, bins=100, label='data')
#popt,pcov = curve_fit(inst.Gaussalt, x, y, p0=[ampl1,sigma1,ampl2,sigma2])
popt,pcov = curve_fit(CrystalBall, x,y, p0 = [beta1,m1])
# print (len(popt))
#plt.plot(x, inst.Gaussalt(x, *popt), ':r', label='fit')
plt.plot(x, CrystalBall(x,*popt), ':r', label='fit')
print(popt)
# plt.text(0.7, 0.8, r'amplitude1 = ' + str(round(popt[0], 1)), transform=ax.transAxes)
# # plt.text(0.7, 0.75, r'$\mu1 = $' + str(round(popt[1], 1)), transform=ax.transAxes)
# plt.text(0.7, 0.75, r'$\sigma1 = $' + str(round(popt[1], 1)), transform=ax.transAxes)
# plt.text(0.7, 0.7, r'amplitude2 = ' + str(round(popt[2], 1)), transform=ax.transAxes)
# # plt.text(0.7, 0.6, r'$\mu2 = $' + str(round(popt[4], 4)), transform=ax.transAxes)
# plt.text(0.7, 0.65, r'$\sigma2 = $' + str(round(popt[3], 4)), transform=ax.transAxes)
beta, m = 45, 80
ax.plot(x, CrystalBall(beta,m),
       'r-', lw=5, alpha=0.6, label='crystalball pdf')

plt.show()




# file2 = uproot.open("reco-Delta45-trig.root")
# RPCReco2 = file2["TrkAnaNeg/trkana"]
# df2 = RPCReco2.pandas.df(flatten=False)
#
# dframes = [df, df2]
#
# result = pd.concat(dframes)
#
#
# # Define your PDF / model ; all the parameters besides the first one will be fitted
#
#
#
# data = result["deent.mom"]
#
# y, bins = np.histogram(data, bins=100);
#
#
# # Convert histogram into a classical plot
# dx = bins[1] - bins[0]
# x = np.linspace(bins[0] + dx / 2, bins[-1] - dx / 2, 100)
#
#
# # Alternative way starts at this point
# #Initial guess for the fit
# inst = GaussClass()
# inst.mu1 = 47.2     # this is the parameter that will stay fixed during the fitting process
# inst.mu2 = 35
# ampl1 = 6648
# #mu1 = 48
# sigma1 = 4
# ampl2 = 200
# #mu2 = 45
# sigma2 = 0.6
#
#
# fig = plt.figure()
# ax = fig.add_subplot()
#
#
# # Fit the data to the gaussian pdf
# plt.hist(data, bins=100, label='data')
# popt,pcov = curve_fit(inst.Gaussalt, x, y, p0=[ampl1,sigma1,ampl2,sigma2])
# print (len(popt))
# plt.plot(x, inst.Gaussalt(x, *popt), ':r', label='fit')
# plt.text(0.7, 0.8, r'amplitude1 = ' + str(round(popt[0], 1)), transform=ax.transAxes)
# # plt.text(0.7, 0.75, r'$\mu1 = $' + str(round(popt[1], 1)), transform=ax.transAxes)
# plt.text(0.7, 0.75, r'$\sigma1 = $' + str(round(popt[1], 1)), transform=ax.transAxes)
# plt.text(0.7, 0.7, r'amplitude2 = ' + str(round(popt[2], 1)), transform=ax.transAxes)
# # plt.text(0.7, 0.6, r'$\mu2 = $' + str(round(popt[4], 4)), transform=ax.transAxes)
# plt.text(0.7, 0.65, r'$\sigma2 = $' + str(round(popt[3], 4)), transform=ax.transAxes)
#
#
# plt.show()
#
#

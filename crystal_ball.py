import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import crystalball
import uproot
import numpy as np

input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

file2 = uproot.open("reco-Delta40-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]

result = pd.concat(dframes)

data = result["deent.mom"]

def CrystalBall(x,beta1,m1,beta2,m2,loc1,scale1,loc2,scale2):
    return (crystalball.pdf(x,beta1,m1,loc1,scale1) + crystalball.pdf(x,beta2,m2,loc2,scale2))

def Gauss(x, ampl, mu, sigma):
    return ampl * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)




y, bins = np.histogram(data, bins=100);

beta1 = 57
m1 = 2
loc1 = 50
scale1 = 4
beta2 = 40
m2 = 2
loc2 = 40
scale2 = 2000

# Convert histogram into a classical plot
dx = bins[1] - bins[0]
x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 100)


plt.hist(data, bins=100, label='data')
popt,pcov = curve_fit(CrystalBall, x,y, p0 = [beta1,m1,loc1,scale1,beta2,m2,loc2,scale2])
print(popt)
plt.plot(x, CrystalBall(x,*popt), ':r', label='fit')
plt.show()
print(popt)
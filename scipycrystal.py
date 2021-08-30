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

data = df["deent.mom"]

def CrystalBall(x,beta1,m1,beta2,m2,loc1,scale1,loc2,scale2):
    return (crystalball.pdf(x,beta1,m1,loc1,scale1) + crystalball.pdf(x,beta2,m2,loc2,scale2))


def singleCrystalBall(x,beta1,m1,loc1,scale1):
    return (crystalball.pdf(x,beta1,m1,loc1,scale1) )

y, bins = np.histogram(data, bins=200);



# Convert histogram into a classical plot
dx = bins[1] - bins[0]
x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 200)

hello = crystalball.fit(y)

beta1 = 4.2
m1 = 3
loc1 = 47
scale1 = .0000831
# plt.hist(data, bins=200, label='data')
# popt,pcov = curve_fit(singleCrystalBall, x,y, p0 = [beta1,m1,loc1,scale1])
# print(popt)
# plt.plot(x, singleCrystalBall(x,*popt), ':r', label='fit')
plt.plot(x, singleCrystalBall(x,*hello), ':r', label='fit')
plt.show()
# print(popt)
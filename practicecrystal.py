from scipy.stats import crystalball
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

file2 = uproot.open("reco-Delta45-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]

result = pd.concat(dframes)

data1 = result["deent.mom"]
data = df2["deent.mom"]

y, bins = np.histogram(data, bins=100);

beta1 = 45
m1 = 2

# Convert histogram into a classical plot
dx = bins[1] - bins[0]

x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 100)

beta, m,loc,scale = 1, 10,2,0.05
#loc is how many units shifted to the left the thing is
#scale is height of peak- divided by the scale
#scale can't be negative


def CrystalBall(x,beta1,m1,loc,scale):#,beta2,m2):
    return (crystalball.pdf(x,beta1,m1,loc,scale)) #+ crystalball.pdf(x,beta2,m2)

x = np.linspace(crystalball.ppf(0.01, beta, m),
                crystalball.ppf(0.99, beta, m), 100)
ax.plot(x, CrystalBall(x,beta, m,loc,scale))
plt.xlim(-10,10)
plt.ylim(-10,10)

plt.xlim(-10,10)

plt.show()
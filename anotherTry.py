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

file2 = uproot.open("reco-Delta35-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]



result = pd.concat(dframes)



def _crystalballPositiveAlpha(x, alpha, n, mu, sigma):


    expArg = -0.5 *pow(abs(alpha),2.)
    gauss = math.exp(expArg)

    A = pow((n / abs(alpha)), n) * gauss
    B = (n / abs(alpha)) - abs(alpha)
    C = n / ((abs(alpha) * (n - 1.))) * gauss
    D = math.sqrt(math.pi / 2.) * (1. + math.erf((abs(alpha) / math.sqrt(2.))))
    N = 1. / (sigma * (C + D))
    #pull = (x-mu)/sigma


    if (x - mu) / sigma > -alpha:
            return( N * math.exp(-1*(pow((x-mu),2)/(2*pow(sigma,2)))))

    else:
            return (N * A * pow((B - (x - mu) / sigma), -n))






def _crystalball(x, alpha, n, mu, sigma, scale):
    if alpha > 0.:
        return scale * _crystalballPositiveAlpha(x, alpha, n, mu, sigma)
    else:
        x1 = 2 * mu - x
        alpha1 = -alpha
        return scale * _crystalballPositiveAlpha(x1, alpha1, n, mu, sigma)

def crystalball(x, alpha,n, mu, sigma,scale):
    returning = []

    # return(_crystalball(x, alpha, n, mu, sigma, scale))
    for i in range(len(x)):
        returning.append(_crystalball(x[i], alpha, n, mu, sigma, scale))
    return returning


def stackedCrystal(x,par1,par2):
    return crystalball(x,*par1) + crystalball(x,*par2)


data1 = df2["deent.mom"]
data = result["deent.mom"]

y, bins = np.histogram(data1, bins=200);

# Convert histogram into a classical plot
dx = bins[1]-bins[0]
x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 200)

# par1 = [5,2,48,5,85000]
par1 = [-1.3,2,48,3.7,34000]
# mu is where switch occurs
# scale is y scale factor
# sigma is how wide
# negative sigma flips which side the power is on
# par2 = [5,2,48,5,85000]
#
# par1 = [alpha, n, mu, sigma, scale]




fig = plt.figure()
ax = fig.add_subplot()

plt.hist(data1, bins=200, label='data')
popt,pcov = curve_fit(crystalball, x,y, p0 = [*par1])
#plt.plot(x, crystalball(x,*par1), )
plt.plot(x, crystalball(x,*popt), ':r')

plt.text(0.8, 0.8, r'$\alpha = $' + str(round(popt[0], 2)), transform=ax.transAxes)
plt.text(0.8, 0.75, r'n =' + str(round(popt[1], 2)), transform=ax.transAxes)
plt.text(0.8, 0.7, r'$\mu = $' + str(round(popt[2], 2)), transform=ax.transAxes)
plt.text(0.8, 0.65, r'$\sigma = $' + str(round(popt[3], 2)), transform=ax.transAxes)
plt.text(0.8, 0.6, r's = ' + str(round(popt[4], 0)), transform=ax.transAxes)
plt.legend()
plt.show()

#
# plt.hist(data, bins=100, label='data')
# popt,pcov = curve_fit(stackedCrystal, x,y, p0 = [par1,par2])
# plt.plot(x, stackedCrystal(x,par1,par2))
# plt.plot(x, stackedCrystal(x,*popt), ':r')
# plt.show()

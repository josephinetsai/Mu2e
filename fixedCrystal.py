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


# class def for fixed mu1 and mu2
class CrystalClass:

    def __init__(self):
        pass

    def _crystalballPositiveAlpha( x, alpha, n,mu, sigma):


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


    def _crystalball(x, alpha, n,mu, sigma, scale):
        if alpha > 0.:
            return scale *CrystalClass._crystalballPositiveAlpha(x, alpha, n, mu,sigma)
        else:
            x1 = 2 * mu - x
            alpha1 = -alpha
            return scale * CrystalClass._crystalballPositiveAlpha( x1, alpha1, n,mu, sigma)

    def crystalball(x, alpha,n,mu,sigma,scale):
        returning = []

        # return(_crystalball(x, alpha, n, mu, sigma, scale))
        for i in range(len(x)):
            returning.append(CrystalClass._crystalball(x[i], alpha, n,mu, sigma, scale))
        return returning


    def stackedCrystal(self,x,alpha1,n1,sigma1,scale1,alpha2,n2,sigma2,scale2):
        first = CrystalClass.crystalball(x,alpha1,n1,self.mu1,sigma1,scale1)
        second = CrystalClass.crystalball(x,alpha2,n2,self.mu2,sigma2,scale2)
        returning = []
        for i in range(len(second)):
            returning.append(first[i] + second[i])
        return returning



data1 = df["deent.mom"]#only background
data2 = df2["deent.mom"]#only signal
data = result["deent.mom"]

y, bins = np.histogram(data, bins=200);

# Convert histogram into a classical plot
dx = bins[1]-bins[0]
x = np.linspace(bins[0]+dx/2, bins[-1]-dx/2, 200)

# par1 = [5,2,48,5,85000]
parback = [-3.31,1.91,47.53,4.08,34069]
par35 = [-2.29,0.98,34.52,0.54,10]#run with fixed for this one
par40 = [-2.73,0.99,39.5,0.6,600]
par45 = [-3.02,1.07,44,0.74,5645]
par50 = [-3.13,0.97,49,0.73,11814]



# mu is where switch occurs
# scale is y scale factor
# sigma is how wide
# negative sigma flips which side the power is on
# par2 = [5,2,48,5,85000]
#
# par1 = [alpha, n, mu, sigma, scale]




fig = plt.figure()
ax = fig.add_subplot()

# plt.hist(data1, bins=200, label='data')
# popt,pcov = curve_fit(crystalball, x,y, p0 = [*parbackground])
# #plt.plot(x, crystalball(x,*par1), )
# plt.plot(x, crystalball(x,*popt), ':r', label = 'fit')
#
# plt.text(0.8, 0.8, r'$\alpha = $' + str(round(popt[0], 2)), transform=ax.transAxes)
# plt.text(0.8, 0.75, r'n =' + str(round(popt[1], 2)), transform=ax.transAxes)
# plt.text(0.8, 0.7, r'$\mu = $' + str(round(popt[2], 2)), transform=ax.transAxes)
# plt.text(0.8, 0.65, r'$\sigma = $' + str(round(popt[3], 2)), transform=ax.transAxes)
# plt.text(0.8, 0.6, r's = ' + str(round(popt[4], 0)), transform=ax.transAxes)
# plt.legend()
# plt.show()

inst = CrystalClass()
inst.mu1 = 47.2
inst.mu2 = 35

plt.hist(data, bins=200, label='data')
popt,pcov = curve_fit(inst.stackedCrystal, x,y, p0 = [-3.31,1.91,4.08,34069,-2.29,0.98,0.54,10 ])
# plt.plot(x, crystalball(x,*parback))
# plt.plot(x, crystalball(x,*par40))
# plt.plot(x, stackedCrystal(x,*parback,*par40))
plt.plot(x, inst.stackedCrystal(x,*popt), ':r', label = 'fit')

#
plt.text(0.8, 0.8, r'$\alpha1 = $' + str(round(popt[0], 2)), transform=ax.transAxes)
plt.text(0.8, 0.75, r'n1 =' + str(round(popt[1], 2)), transform=ax.transAxes)
# plt.text(0.8, 0.7, r'$\mu1 = $' + str(round(popt[2], 2)), transform=ax.transAxes)
plt.text(0.8, 0.65, r'$\sigma1 = $' + str(round(popt[2], 2)), transform=ax.transAxes)
plt.text(0.8, 0.6, r's1 = ' + str(round(popt[3], 0)), transform=ax.transAxes)
plt.text(0.8, 0.55, r'$\alpha2 = $' + str(round(popt[4], 2)), transform=ax.transAxes)
plt.text(0.8, 0.5, r'n2 =' + str(round(popt[5], 2)), transform=ax.transAxes)
# plt.text(0.8, 0.45, r'$\mu2 = $' + str(round(popt[7], 2)), transform=ax.transAxes)
plt.text(0.8, 0.4, r'$\sigma2 = $' + str(round(popt[6], 2)), transform=ax.transAxes)
plt.text(0.8, 0.35, r's2 = ' + str(round(popt[7], 0)), transform=ax.transAxes)
plt.legend()
plt.show()
plt.show()

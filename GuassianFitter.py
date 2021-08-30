import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uproot



input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

file2 = uproot.open("reco-Delta40-trig.root")
RPCReco2 = file2["TrkAnaNeg/trkana"]
df2 = RPCReco2.pandas.df(flatten=False)

dframes = [df, df2]

result = pd.concat(dframes)
fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(result["deent.mom"],
                           bins=1000,
                           range=(0,100),
                           label="Generated Spectrum")


fig.savefig("COMBINED SIGNAL + BACKGROUND")

# Define your PDF / model ; all the parameters besides the first one will be fitted
def Gauss(x, ampl, mu, sigma):
    return ampl * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def GaussFit(x1, ampl1, mu1, sigma1):
    returning = []
    for i in range(len(x1)):
        returning.append(ampl1 * np.exp(-(x1[i] - mu1) ** 2 / 2. / sigma1 ** 2))

    return returning



ind = 0
ampleDiff = 0

data = df["deent.mom"]

y, bins = np.histogram(data, bins=100);


# Convert histogram into a classical plot
dx = bins[1] - bins[0]
x = np.linspace(bins[0] + dx / 2, bins[-1] - dx / 2, 100)


# Alternative way starts at this point
#Initial guess for the fit
# ampl = 6500
# mu = 48
# sigma = 0.2


#popt4,pcov4 = curve_fit(GaussFit, x, y,p0=[ampl,mu,sigma])
popt4,pcov4 = curve_fit(GaussFit, x, y)
# Fit the data to the gaussian pdf
# popt, pcov = curve_fit(Gauss, x, y,p0=[ampl,mu,sigma])

# Plots
fig = plt.figure()
ax = fig.add_subplot()
plt.hist(data, bins=200, label='data')
# plt.plot(x, y, '-b', label='data')
plt.plot(x, GaussFit(x, *popt4), ':r', label='fit')


plt.text(0.85, 0.8, r'amplitude = ' + str(round(popt4[0], 4)), transform=ax.transAxes)
plt.text(0.85, 0.75, r'$\mu = $' + str(round(popt4[1], 4)), transform=ax.transAxes)
plt.text(0.85, 0.7, r'$\sigma = $' + str(round(popt4[2], 4)), transform=ax.transAxes)
plt.legend()
plt.show()




#second fitting

# data1 = df2["deent.mom"]
#
# y1, bins1 = np.histogram(data1, bins=100);
#
#
# # Convert histogram into a classical plot
# dx = bins1[1] - bins1[0]
# x1 = np.linspace(bins1[0] + dx / 2, bins1[-1] - dx / 2, 100)


## Alternative way starts at this point
# Initial guess for the fit
# ampl = 3000
# mu = 40
# sigma = 0.2
#
# # Fit the data to the gaussian pdf
# popt1, pcov1 = curve_fit(Gauss, x1, y1,p0=[ampl,mu,sigma])
# #gives values of parameters ampl,mu,sigma
#
# list1 = Gauss(x, *popt).tolist()
# list2 = Gauss(x, *popt1).tolist()
# # Plots
# fig = plt.figure()
# ax = fig.add_subplot()
#
# # plt.plot(x, y, '-b', label='data')
#
#
# plt.text(0.85, 0.8, r'amplitude2 = ' + str(round(popt[0], 4)), transform=ax.transAxes)
# plt.text(0.85, 0.75, r'$\mu2 = $' + str(round(popt[1], 4)), transform=ax.transAxes)
# plt.text(0.85, 0.7, r'$\sigma2 = $' + str(round(popt[2], 4)), transform=ax.transAxes)
# plt.text(0.85, 0.65, r'amplitude1 = ' + str(round(popt1[0], 4)), transform=ax.transAxes)
# plt.text(0.85, 0.6, r'$\mu1 = $' + str(round(popt1[1], 4)), transform=ax.transAxes)
# plt.text(0.85, 0.55, r'$\sigma1 = $' + str(round(popt1[2], 4)), transform=ax.transAxes)
# plt.legend()
# plt.show()
#
#
#
# #  d = f - g
# #  for i in range(len(d) - 1):
# #      if d[i] == 0. or d[i] * d[i + 1] < 0.:
# # #         # crossover at i
# # #         x_ = x[i]
#
# # idx = np.argwhere(np.diff(np.sign((Gauss(x1, *popt1)) - (Gauss(x, *popt)))))
# # print("hello")
# # print(idx)
#
# #print(idx)
# print(type((Gauss(x, *popt1))))
# print(len((Gauss(x, *popt1))))
#
# print(type((Gauss(x, *popt))))
# print(len((Gauss(x, *popt1))))
#
#
#
# prevdiff = (list1[0]) - list2[0]
# hello = []
#
# def sign(x):
#     if(x>=0):
#         return 1
#     else:
#         return -1
# difference = []
#
# # plt.plot(x1[idx], Gauss(x1, *popt1)[idx], 'ro')
#
# print("hi there")
# print(hello)
# print(difference)
#
#
#
#
#
# # print(hello)#before was 25 and 27
# # for i in range(len(hello)):
# #     plt.plot(x[hello[i]], Gauss(x, popt *)[hello[i]], 'ro')
#
# y_overlap = []
#
# # for i in range(100):
# #     if(list1[i] == list2[i]):
# #         print(i)
# #         y_overlap.append(list1[i])
#
# # print(y_overlap)
#
# #plt.plot(x1, Gauss(x1, *popt1), ':r', label='fit')
# #plt.plot(x, Gauss(x, *popt),':r', label='fit')
# # np.piecewise(x, [x < 25, x >=25 & x<=27,x > 27], [Gauss(x, *popt),Gauss(x1, *popt1)],Gauss(x, *popt))
#
#
#
#
# for i in range(len(list1)-1):
#     i += 1
#     one = list2[i]#is type float
#     two = list1[i]
#
#
#     diff = int(list1[i] - list2[i])
#     difference.append(diff)
#
#     if (sign(diff) != sign(prevdiff)):
#         hello.append(i)
#     i -= 1
#     prevdiff = diff
#
# def GaussFit(b1,b2,x1, ampl1, mu1, sigma1, x2, ampl2, mu2, sigma2):
#     returning = []
#
#     for i in range(b1):
#         returning.append(ampl1 * np.exp(-(x1[i] - mu1) ** 2 / 2. / sigma1 ** 2))
#     for i in range(b2-b1):
#         i = i + b1
#         # if(ampl2 * np.exp(-(x1[i] - mu2) ** 2 / 2. / sigma2 ** 2) >= mu2 -1 & ampl2 * np.exp(-(x1[i] - mu2) ** 2 / 2. / sigma2 ** 2) <= mu2 +1):
#         #     ind = i
#         returning.append(ampl2 * np.exp(-(x1[i] - mu2) ** 2 / 2. / sigma2 ** 2))
#
#     for i in range(len(x1)-b2):
#         i = b2 + i
#         returning.append(ampl1 * np.exp(-(x1[i] - mu1) ** 2 / 2. / sigma1 ** 2))
#     # ampleDiff = ampl2- (ampl1 * np.exp(-(x1[ind] - mu1) ** 2 / 2. / sigma1 ** 2))
#     return returning
#
# def newGaussFit(b1,b2,x1, ampl1, mu1, sigma1, x2, ampl2, mu2, sigma2):
#     returning = []
#
#     for i in range(len(x1)):
#         returning.append(ampl1 * np.exp(-(x1[i] - mu1) ** 2 / 2. / sigma1 ** 2) +ampl2 * np.exp(-(x1[i] - mu2) ** 2 / 2. / sigma2 ** 2) )
#
#     return returning
#
# #plt.plot(x1, Gauss(x1, *popt1), ':r', label='fit')
# #plt.plot(x, Gauss(x, *popt),':r', label='fit')
# #plt.plot(x,GaussFit(*hello,x, *popt, x1, *popt1), label = 'fit')
# data3 = result["deent.mom"]
# plt.hist(data3, bins=100, label='data')
#
# # plt.text(0.85, 0.8, r'Delta Peak height' + str(ampleDiff), transform=ax.transAxes)
#
# plt.show()

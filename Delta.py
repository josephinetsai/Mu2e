import uproot
import matplotlib.pyplot as plt
import math
import numpy as np

file1 = uproot.open("GenDelta_35.root")
file2 = uproot.open("GenDelta_40.root")
file3 = uproot.open("GenDelta_45.root")
file4 = uproot.open("GenDelta_50.root")

RPCReco1 = file1["generate/GenTree"] # opens the 'GenTree' tree in the 'generate' folder
RPCReco2 = file2["generate/GenTree"]
RPCReco3 = file3["generate/GenTree"]
RPCReco4 = file4["generate/GenTree"]

df1 = RPCReco1.pandas.df(flatten=False)
df2 = RPCReco2.pandas.df(flatten=False)
df3 = RPCReco3.pandas.df(flatten=False)
df4 = RPCReco4.pandas.df(flatten=False)

a = []
b = []

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df1["pmag_gen"],
                           bins=2,
                           range=(34,36),
                           label="Generated Spectrum")
tot = 0
for i in range(2):
    tot += n[i]

a.append(tot)

std_mcmom = round(np.std(df1["pmag_gen"]),2)
mean_mcmom = round(np.mean(df1["pmag_gen"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=8)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("IPA_Delta_Gen.pdf")



fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df2["pmag_gen"],
                           bins=20,
                           range=(30,50),
                           label="Generated Spectrum")
tot = 0
for i in range(20):
    tot += n[i]

a.append(tot)
fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df3["pmag_gen"],
                           bins=20,
                           range=(35,55),
                           label="Generated Spectrum")
tot = 0
for i in range(20):
    tot += n[i]

a.append(tot)
fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df4["pmag_gen"],
                           bins=20,
                           range=(40,60),
                           label="Generated Spectrum")
tot = 0
for i in range(20):
    tot += n[i]

a.append(tot)

print(a)

file1 = uproot.open("reco-Delta35-trig.root")
file2 = uproot.open("reco-Delta40-trig.root")
file3 = uproot.open("reco-Delta45-trig.root")
file4 = uproot.open("reco-Delta50-trig.root")

RPCReco1 = file1["TrkAnaNeg/trkana"] # opens the 'GenTree' tree in the 'generate' folder
RPCReco2 = file2["TrkAnaNeg/trkana"]
RPCReco3 = file3["TrkAnaNeg/trkana"]
RPCReco4 = file4["TrkAnaNeg/trkana"]
df1 = RPCReco1.pandas.df(flatten=False)
df2 = RPCReco2.pandas.df(flatten=False)
df3 = RPCReco3.pandas.df(flatten=False)
df4 = RPCReco4.pandas.df(flatten=False)

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df1["deent.mom"],
                           bins=20,
                           range=(25,45),
                           label="Generated Spectrum")

tot = 0
for i in range(20):
    tot += n[i]

b.append(tot)

std_mcmom = round(np.std(df1["deent.mom"]),2)

mean_mcmom = round(np.mean(df1["deent.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(0.52,0.81,"STD: " + str(std_mcmom), fontsize=8)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("deent_mom.Delta_Gen35.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df1["demcgen.mom"],
                           bins=20,
                           range=(25,45),
                           label="Generated Spectrum")

std_mcmom = round(np.std(df1["demcgen.mom"]),2)
mean_mcmom = round(np.mean(df1["demcgen.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=40)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("demcgen_mom.Delta_Gen35.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df2["deent.mom"],
                           bins=20,
                           range=(30,50),
                           label="Generated Spectrum")
tot = 0
for i in range(20):
    tot += n[i]

b.append(tot)

std_mcmom = round(np.std(df2["deent.mom"]),2)

mean_mcmom = round(np.mean(df2["deent.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(0.52,0.81,"STD: " + str(std_mcmom), fontsize=8)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("deent_mom.Delta_Gen40.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df2["demcgen.mom"],
                           bins=20,
                           range=(30,50),
                           label="Generated Spectrum")

std_mcmom = round(np.std(df2["demcgen.mom"]),2)
mean_mcmom = round(np.mean(df2["demcgen.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=40)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("demcgen_mom.Delta_Gen40.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df3["deent.mom"],
                           bins=20,
                           range=(35,55),
                           label="Generated Spectrum")

tot = 0
for i in range(20):
    tot += n[i]

b.append(tot)

std_mcmom = round(np.std(df3["deent.mom"]),2)
mean_mcmom = round(np.mean(df3["deent.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(0.52,0.81,"STD: " + str(std_mcmom), fontsize=8)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("deent_mom.Delta_Gen45.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df3["demcgen.mom"],
                           bins=20,
                           range=(35,55),
                           label="Generated Spectrum")

std_mcmom = round(np.std(df3["demcgen.mom"]),2)
mean_mcmom = round(np.mean(df3["demcgen.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=40)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("demcgen_mom.Delta_Gen45.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df4["deent.mom"],
                           bins=20,
                           range=(40,60),
                           label="Generated Spectrum")

tot = 0
for i in range(20):
    tot += n[i]

b.append(tot)

std_mcmom = round(np.std(df4["deent.mom"]),2)
mean_mcmom = round(np.mean(df4["deent.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(0.52,0.81,"STD: " + str(std_mcmom), fontsize=8)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("deent_mom.Delta_Gen50.pdf")

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df4["demcgen.mom"],
                           bins=20,
                           range=(40,60),
                           label="Generated Spectrum")

std_mcmom = round(np.std(df4["demcgen.mom"]),2)
mean_mcmom = round(np.mean(df4["demcgen.mom"]),2)
min_ylim, max_ylim = plt.ylim()
min_xlim, max_xlim = plt.xlim()
plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=40)
plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)

plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("demcgen_mom.Delta_Gen50.pdf")


bin_num = []
efficiency = []
for i in range(4):

    if b[i] == 0:
        rat1 = 0

    elif b[i] != 0:
        rat1 = b[i]/a[i]



    else:
        rat1 = 0


    efficiency.append(rat1)

    bin_num.append(i+1)


real_b = np.asarray(efficiency)
real_bins = np.asarray(bin_num)


fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(x = real_bins, y = efficiency)
ax.set_ylabel('reconstructed p/IPA generated p')
ax.set_xlabel('Bin Number(10 MeV ea)')
fig.savefig("efficiency_delta.png")






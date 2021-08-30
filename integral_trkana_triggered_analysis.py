import uproot
import matplotlib.pyplot as plt
import math
import numpy as np

file = uproot.open("GenIPA.root")


RPCReco = file["generate/GenTree"] # opens the 'GenTree' tree in the 'generate' folder


df = RPCReco.pandas.df(flatten=False)
event_count = len(df["pmag_gen"])
print("hiiiii")
print(event_count)

fig, ax = plt.subplots(1,1)
counts, bins, patches = ax.hist(df["pmag_gen"],
                           bins=10,
                           range=(0,100),
                           label="Generated Spectrum ")
a = []
for i in range(10):
    a.append(counts[i])
print(a)

fig.savefig("split_IPA_gen.png")





#print(counts[0] + counts[1] + counts[2] + counts[3] + counts[4] + counts[5] + counts[6] + counts[7] + counts[8] + counts[9])
#make an array of length ten that stores each of the number of electrons in each bin for a, b, and c
#make another array of length ten that stores the ratio of a to b and a to c- and array with ratio and bin size, then plot scatter plot
#then plot it

#ax.set_ylabel('Electrons/bin')
#ax.set_xlabel('Gen. Mom [MeV/c]')
#fig.show()

#fig.savefig("split_IPAGen.pdf")

input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

#std_mcmom = round(np.std(df["demcgen.mom"]),2)
#mean_mcmom = round(np.mean(df["demcgen.mom"]),2)
#min_ylim, max_ylim = plt.ylim()
#min_xlim, max_xlim = plt.xlim()
fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df["demcgen.mom"],
                           bins=10,
                           range=(0,100),
                           label="RPC")

b = []
for i in range(10):
    b.append(n[i])
print(b)

#event_count_mom = len(df["demcgen.mom"])
#plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=8)
#plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)
#plt.figtext(max_xlim*0.78,max_ylim*0.77,"NEvents: " + str(event_count_mom), fontsize=8)
#axes = plt.gca()
#xmin, xmax = axes.get_xlim()
#print(xmax-xmin)
#print(xmin)
#plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
#plt.text(mean_mcmom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mcmom), fontsize = 7)
#ax.set_ylabel('Electrons/bin')
#ax.set_xlabel('Momentum [MeV/c]')
#fig.savefig("split_demcgen.mom_2.pdf")


#std_mom = round(np.std(df["deent.mom"]),2)
#mean_mom = round(np.mean(df["deent.mom"]),2)
#min_ylim, max_ylim = plt.ylim()
#min_xlim, max_xlim = plt.xlim()
#fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df["deent.mom"],
                           bins=10,
                           range=(0,100),
                           label="RPC")


c = []
for i in range(10):
    c.append(n[i])
print("this is c")
print(c)
#print(c)
#event_count_mom = len(df["deent.mom"])
#plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mom), fontsize=8)
#plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mom), fontsize=8)
#plt.figtext(max_xlim*0.78,max_ylim*0.77,"NEvents: " + str(event_count_mom), fontsize=8)
#axes = plt.gca()
#xmin, xmax = axes.get_xlim()
#print(xmax-xmin)
#print(xmin)
#plt.axvline(mean_mom, color='k', linestyle='dashed', linewidth=1)
#plt.text(mean_mom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mom), fontsize = 7)
#ax.set_ylabel('Electrons/bin')
#ax.set_xlabel('Reconstructed Momentum [MeV/c]')
#fig.savefig("split_deent.mom_2.pdf")


#ratios

btoa = []
ctoa = []
bin_num = []
for i in range(7):
    if b[i] == 0:
        rat1 = 0
    if c[i] == 0:
        rat2 = 0
    elif a[i] != 0:
        rat1 = b[i]/a[i]
        rat2 = c[i]/a[i]
    else:
        rat1 = 0
        rat2 = 0

    btoa.append(rat1)
    ctoa.append(rat2)
    bin_num.append(i+1)

real_b = np.asarray(btoa)
real_c = np.asarray(ctoa)
real_bins = np.asarray(bin_num)


#plt.scatter(real_bins,real_b,c = "blue")
#plt.savefig("hello.png")

fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(x = real_bins, y = real_b)
fig.savefig("btoa.png")

fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(x = real_bins, y = real_c)
ax.set_ylabel('reconstructed p/IPA generated p')
ax.set_xlabel('Bin Number(10 MeV ea)')
fig.savefig("ctoa.png")

print(real_b)


print(str((btoa)) + "asdfasfasfas")
print(ctoa)
print(bin_num)




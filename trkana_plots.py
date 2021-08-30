import uproot
import matplotlib.pyplot as plt
import math
import numpy as np


input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)

#std_mcmom = round(np.std(df["demcgen.mom"]),2)
#mean_mcmom = round(np.mean(df["demcgen.mom"]),2)
#min_ylim, max_ylim = plt.ylim()
#min_xlim, max_xlim = plt.xlim()
#fig, ax = plt.subplots(1,1)
#n, bins, patches = ax.hist(df["demcgen.mom"],
                           #bins=75,
                           #range=(0,65),
                           #label="RPC")
#event_count_mom = len(df["demcgen.mom"])
#plt.figtext(max_xlim*0.78,max_ylim*0.8,"STD: " + str(std_mcmom), fontsize=8)
#plt.figtext(max_xlim*0.78,max_ylim*0.83,"Mean: " + str(mean_mcmom), fontsize=8)
#plt.figtext(max_xlim*0.78,max_ylim*0.77,"NEvents: " + str(event_count_mom), fontsize=8)
#axes = plt.gca()
#xmin, xmax = axes.get_xlim()
#print(xmax-xmin)
#print(xmin)
#plt.axvline(mean_mcmom, color='k', linestyle='dashed', linewidth=1)
#plt.text(mean_mom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mom), fontsize = 7)
#ax.set_ylabel('Electrons/bin')
#ax.set_xlabel('Momentum [MeV/c]')
#fig.savefig("demcgen.mom_2.pdf")


#std_mom = round(np.std(df["deent.mom"]),2)
#mean_mom = round(np.mean(df["deent.mom"]),2)
#min_ylim, max_ylim = plt.ylim()
#min_xlim, max_xlim = plt.xlim()
#fig, ax = plt.subplots(1,1)
#n, bins, patches = ax.hist(df["deent.mom"],
                           #bins=75,
                           #range=(0,65),
                           #label="RPC")
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
#fig.savefig("deent.mom_2.pdf")




#
#
#
fig, ax = plt.subplots(1,1)
arr = [];
mc = df["demcgen.mom"]
mom = df["deent.mom"]

for i in range(130660):
    arr.append(mom[i] - mc[i])
print(arr)
n, bins, patches = ax.hist(arr,
                           bins=80,

                           range=(-40,40),
                           label="RPC")
ax.set_yscale('log')

std_diff = round(np.std(arr),2)
mean_diff = round(np.mean(arr),2)
#min_ylima, max_ylima = plt.ylim()
#min_xlima, max_xlima = plt.xlim()

event_count_diff = len(arr)
#print(max_xlima*0.78)
plt.figtext(0.74,0.85,"STD: " + str(std_diff), fontsize=8)
plt.figtext(0.74,0.82,"Mean: " + str(mean_diff), fontsize=8)
plt.figtext(0.74,0.79,"NEvents: " + str(event_count_diff), fontsize=8)


#print(xmax-xmin)
#print(xmin)
plt.axvline(mean_diff, color='k', linestyle='dashed', linewidth=1)
#plt.text(mean_mom*1.01, max_ylim*0.967, 'Mean: {:.2f}'.format(mean_mom), fontsize = 7)
ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Momentum-Reconstructed Momentum [MeV/c]')
fig.savefig("demcgen.mom-deent.mom_2.png")

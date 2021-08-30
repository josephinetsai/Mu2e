import uproot
import matplotlib.pyplot as plt
import math
import numpy as np

input_file = uproot.open("trkana.Triggered.root")
input_tree = input_file["TrkAnaNeg/trkana"]
df = input_tree.pandas.df(flatten = False)


#branching fuction calculation

E =  [0.001228,0.026721,0.075994]
E_sub = []
E_sub.append(0.001228/2)
E_sub.append(0.001228)
E_sub.append((0.001228+0.026721)/2)
E_sub.append(0.026721)
E_sub.append((0.026721 + 0.075994)/2)
E_sub.append(0.075994)
print(E_sub)
print(len(E_sub))



bin_number = []

tot_mu_decays = 0.86*(1.19*(10**13))
tot_mu_decays_Michel = 0.86*8599341#8599341 total number of events in IPAGen
N = []
Ns = []
BF = []
BF_2 = []

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df["deent.mom"],
                           bins=6,
                           range=(30,60),
                           label="RPC")
for i in range(6):
   # BF.append(0)
    BF_2.append(0)
    bin_number.append(i+1)

for i in range(6):
    print(i)
    N.append(n[i]*1000000)
    Ns.append(1.64*math.sqrt(N[i]))
    #BF.append(Ns[i]/(E[i]*tot_mu_decays_Michel))
    BF_2.append(Ns[i]/(E_sub[i]*tot_mu_decays))
    bin_number.append(i+7)

print(N)
print(Ns)

branchingFraction = np.asarray(BF)
branchingFraction_2 = np.asarray(BF_2)
bins_ = np.asarray(bin_number)
print(len(BF_2))
print(len(bin_number))




fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(x = bins_, y = branchingFraction_2)
ax.set_ylabel('Branching Fraction')
ax.set_xlabel('Bin Number(5 MeV ea)')
fig.savefig("Branching_Fraction_2_sub.png")

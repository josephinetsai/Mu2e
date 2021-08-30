import uproot
import matplotlib.pyplot as plt
import math
import numpy as np

file = uproot.open("GenIPA.root")

RPCReco = file["generate/GenTree"] # opens the 'GenTree' tree in the 'generate' folder
df = RPCReco.pandas.df(flatten=False)

fig, ax = plt.subplots(1,1)
n, bins, patches = ax.hist(df["pmag_gen"],
                           bins=50,
                           range=(0,100),
                           label="Generated Spectrum")

ax.set_ylabel('Electrons/bin')
ax.set_xlabel('Gen. Mom [MeV/c]')
fig.show()
fig.savefig("IPAGen.pdf")


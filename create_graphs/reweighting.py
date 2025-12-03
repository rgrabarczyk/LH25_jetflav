import json
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_pts_from_json(file):
    """Load jet pt values from a single JSON file, skipping jets with < 3 fourmomenta entries."""
    with open(file) as f:
        data = json.load(f)
    pts = []
    for jet_list in data:
        for jet in jet_list:
            if "fourmomenta" in jet and len(jet["fourmomenta"]) >= 3:
                pts.append(jet["pt"])
    return np.array(pts)

# --- Load samples ---
sample_A_files = "CMP_30GeV_cjets.json"
sample_B_files = "IFN_30GeV_cjets.json"

pts_A = load_pts_from_json(sample_A_files)#np.concatenate([load_pts_from_json(f) for f in sample_A_files])
pts_B = load_pts_from_json(sample_B_files)#np.concatenate([load_pts_from_json(f) for f in sample_B_files])

# --- Histogram bins ---
bins = np.linspace(30, 150, 50)

# --- Compute densities ---
hist_A, bin_edges = np.histogram(pts_A, bins=bins, density=True)
hist_B, _ = np.histogram(pts_B, bins=bins, density=True)

# Avoid divide by zero
hist_A = np.where(hist_A == 0, 1e-10, hist_A)

# --- Compute weights: w = p_B / p_A ---
weights = 2 * hist_B / hist_A

# --- Assign weights to each pt in A ---
bin_indices = np.digitize(pts_A, bins) - 1
bin_indices = np.clip(bin_indices, 0, len(weights) - 1)
event_weights = weights[bin_indices]
print("len(pts_A):", len(pts_A))
print("len(bin_indices):", len(bin_indices))
print("len(event_weights):", len(event_weights))
print("len(weights):", len(weights))
print("bins.shape:", bins.shape)
print("max bin index:", np.max(bin_indices))
print(len(event_weights))
np.save("weights_sample_A.npy", event_weights)

# --- Plot original distributions ---
plt.figure(figsize=(10, 6))
plt.hist(pts_A, bins=bins, alpha=0.5, label=r"CMP", density=True)
plt.hist(pts_B, bins=bins, alpha=0.5, label=r"IFN", density=True)

# --- Plot reweighted Sample A ---
plt.hist(pts_A, bins=bins, weights=event_weights, 
         alpha=0.5, label=r"CMP(reweighted)", density=True, histtype='step', linewidth=2)

plt.xlabel(r"c jet $p_T$ [GeV]",fontsize=17)
plt.ylabel(r"$\frac{1}{\sigma}\frac{d\sigma}{d p_T}$",fontsize=17)
plt.legend()
#plt.title("Jet $p_T$ Distributions and Reweighting")
plt.tight_layout()
plt.savefig("reweighting.pdf")

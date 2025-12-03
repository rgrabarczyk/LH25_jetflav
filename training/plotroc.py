import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from matplotlib.ticker import LogFormatter

def plot_efficiency_rejection_log(file_names):
    """
    Loads FPR and TPR from .npz files, plots Signal Efficiency (TPR)
    vs. Background Rejection (1/FPR) on a logarithmic y-axis,
    and calculates AUC.

    Args:
        file_names (list): A list of .npz file paths.
                           e.g., ['ATLASROC.npz', 'IFNROC.npz'].
    """
    plt.figure(figsize=(9, 7))

    for file_name in file_names:
        try:
            data = np.load(file_name)
            fpr = data['fpr']
            tpr = data['tpr']

            # Calculate AUC using original FPR and TPR (standard ROC definition)
            roc_auc = auc(fpr, tpr)

            # Filter out points where FPR is zero to avoid division by zero or log(0)
            # This is crucial for the 1/FPR calculation on a log scale
            mask = fpr > 0
            filtered_fpr = fpr[mask]
            filtered_tpr = tpr[mask]

            # Calculate Background Rejection as 1/FPR for the y-axis
            background_rejection_y_axis = 1.0 / filtered_fpr

            # Determine label name based on file name
            if "ATLAS" in file_name:
                label_name = "ATLAS"
            elif "IFN" in file_name:
                label_name = "IFN"
            elif "AKT" in file_name:
                label_name = "AKT"
            elif "CMP" in file_name:
                label_name = "CMP (reweighted)"
            else:
                label_name = file_name.replace('.npz', '')

            # Plotting: X-axis is c-jet efficiency (TPR), Y-axis is Background Rejection (1/FPR)
            plt.plot(filtered_tpr, background_rejection_y_axis, lw=2, label=f'{label_name} (AUC = {roc_auc:.2f})')
            print(f"AUC for {file_name}: {roc_auc:.2f}")

        except FileNotFoundError:
            print(f"Error: The file '{file_name}' was not found. Please ensure it's in the same directory as the script.")
        except Exception as e:
            print(f"An error occurred while processing '{file_name}': {e}")

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set plot limits and labels
    plt.xlim([0.1, 1.0]) # c-jet efficiency typically from 0 to 1
    # Adjust ylim based on your expected rejection values, common starting point is 1
    # If your rejection goes very high, adjust the upper limit, e.g., 1e4, 1e5, 1e6
    plt.ylim([1.0, 500]) # Example: from 1 to 10,000 for background rejection
    formatter = LogFormatter(labelOnlyBase=True)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel(r'$c$-jet efficiency (True Positive Rate)', fontsize=12)
    plt.ylabel(r'Background Rejection ($1/\text{False Positive Rate}$)', fontsize=12)
    #plt.title('Signal Efficiency vs. Background Rejection', fontsize=14)
    plt.legend(loc="upper right", fontsize=10) # Often 'upper right' is good for this plot
    plt.grid(True, which="both", ls="--", alpha=0.7) # Grid for both major and minor ticks
    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.savefig("ROCCURVES.pdf")

if __name__ == "__main__":
    # Call the function with your .npz file names
    # Make sure your files are named 'ATLASROC.npz' and 'IFNROC.npz' as in your example.
    plot_efficiency_rejection_log(['ATLASroccurve.npz', 'AKTroccurve.npz', 'IFNroccurve.npz', 'CMProccurve.npz'])
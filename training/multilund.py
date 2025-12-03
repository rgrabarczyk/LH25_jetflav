#!/usr/bin/env python3

import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_curve, auc
import tqdm
import dgl
from lundhelpers import train, evaluate, LundWeightedDataset, collate_wrapper_weighted_tree
from LundNet import LundNet


def load_graphs_and_labels(signal_path, background_path, signal_weights_path=None, background_weights_path=None):
    sig_graphs, _ = dgl.load_graphs(signal_path)
    bkg_graphs, _ = dgl.load_graphs(background_path)

    graphs = sig_graphs + bkg_graphs
    labels = [1] * len(sig_graphs) + [0] * len(bkg_graphs)

    # Default weights
    weights = [1.0] * (len(sig_graphs) + len(bkg_graphs))

    if signal_weights_path is not None:
        sig_weights = np.load(signal_weights_path)
        assert len(sig_weights) == len(sig_graphs), "Signal weights length mismatch"
        weights[:len(sig_weights)] = sig_weights

    if background_weights_path is not None:
        bkg_weights = np.load(background_weights_path)
        assert len(bkg_weights) == len(bkg_graphs), "Background weights length mismatch"
        weights[-len(bkg_weights):] = bkg_weights

    return graphs, labels, weights

def build_model(config):
    conv_params = [[32, 32], [32, 32], [128, 128], [128, 128]]
    fc_params = [(256, 0.1)]
    use_fusion = True

    input_dims = 3
    if '5' in config['model']['model_type']:
        input_dims = 5

    return LundNet(input_dims=input_dims,
                   num_classes=2,
                   conv_params=conv_params,
                   fc_params=fc_params,
                   use_fusion=use_fusion)


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs, labels, weights = load_graphs_and_labels(
        args.signal, args.background,
        signal_weights_path=args.signal_weights,
        background_weights_path=args.background_weights
    )
    dataset = LundWeightedDataset(graphs, labels, weights)

    n_total = len(dataset)
    n_train = int(config['train_fraction'] * n_total)
    n_val = int(config['val_fraction'] * n_total)
    n_test = n_total - n_train - n_val

    train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_wrapper_weighted_tree)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate_wrapper_weighted_tree)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False,
                             collate_fn=collate_wrapper_weighted_tree)

    model = build_model(config).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, config['scheduler']['step_size'],
                                                gamma=config['scheduler']['gamma'])

    best_acc = 0
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch}")
        train(model, opt, scheduler, train_loader, dev, config)
        val_acc = evaluate(model, val_loader, dev)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "CMPbest_lundnet.pt")
        print(f"Validation accuracy: {val_acc:.4f} (best: {best_acc:.4f})")

    # ROC on test
    print("Evaluating test set and computing ROC...")
    model.load_state_dict(torch.load("CMPbest_lundnet.pt"))
    model.eval()

    scores = evaluate(model, test_loader, dev, return_scores=True)
    class1_probs = scores[:, 1]

    # Collect labels from test_loader
    y_true = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for batch in tq:
                y_true.append(batch.labels.cpu().numpy())  # convert tensor to numpy array

    # Flatten y_true to a 1D array
    y_true = np.concatenate(y_true).astype(int)

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true, class1_probs)
    roc_auc = auc(fpr, tpr)

    # Save
    np.savez("CMProccurve", fpr=fpr, tpr=tpr)
    print(f"AUC = " + roc_auc)
    #plt.figure()
    #plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.title("LundNet ROC Curve (Test Set)")
    #plt.legend(loc='lower right')
    #plt.grid()
    #plt.savefig("lundnet_roc_curve.png")
    #print("Saved ROC to lundnet_roc_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--signal', type=str, required=True, help='Path to signal bin file')
    parser.add_argument('--background', type=str, required=True, help='Path to background bin file')
    parser.add_argument('--signal-weights', type=str, default=None, help='Optional .npy file with weights for signal')
    parser.add_argument('--background-weights', type=str, default=None, help='Optional .npy file with weights for background')
    args = parser.parse_args()

    main(args)

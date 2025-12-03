# This file is part of MultiFold unfolding software adapted to Lund Trees for the  ATLAS Z + bbbar analysis
# parts of the code are adapted from https://github.com/fdreyer/LundNet
# R. Grabarczyk, 2023
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import tqdm
import dgl
## FUNCTIONS SPECIFIC FOR LUNDNET: train, evaluate, LundWeightedDataset, weighted loss, collate_fn, ...
def train(model, opt, scheduler, train_loader, dev, training_info):
    model.train()
    dev = torch.device(training_info['device'])
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    tic = time.time()
    # tqdm = track bar
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        # for each batch we perform a training step
        for batch in tq:
            weights = batch.weights
            labels = batch.labels
            num_examples = labels.shape[0]
            labels = labels.to(dev).squeeze().long()
            weights = weights.to(dev).squeeze()
            # torch by deafult accumulates the gradients over batches,
            # but here we want to set the gradient to zero for each batch
            opt.zero_grad()
            logits = model(batch.batch_graph.to(dev), batch.features.to(dev))
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            opt.step()
            _, preds = logits.max(1)
            num_batches += 1
            count += num_examples
            loss = loss.item()
            correct = (preds == labels).sum().item()
            total_loss += loss
            total_correct += correct
            tq.set_postfix({
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})
    scheduler.step()
    ts = time.time() - tic
    print('Trained over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
        count=count, ts=ts, speed=count / ts
    ))


def evaluate(model, test_loader, dev, return_scores=False, return_time=False):
    model.eval()
    total_correct = 0
    sumweights = 0
    count = 0
    scores = []
    tic = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for batch in tq:
                label = batch.labels
                weights = batch.weights
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                logits = model(batch.batch_graph.to(dev), batch.features.to(dev))
                _, preds = logits.max(1) # logits.max(1) outputs (maximum value of logit), (prediction (index with max value))
                                         # the logits DO NOT add up to 1. To get a probability from them, use softmax (just like in return_scores)
#                print(logits.max(1))
#                print(label) # all labels are 1 for some reason?
                if return_scores:
                    # convert the raw outputs to probabilities using softmax:
                    scores.append(torch.softmax(logits, dim=1).cpu().detach().numpy())
                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples
                tq.set_postfix({
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgWeightedAcc': '%.5f' % (total_correct / count)})

    ts = time.time() - tic
    print('Tested over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(count=count,ts=ts,speed=count/ts))
    if return_time:
        return ts

    if return_scores:
        return np.concatenate(scores)
    else:
        return total_correct / count


class LundWeightedDataset(Dataset):
    # Lund Trees dataset, with required PyTorch functionalities.
    # (must override the  __len__ and __getitem__ of the inherited Dataset class)
    def __init__(self, graphs, labels, weights): # TODO: add a lunddim argument here that would cut away Lund Coords that we do not wish to use
        # Here, in our case, it would be senseless to extract graphs all the way from the filepath here, as this funciton is only supposed to apply weights to
        # samples and make them iterable; instead we extract the graphs in the appropriate format elsewhere in the code, to not repeat it many times
        self.data = graphs
        self.labels = torch.tensor(labels)
        self.weights = torch.tensor(weights)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = {'graph': self.data[i], 'label': self.labels[i], 'weight': self.weights[i]}
        return sample


def weighted_loss(logits, labels, weights):
    function = torch.nn.CrossEntropyLoss(reduction = 'none')
    # might need a cuda exception, but maybe not since weights are part of the dataset
    # which is already on the gpu
    loss = function(logits, labels)*weights
    loss = torch.mean(loss)
    return loss


class _WeightedLundTreeBatch:

    def __init__(self, data):

        graphs = [l['graph'] for l in data]

        self.batch_graph = dgl.batch(graphs)

        self.features = self.batch_graph.ndata.pop('features').float()  

        try:
            self.labels = torch.tensor([l['label'] for l in data]).float()  
        except:
            self.labels = [l['label'] for l in data]
        try:
            self.weights = torch.tensor([l['weight'] for l in data]).float()  
        except:
            self.weights = [l['weight'] for l in data]

    def pin_memory(self):
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.weights = self.weights.pin_memory()
        return self


def collate_wrapper_weighted_tree(batch):
    return _WeightedLundTreeBatch(batch)

def set_iteration_parameters(list_of_infos):
    previous_target_model_reco = None
    previous_target_model_truth = None
    for i, dictionary in enumerate(list_of_infos):
        dictionary['info_file'] = f'iteration-{i+1}-info-file.txt'
        dictionary['model']['target_model_path_reco'] = f'iteration-{i+1}-target-reco.pt' 
        dictionary['model']['target_model_path_truth'] = f'iteration-{i+1}-target-truth.pt'
        if previous_target_model_reco is not None:
            dictionary['model']['source_model_path_reco'] = previous_target_model_reco
        if previous_target_model_truth is not None:
            dictionary['model']['source_model_path_truth'] = previous_target_model_truth
        previous_target_model_reco = dictionary.get('model').get('target_model_path_reco')
        previous_target_model_truth = dictionary.get('model').get('target_model_path_truth')
        
    return list_of_infos
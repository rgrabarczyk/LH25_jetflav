import ijson
import json
import fastjet as fj
import numpy as np 
import networkx as nx
import argparse
from networkx.drawing.nx_agraph import graphviz_layout
import dgl
from dgl.data.utils import save_graphs

def LundDeclustering(jharder, jsofter):
    lnz = -np.float32(np.log(jsofter.pt() / (jharder.pt() + jsofter.pt())))
    delta = jharder.delta_R(jsofter)
    lndelta = np.float32(np.log(0.4/delta))
    lnkt = np.float32(np.log(jsofter.pt() * delta))
    lnm = np.float32(0.5 * np.log(abs((jharder + jsofter).m2())))
    psi = np.float32(np.arctan2(jsofter.rap() - jharder.rap(), jsofter.phi() - jharder.phi()))
    return [lnz, lndelta, psi, lnm, lnkt]

def read_json_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description='Parser for input JSON file')
    parser.add_argument('jsonpath', type=str, help='Path to the input JSON file')
    parser.add_argument('-nev', '--nevents', type=int, default=2, help='Number of events to process')
    args = parser.parse_args()
    
    #data = read_json_file(args.jsonpath)
    #totalentries = min(len(data), args.nevents)
    LundTreeRecoCollection = []
    LundTreeTruthCollection = []
    LundTreeFakeCollection = []
    ptCollection = []
    tptCollection = []
    rcounter = 0
    tcounter = 0
    fcounter = 0
    with open(args.jsonpath, "r") as f:
        json_thing = ijson.items(f, "item",use_float=True)
        for i, entry in enumerate(json_thing):
            for j, entryreal in enumerate(entry):
                if tcounter == args.nevents:
                    break
                if not entryreal:
                    continue
                fourmomenta = entryreal["fourmomenta"]
                if(len(fourmomenta) < 3): continue
                #print("#### FOURMOMENTA = ", fourmomenta)

                LundTreeTruthCollection.append(dgl.from_networkx(networkx_from_constits(fourmomenta, follow_primary=False), node_attrs=['features']))
                tcounter += 1

        dgl.save_graphs("./"  + str(tcounter) + "C21_l" + "jets.bin", LundTreeTruthCollection)

        
def networkx_from_constits(fourmomenta, follow_primary = False):
    jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
    input_pjs = [fj.PseudoJet(float(px), float(py), float(pz), float(E)) for px, py, pz, E in fourmomenta]
    # 22.08.2023: attempt at looking directly at the cs.jets() list
    # and creating a Lund Tree from it
    cs = fj.ClusterSequence(input_pjs, jet_def)
    jetlist = cs.jets()[len(input_pjs):] # the PseudoJets cut out here are just input_pjs
    jetlist = jetlist[::-1] # now the first element = full pseudojet
    if len(jetlist) == 0:
        return nx.Graph()
    if follow_primary:
        hardguys = [jetlist[0]]

    assert len(jetlist) == len(input_pjs) - 1 # if cs.jets() works correctly, must be the case

    LundTree = nx.Graph()
    
    for i_j in range(len(jetlist)):
        # check if the jet is a result of a recombination
        # the first n entries in jetlist are just input_pjs
        # so they should have been cut out by now anyway

        current_jet = jetlist[i_j]
        child = fj.PseudoJet()
        parharder = fj.PseudoJet()
        parsofter = fj.PseudoJet()
        if not current_jet.has_parents(parharder,parsofter): continue
        
        if parsofter.pt() > parharder.pt(): parharder, parsofter = parsofter, parharder

        LundCoord = LundDeclustering(parharder,parsofter)
        LundTree.add_node(i_j, coordinates = LundCoord, features = LundCoord)

        if follow_primary:
            hardchild = fj.PseudoJet()
            if (not parharder.has_child(hardchild)) or (hardchild in hardguys):
                hardguys.append(parharder)
                LundTree.nodes[i_j]["follow_primary"] = "primary"
            else:
                LundTree.nodes[i_j]["follow_primary"] = "secondary"

        if current_jet.has_child(child):
            LundTree.add_edge(jetlist.index(child), i_j)

    return LundTree

if __name__ == '__main__':
    main()

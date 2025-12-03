// helpers.cpp
#include "helpers.h"
#include "HepMC3/GenParticle.h" // Include the full definition of GenParticle
#include "HepMC3/GenVertex.h"   // Include the full definition of GenVertex
#include <cmath>
#include <cmath>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include <iostream>
#include <fstream>

using namespace HepMC3;

// Check if the particle is the last pre-decay B-hadron
bool is_b_hadron(int pdg_id) {
    int abs_pdg = std::abs(pdg_id);
    return (abs_pdg >= 500 && abs_pdg < 600) || (abs_pdg >= 5000 && abs_pdg < 6000);
}

bool is_c_hadron(int pdg_id) {
    int abs_pdg = std::abs(pdg_id);
    return (abs_pdg >= 400 && abs_pdg < 500) || (abs_pdg >= 4000 && abs_pdg < 5000);
}

bool is_last_pre_decay_b_hadron(const std::shared_ptr<GenParticle>& particle) {
    // Ensure the particle itself is a B-hadron
    if (!is_b_hadron(particle->pid())) return false;

    // Check all children to see if any are also B-hadrons
    for (const auto& child : particle->end_vertex()->particles_out()) {
        if (is_b_hadron(child->pid())) {
            return false; // It's not the last pre-decay B-hadron
        }
    }

    return true; // It's the last pre-decay B-hadron
}

bool is_last_pre_decay_c_hadron(const std::shared_ptr<GenParticle>& particle) {
    // Ensure the particle itself is a C-hadron
    if (!is_c_hadron(particle->pid())) return false;

    // Check all children to see if any are also C-hadrons
    for (const auto& child : particle->end_vertex()->particles_out()) {
        if (is_c_hadron(child->pid())) {
            return false; // It's not the last pre-decay B-hadron
        }
    }

    return true; // It's the last pre-decay B-hadron
}

bool is_c_hadron_child_of_b_hadron(const std::shared_ptr<GenParticle>& particle) {
    // Get the production vertex of the particle
    const auto& production_vertex = particle->production_vertex();
    if (!production_vertex) return false; // No production vertex means no parent

    // Traverse all ancestors of this particle
    for (const auto& ancestor : production_vertex->particles_in()) {
        int pid = ancestor->pid();
        if (ancestor->status() == 2 && is_b_hadron(pid)) {
            return true; // Found a B-hadron ancestor
        }
        // Recursively check the parent's ancestors
        if (is_c_hadron_child_of_b_hadron(ancestor)) {
            return true;
        }
    }

    return false; // No B-hadron ancestor found
}

bool is_child_of_b_or_c_hadron(const std::shared_ptr<HepMC3::GenParticle>& particle) {
    // Get the production vertex of the particle
    auto prod_vertex = particle->production_vertex();
    if (!prod_vertex) return false; // No production vertex means no parent

    // Check all parents (particles entering the production vertex)
    for (const auto& parent : prod_vertex->particles_in()) {
        if (is_b_hadron(parent->pid()) || is_c_hadron(parent->pid())) {
            return true; // Parent is a B or C hadron
        }
        // Recursively check the parent's ancestors
        if (is_child_of_b_or_c_hadron(parent)) {
            return true;
        }
    }

    return false; // No B or C hadron found in the ancestry
}

void print_jet_to_json(fastjet::PseudoJet& jet, std::ofstream& JSON_file, 
                       bool& found_other_algo_b, bool& found_other_algo_c) {
    if (!JSON_file.is_open()) {
        std::cerr << "Output file stream is not open!" << std::endl;
        return;
    }

    std::vector<fastjet::PseudoJet> constits = jet.constituents();

    JSON_file << "{\n";
    JSON_file << "  \"pt\": " << jet.pt() << ",\n";
    JSON_file << "  \"eta\": " << jet.eta() << ",\n";
    JSON_file << "  \"phi\": " << jet.phi() << ",\n";
    JSON_file << "  \"mass\": " << jet.m() << ",\n";
    JSON_file << "  \"fourmomenta\":[";
    for (int j = 0; j < constits.size(); j++) {
        fastjet::PseudoJet part = constits[j];
        JSON_file << "[" << part.px() << "," << part.py() << "," << part.pz() << "," << part.E(); 
        if (j != constits.size() - 1){
            JSON_file << "],";
        } else {
            JSON_file << "]]";
        }
    } 
    JSON_file << ",\n"; 
    int found_other_b_int = found_other_algo_b ? 1 : 0;
    int found_other_c_int = found_other_algo_c ? 1 : 0;
    JSON_file << "  \"found_other_algo_b\": " << found_other_b_int << ",\n";
    JSON_file << "  \"found_other_algo_c\": " << found_other_c_int << "\n";

    JSON_file << "}\n";
}

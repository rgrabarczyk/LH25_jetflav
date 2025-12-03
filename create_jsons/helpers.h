// helpers.h
#ifndef HELPERS_H
#define HELPERS_H

#include "HepMC3/GenEvent.h"
#include <memory>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include <iostream>
#include <fstream>

// Declare functions
bool is_b_hadron(int pdg_id);
bool is_last_pre_decay_b_hadron(const std::shared_ptr<HepMC3::GenParticle>& particle);
bool is_c_hadron(int pdg_id);
bool is_last_pre_decay_c_hadron(const std::shared_ptr<HepMC3::GenParticle>& particle);
bool is_c_hadron_child_of_b_hadron(const std::shared_ptr<HepMC3::GenParticle>& particle);
bool is_child_of_b_or_c_hadron(const std::shared_ptr<HepMC3::GenParticle>& particle);
void print_jet_to_json(fastjet::PseudoJet& jet,
                       std::ofstream& JSON_file,
                       bool& found_other_algo_b,
                       bool& found_other_algo_c);

#endif // HELPERS_H

// Les Houches 2025 Jet Flavour
// This reads in HepMC3 outputs and produces JSON files: 1 jet per entry
// This is untouched from Les Houches 2025, where I just reran this script for each algo separately
// Obvious improvement will be to run all of them at once.
// Entries are of the form:
//      [
//      {
//        "pt": 86.3693,                                           <-- jet pt
//        "eta": -1.26271,                                         <-- jet eta
//        "phi": 0.482061,                                         <-- jet phi
//        "mass": 9.07214,                                         <-- jet mass
//        "fourmomenta":[[1.45284,0.551922,-3.5868,3.90903], ... ] <-- fourmomenta of constituents
//        "found_other_algo_b": 0                                  <-- these are perhaps a bit silly, I ran with pairs of algos at the time                                       
//        "found_other_algo_c": 0                                  <-- and this tells you whether the other algo that you're running considered this a b or c jet
//      }
//      ,
//      { ...
//
// this can later be fed to another script that creates Lund Trees on which I trained the GNN classifier.

#include "HepMC3/GenEvent.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/Print.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/contrib/FlavInfo.hh"
#include "fastjet/contrib/IFNPlugin.hh"
#include "fastjet/contrib/CMPPlugin.hh"
#include "fastjet/contrib/GHSAlgo.hh"
#include "fastjet/contrib/SDFlavPlugin.hh"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "helpers.h" 

using namespace HepMC3;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <HepMC3_input_file>" << std::endl;
        exit(-1);
    }
    // Input file
    ReaderAscii input_file(argv[1]);
    std::string inputStr(argv[1]);
    std::string namestr = std::string(1, inputStr[inputStr.size() - 8]) +
                          std::string(1, inputStr[inputStr.size() - 7]);
    // Output files that later go into the Lund graph making code  
    std::ofstream ATLAS_cJSON_file("/eos/home-r/ragrabar/Flavour/LesHouches/LH25JSONS/" + namestr + "ATLAS_30GeV_cjets.json");
    std::ofstream ATLAS_bJSON_file("/eos/home-r/ragrabar/Flavour/LesHouches/LH25JSONS/" + namestr + "ATLAS_30GeV_bjets.json");
    std::ofstream ATLAS_lJSON_file("/eos/home-r/ragrabar/Flavour/LesHouches/LH25JSONS/" + namestr + "ATLAS_30GeV_ljets.json");
    std::ofstream IFN_cJSON_file("/eos/home-r/ragrabar/Flavour/LesHouches/LH25JSONS/" + namestr + "IFN_30GeV_lightZ_cjets.json");
    std::ofstream IFN_bJSON_file("/eos/home-r/ragrabar/Flavour/LesHouches/LH25JSONS/" + namestr + "IFN_30GeV_lightZ_bjets.json");
    std::ofstream IFN_lJSON_file("/eos/home-r/ragrabar/Flavour/LesHouches/LH25JSONS/" + namestr + "IFN_30GeV_lightZ_ljets.json");
    ATLAS_bJSON_file << "[\n";
    ATLAS_cJSON_file << "[\n";
    ATLAS_lJSON_file << "[\n";
    IFN_bJSON_file << "[\n";
    IFN_cJSON_file << "[\n";
    IFN_lJSON_file << "[\n";
    // Define all algos used in this analysis:
    const double R = 0.4;
    fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
    fastjet::Selector select_pt = fastjet::SelectorPtRange(30,150);
    fastjet::Selector select_pt_IFN = fastjet::SelectorPtRange(20,160);
    fastjet::contrib::FlavRecombiner flav_recombiner(fastjet::contrib::FlavRecombiner::net);
    //SDFlavourCalc sdFlavCalc;
    //double CMP_a = 0.1;
    //fastjet::CMPPlugin::CorrectionType CMP_corr = fastjet::CMPPlugin::CorrectionType::OverAllCoshyCosPhi_a2;
    //fastjet::CMPPlugin::ClusteringType CMP_clust = fastjet::CMPPlugin::ClusteringType::DynamicKtMax;
    //fastjet::JetDefinition CMP_jet_def(new fastjet::CMPPlugin(R, CMP_a, CMP_corr, CMP_clust, false, fastjet::contrib::FlavRecombiner::net));
    //CMP_jet_def.set_recombiner(&flav_recombiner);
    //CMP_jet_def.delete_plugin_when_unused();
    //fastjet::JetDefinition base_jet_def(fastjet::antikt_algorithm, R);
    //base_jet_def.set_recombiner(&flav_recombiner);
    auto FNPlugin = new fastjet::contrib::FlavNeutraliserPlugin(jet_def, 2.0, 1.0, fastjet::contrib::FlavRecombiner::net);
    fastjet::JetDefinition flav_neut_jet_def(FNPlugin);
    //fastjet::JetDefinition jet_def2(fastjet::antikt_algorithm, R);
    //jet_def2.set_recombiner(&flav_recombiner);
    // Counters to check the sample
    int IFN_b_counter = 0;
    int IFN_c_counter = 0;
    int IFN_l_counter = 0;
    int ATLAS_b_counter = 0;
    int ATLAS_c_counter = 0;
    int ATLAS_l_counter = 0;

    int events_parsed = 0;
    while (!input_file.failed()) {
        GenEvent evt(Units::GEV, Units::MM);

        // Read event from input file
        input_file.read_event(evt);

        // If reading failed - exit loop
        if (input_file.failed()) break;

        // Collect final-state particles for clustering
        std::vector<fastjet::PseudoJet> particles;
        for (const auto& particle : evt.particles()) {
            if (particle->status() == 1 && abs(particle->pdg_id()) != 14 
                                        && abs(particle->pdg_id()) != 16 
                                        && abs(particle->pdg_id()) != 18) { // Final-state non-neutrinos
                auto momentum = particle->momentum();
                particles.emplace_back(momentum.px(), momentum.py(), momentum.pz(), momentum.e());
            }
        }

        // Find B-hadrons in the event
        std::vector<std::shared_ptr<GenParticle>> b_hadrons;
        std::vector<std::shared_ptr<GenParticle>> c_hadrons;
        for (const auto& particle : evt.particles()) {
            if (is_last_pre_decay_b_hadron(particle)) {
                b_hadrons.push_back(particle);
            }
            if (is_last_pre_decay_c_hadron(particle) 
                && !is_c_hadron_child_of_b_hadron(particle)) {
                c_hadrons.push_back(particle);
            }
        }
    
        // make IRC safe inputs
        std::vector<fastjet::PseudoJet> IRC_input_particles;
        for (const auto& b_hadron : b_hadrons) {
            auto pb = b_hadron->momentum();
            fastjet::PseudoJet bpj(pb.px(), pb.py(), pb.pz(), pb.e());
            int flavinfo_idx = b_hadron->pdg_id() > 0 ? 5 : -5;
            bpj.set_user_info(new fastjet::contrib::FlavInfo(flavinfo_idx));
            IRC_input_particles.push_back(bpj);
        }
        for (const auto& c_hadron : c_hadrons) {
            auto pc = c_hadron->momentum();
            fastjet::PseudoJet cpj(pc.px(), pc.py(), pc.pz(), pc.e());
            int flavinfo_idx = c_hadron->pdg_id() > 0 ? 4 : -4;
            cpj.set_user_info(new fastjet::contrib::FlavInfo(flavinfo_idx));
            IRC_input_particles.push_back(cpj);
        }
        for (const auto& particle : evt.particles()) {
            if (particle->status() != 1) continue;
            if (is_child_of_b_or_c_hadron(particle)) continue;
            if (abs(particle->pdg_id()) == 14 || abs(particle->pdg_id()) == 16 || abs(particle->pdg_id()) == 18) continue;
            auto momentum = particle->momentum();
            fastjet::PseudoJet light(momentum.px(), momentum.py(), momentum.pz(), momentum.e());
            light.set_user_info(new fastjet::contrib::FlavInfo(0));
            IRC_input_particles.push_back(light);     
        }
        // Run GHS
        fastjet::ClusterSequence neutralised_clustering(IRC_input_particles, flav_neut_jet_def);
        fastjet::vector<fastjet::PseudoJet> base_jets = neutralised_clustering.inclusive_jets();
        //std::vector<fastjet::PseudoJet> GHS_jets  = fastjet::contrib::run_GHS(base_jets, 30.0, 1.0, 2.0, flav_recombiner);
        std::vector<fastjet::PseudoJet> IFN_pseudojets = select_pt_IFN(base_jets);
        //IFN_pseudojets = sdFlavCalc(IFN_pseudojets);
        // Collect the b and c jets that it found
        std::vector<fastjet::PseudoJet> b_IFN_pseudojets;
        std::vector<fastjet::PseudoJet> c_IFN_pseudojets;
        for (size_t j = 0; j < IFN_pseudojets.size(); ++j) {
            fastjet::PseudoJet current_IFN_jet = IFN_pseudojets[j];
            //sdFlavCalc(current_IFN_jet);
            if (fastjet::contrib::FlavHistory::current_flavour_of(current_IFN_jet)[5] != 0) {
                b_IFN_pseudojets.push_back(current_IFN_jet);
            } else if (fastjet::contrib::FlavHistory::current_flavour_of(current_IFN_jet)[4] != 0) {
                c_IFN_pseudojets.push_back(current_IFN_jet);
            }
        }
        // At this point we have nice lists of IFN b and c jets 
        // (with different kinematics to standard antikt as they have undecayed Bs!)
        // Perform jet clustering on *final state* particles now
        fastjet::ClusterSequence cs(particles, jet_def);
        std::vector<fastjet::PseudoJet> jets = select_pt(cs.inclusive_jets());

        // Define ATLAS b c and light jets and match IFN
        // b and c jets to them anglewise
        for (size_t i = 0; i < jets.size(); ++i) {
            bool found_b = false;
            bool found_c = false;
            fastjet::PseudoJet current_jet = jets[i];
            for (const auto& b_hadron : b_hadrons) {
                auto bmom = b_hadron->momentum();
                if (bmom.px()*bmom.px() + bmom.py()*bmom.py() < 5*5) continue;
                fastjet::PseudoJet b(bmom.px(), bmom.py(), bmom.pz(), bmom.e());
                if(b.delta_R(current_jet) < 0.3) {
                    found_b = true;
                    break;
                }

            }
            if (!found_b) {
                for (const auto& c_hadron : c_hadrons) {
                    auto cmom = c_hadron->momentum();
                    if (cmom.px()*cmom.px() + cmom.py()*cmom.py() < 5*5) continue;
                    fastjet::PseudoJet c(cmom.px(), cmom.py(), cmom.pz(), cmom.e());
                    if(c.delta_R(current_jet) < 0.3) {
                        found_c = true;
                        break;
                    }
                }
            }

            // Now, do the same thing for IFN truth labeled jets
            bool found_IFN_b = false;
            bool found_IFN_c = false;
            for (const auto& b_IFN : b_IFN_pseudojets) {
                if(b_IFN.delta_R(current_jet) < 0.3) {
                    found_IFN_b = true;
                    break;
                } 
            }
            if (!found_IFN_b) {
                for (const auto& c_IFN : c_IFN_pseudojets) {
                    if(c_IFN.delta_R(current_jet) < 0.3) {
                        found_IFN_c = true;
                        break;
                    } 
                }
            }

            if (found_b) {
                if (ATLAS_b_counter > 0) {
                    ATLAS_bJSON_file << ",\n";  // Only print comma before subsequent entries
                }
                print_jet_to_json(current_jet, ATLAS_bJSON_file, found_IFN_b, found_IFN_c);
                ATLAS_b_counter += 1;
            } else if (found_c) {
                if (ATLAS_c_counter > 0) {
                    ATLAS_cJSON_file << ",\n";  // Only print comma before subsequent entries
                }
                print_jet_to_json(current_jet, ATLAS_cJSON_file, found_IFN_b, found_IFN_c);
                ATLAS_c_counter += 1;
            } else {
                if (ATLAS_l_counter > 0) {
                    ATLAS_lJSON_file << ",\n";  // Only print comma before subsequent entries
                }
                print_jet_to_json(current_jet, ATLAS_lJSON_file, found_IFN_b, found_IFN_c);
                ATLAS_l_counter += 1;
            }

            if (found_IFN_b) {
                if (IFN_b_counter > 0) {
                    IFN_bJSON_file << ",\n";  // Only print comma before subsequent entries
                }
                print_jet_to_json(current_jet, IFN_bJSON_file, found_b, found_c);
                IFN_b_counter += 1;
            } else if (found_IFN_c) {
                if (IFN_c_counter > 0) {
                    IFN_cJSON_file << ",\n";  // Only print comma before subsequent entries
                }
                print_jet_to_json(current_jet, IFN_cJSON_file, found_b, found_c);
                IFN_c_counter += 1;
            } else {
                if (IFN_l_counter > 0) {
                    IFN_lJSON_file << ",\n";  // Only print comma before subsequent entries
                }
                print_jet_to_json(current_jet, IFN_lJSON_file, found_b, found_c);
                IFN_l_counter += 1;
            }

        }

        ++events_parsed;
        //cout << events_parsed << endl;
        if (events_parsed % 100 == 0) {
            std::cout << "Events parsed: " << events_parsed << std::endl;
        }
    }

    cout << "ATLAS b jets: " << ATLAS_b_counter << "\n";
    cout << "ATLAS c jets: " << ATLAS_c_counter << "\n";
    cout << "ATLAS l jets: " << ATLAS_l_counter << "\n";
    cout << "IFN b jets: " << IFN_b_counter << "\n";
    cout << "IFN c jets: " << IFN_c_counter << "\n";
    cout << "IFN l jets: " << IFN_l_counter << "\n";

    input_file.close();
    ATLAS_bJSON_file << "\n]\n";
    ATLAS_cJSON_file << "\n]\n";
    ATLAS_lJSON_file << "\n]\n";
    IFN_bJSON_file << "\n]\n";
    IFN_cJSON_file << "\n]\n";
    IFN_lJSON_file << "\n]\n";
    ATLAS_bJSON_file.close();
    ATLAS_cJSON_file.close();
    ATLAS_lJSON_file.close();
    IFN_bJSON_file.close();
    IFN_cJSON_file.close();
    IFN_lJSON_file.close();

    return 0;
}

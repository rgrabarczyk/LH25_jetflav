cmake_minimum_required(VERSION 3.12)
project(LesHouchesFlavour)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HepMC3
list(APPEND CMAKE_PREFIX_PATH "/cvmfs/sft.cern.ch/lcg/views/LCG_106b/x86_64-el9-gcc11-opt/share/HepMC3/cmake")
find_package(HepMC3 REQUIRED)

# Use fastjet-config to get FastJet flags
execute_process(COMMAND /eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/bin/fastjet-config --cxxflags
                OUTPUT_VARIABLE FASTJET_CXXFLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND /eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/bin/fastjet-config --libs
                OUTPUT_VARIABLE FASTJET_LIBS
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(FASTJET_CONTRIB_LIBS "/eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/lib/libIFNPlugin.a" "/eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/lib/libCMPPlugin.a" "/eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/lib/libGHSAlgo.a" "/eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/lib/libSDFlavPlugin.a" "/eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/lib/libRecursiveTools.a" "/eos/home-r/ragrabar/Flavour/LesHouches/fastjet-install/lib/libfastjetplugins.a")

# Add the executable
add_executable(NETflavour NETflavour.cpp helpers.cpp)
add_executable(LundNetjsons LundNetjsons.cpp helpers.cpp)

# Include HepMC3 and FastJet directories
include_directories(${HepMC3_INCLUDE_DIRS})
target_compile_options(NETflavour PRIVATE ${FASTJET_CXXFLAGS})
target_compile_options(LundNetjsons PRIVATE ${FASTJET_CXXFLAGS})

# Link libraries
target_link_libraries(NETflavour HepMC3::HepMC3 ${FASTJET_LIBS} ${FASTJET_CONTRIB_LIBS})
target_link_libraries(LundNetjsons HepMC3::HepMC3 ${FASTJET_LIBS} ${FASTJET_CONTRIB_LIBS})

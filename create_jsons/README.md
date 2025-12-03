Code for creating JSON files that looks like this

     [
     {
       "pt": 86.3693,                                           <-- jet pt
       "eta": -1.26271,                                         <-- jet eta
       "phi": 0.482061,                                         <-- jet phi
       "mass": 9.07214,                                         <-- jet mass
       "fourmomenta":[[1.45284,0.551922,-3.5868,3.90903], ... ] <-- fourmomenta of constituents
       "found_other_algo_b": 0                                  <-- these are perhaps a bit silly, I ran with pairs of algos at the time                                       
       "found_other_algo_c": 0                                  <-- and this tells you whether the other algo that you're running considered this a b or c jet
     }
     ,
     { ...

starting from HepMC files, perhaps from the generator. To compile, modify CMakeLists.txt with paths to your fastjet and fjcontrib libraries.
To run after compiling, simply do

    ./LundNetjsons < path to HepMC >

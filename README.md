<snippet>
  <content>
# Sarsa(lambda) and True Online Sarsa(lambda)

This repository is supposed to contain a snapshot of the code used to generate the results in the paper "Title". It contains an implementation of Sarsa(lambda) and an implementation of True Online Sarsa(lambda) on the Arcade Learning Environment (Bellemare et al., 2013).

## Installation

To be self-contained I added the ALE as well. The steps to install the code are below:

1. Go to lib/ale_0_4 and copy the file makefile.mac or makefile.unix to makefile and make it.
2. Go to src/ and make it. It should compile without any problems.
3. Go to research/true_online_sarsa/ and make it. It should compile without any problems.
4. Everything should be installed and working. Notice that if you are using osX you may have to export export the library path: DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4" (or export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../../lib/ale_0_4"

## Usage

In the correct directory (src for Sarsa or research/true_online_sarsa for True Online Sarsa), run the command ./learner. It should give you the parameters to use:

   -s     [REQUIRED] seed to random number generator.
   -c     [REQUIRED] path to file with configuration info.
   -r     [REQUIRED] path to the rom to be played by the agent.

The parameter -t is useless in this code. The file with configuration info is at conf/ (examples) while I do not have the rights to distribute the roms.

## Contributing

This code is not supposed to be further extended (not in this repository). It is an snapshot of the code used to generate the results in the paper "Title".

## Disclaimer

This code is a "research code", it was not implemented to be the most readable or reusable code. Only its relevant files are being published here. If you have any doubts, please let me know.

</content>
  <tabTrigger>readme</tabTrigger>
</snippet>

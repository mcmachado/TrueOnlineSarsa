# Makefile
#
# Author: Marlos C. Machado

ALE := ../lib/ale_0_4/

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 0

# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
FLAGS := -O3 -I$(ALE)/src -I/opt/local/include -L$(ALE) -D_GNU_SOURCE=1 -D_REENTRANT
CXX := g++
OUT_FILE := learner
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image 
endif

all: learner

learner: main.o Mathematics.o Parameters.o Timer.o Features.o Background.o BasicFeatures.o RLLearner.o SarsaLearner.o
	$(CXX) $(FLAGS) bin/main.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/RLLearner.o bin/SarsaLearner.o $(LDFLAGS) -o $(OUT_FILE) 

main.o: main.cpp
	$(CXX) $(FLAGS) -c main.cpp -o bin/main.o

Mathematics.o: common/Mathematics.cpp
	$(CXX) $(FLAGS) -c common/Mathematics.cpp -o bin/Mathematics.o

Timer.o: common/Timer.cpp
	$(CXX) $(FLAGS) -c common/Timer.cpp -o bin/Timer.o

Parameters.o: common/Parameters.cpp
	$(CXX) $(FLAGS) -c common/Parameters.cpp -o bin/Parameters.o

Features.o: features/Features.cpp
	$(CXX) $(FLAGS) -c features/Features.cpp -o bin/Features.o

Background.o: features/Background.cpp
	$(CXX) $(FLAGS) -c features/Background.cpp -o bin/Background.o
	
BasicFeatures.o: features/BasicFeatures.cpp
	$(CXX) $(FLAGS) -c features/BasicFeatures.cpp -o bin/BasicFeatures.o

RLLearner.o: agents/rl/RLLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/RLLearner.cpp -o bin/RLLearner.o

SarsaLearner.o: agents/rl/sarsa/SarsaLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/sarsa/SarsaLearner.cpp -o bin/SarsaLearner.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"

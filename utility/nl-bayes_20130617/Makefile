# C source code
CSRC	= ./Utilities/io_png.c \
		./Utilities/mt19937ar.c
# C++ source code
CXXSRC	= main.cpp \
		./NlBayes/NlBayes.cpp \
		./NlBayes/LibMatrix.cpp \
		./Utilities/LibImages.cpp \
		./Utilities/Utilities.cpp

# all source code
SRC	= $(CSRC) $(CXXSRC)

# C objects
COBJ	= $(CSRC:.c=.o)
# C++ objects
CXXOBJ	= $(CXXSRC:.cpp=.o)
# all objects
OBJ	= $(COBJ) $(CXXOBJ)
# binary target
BIN	= NL_Bayes

# C optimization flags
COPT	= -O3 -ftree-vectorize -funroll-loops

# C++ optimization flags
CXXOPT	= $(COPT)

# C compilation flags
CFLAGS	= $(COPT) -Wall -Wextra \
	-Wno-write-strings -ansi
# C++ compilation flags
CXXFLAGS	= $(CXXOPT) -Wall -Wextra \
	-Wno-write-strings -Wno-deprecated -ansi
# link flags. for brew install on macOS of libpng use /opt/homebrew/lib/libpng.dylib
LDFLAGS	= -L/opt/homebrew/lib/libpng.dylib -lm

# use openMP with `make OMP=1`
ifdef OMP
CFLAGS	+= -fopenmp
CXXFLAGS	+= -fopenmp
LDFLAGS += -lgomp
else
CFLAGS	+= -Wno-unknown-pragmas
CXXFLAGS  += -Wno-unknown-pragmas
endif

# partial compilation of C source code
%.o: %.c %.h
	$(CC) -c -o $@  $< $(CFLAGS)
# partial compilation of C++ source code
%.o: %.cpp %.h
	$(CXX) -c -o $@  $< $(CXXFLAGS)

# link all the object code
$(BIN): $(OBJ) $(LIBDEPS)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)
python_plotter := python -c 'from pylab import*;X=loadtxt(sys.stdin);scatter(X[0],X[1]);title("Embedding");grid();show()'
plotter := $(python_plotter)

default:
	@(git submodule update --init)
	@(mkdir -p build; cd build; cmake -DBUILD_EXAMPLES=ON ..; make)

install: default
	@(cd build; make install)

test:
	@(git submodule update --init)
	@(mkdir -p build; cd build; cmake -DBUILD_TESTS=ON ..; make)
	@(cd build; ctest -VV)

codeql:
	@(mkdir -p build; cd build; cmake -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..; make)

minimal: default
	@(echo '--- Code ---';               \
	  cat ./examples/minimal/minimal.cpp;\
	  echo '--- Description ---';        \
	  cat ./examples/minimal/minimal.md; \
	  echo '--- Running ---';            \
	  echo 'Result is: ';                \
	  ./bin/minimal)

rna: default
	@(echo '--- Code ---';                        \
	  cat ./examples/rna/rna.cpp;                 \
	  echo '--- Description ---';                 \
	  cat ./examples/rna/rna.md;                  \
	  echo '--- Plotting ---';                    \
	  ./bin/rna examples/rna/rna.dat | $(plotter) \
	  );

precomputed: default
	@(echo '--- Code ---';                       \
	  cat ./examples/precomputed/precomputed.cpp;\
	  echo '--- Description ---';                \
	  cat ./examples/precomputed/precomputed.md; \
	  echo '--- Running ---';                    \
	  echo 'Result is: ';                        \
	  ./bin/precomputed)

langs:
	@(if (python -c 'from modshogun import LocallyLinearEmbedding' > /dev/null 2>&1); \
	  then                                         \
	    echo '--- Description ---';                \
	    cat ./examples/langs/langs.md;     \
	    echo '--- Python example ---';  \
	    cat examples/langs/lle.py;    \
	    echo '--- Running ---';      \
	    python examples/langs/lle.py;    \
	    echo '--- Octave example ---';  \
	    cat examples/langs/ltsa.m;    \
	    echo '--- Running ---';      \
	    octave examples/langs/ltsa.m;    \
	  else                                         \
	    echo 'Shogun machine learning toolbox is not installed or compiled without Tapkee (may lack some dependencies)' \
	         ' (https://github.com/shogun-toolbox/shogun)';     \
	  fi;)

promoters:
	@(if (python -c 'from modshogun import LocallyLinearEmbedding' > /dev/null 2>&1); \
	  then                                         \
	    echo '--- Description ---';                \
	    cat ./examples/promoters/promoters.md;     \
	    echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';  \
	    python examples/promoters/promoters.py data/mml.txt;    \
	  else                                         \
	    echo 'Shogun machine learning toolbox is not installed or compiled without Tapkee (may lack some dependencies)' \
	         ' (https://github.com/shogun-toolbox/shogun)';     \
	  fi;)

mnist: default
	@(echo '--- Description ---';               \
	  cat ./examples/mnist/mnist.md;            \
	  echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';    \
	  python examples/mnist/mnist.py data/mnist.json)

faces: default
	@(echo '--- Description ---';               \
	  cat ./examples/faces/faces.md;              \
	  echo '--- Embedding and plotting (please wait, a window will appear in a few seconds) ---';    \
	  python examples/faces/faces.py data/faces)

format: default
	@(find . -iname *.hpp -o -iname *.cpp -iname *.h | xargs clang-format -i)

pip-package:
	pip wheel packages/python -w dist

.PHONY: test minimal rna precomputed promoters mnist faces pip-package

python_plotter := python -c 'from pylab import*;X=loadtxt(sys.stdin);scatter(X[0],X[1]);show()'
plotter := $(python_plotter)

default:
	@(mkdir -p build; cd build; cmake ..; make)

test: default
	@(cd build; ctest -VV)

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

promoters:
	@(if (python -c 'from modshogun import LocallyLinearEmbedding' > /dev/null 2>&1); \
	  then                                         \
	    echo '--- Description ---';                \
	    cat ./examples/promoters/promoters.md;     \
	    echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';  \
	    python examples/promoters/promoters.py;    \
	  else                                         \
	    echo 'Shogun machine learning toolbox is not installed' \
	         ' (https://github.com/shogun-toolbox/shogun)';     \
	  fi;)

mnist: default
	@(echo '--- Description ---';               \
	  cat ./examples/mnist/mnist.md;            \
	  echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';    \
	  python examples/mnist/mnist.py data/mnist2000.dat)

cbcl: default
	@(echo '--- Description ---';               \
	  cat ./examples/cbcl/cbcl.md;              \
	  echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';    \
	  python examples/cbcl/cbcl.py)

faces: default
	@(echo '--- Description ---';               \
	  cat ./examples/faces/faces.md;              \
	  echo '--- Embedding and plotting (please wait, a window will appear in a few seconds) ---';    \
	  python examples/faces/faces.py)

.PHONY: test minimal rna precomputed promoters

python_plotter := python -c 'from pylab import*;X=loadtxt(sys.stdin);scatter(X[0],X[1]);show()'
plotter := $(python_plotter)

default:
	@(mkdir -p build; cd build; cmake ..; make)

test: default
	@(cd build; ctest -VV)

minimal: default
	@(echo '--- Description ---';        \
	  cat ./examples/minimal/minimal.md; \
	  echo '--- Running ---';            \
	  echo 'Result is: ';                \
	  ./bin/minimal)

rna: default
	@(echo '--- Description ---';                 \
	  cat ./examples/rna/rna.md;                  \
	  echo '--- Plotting ---';                    \
	  ./bin/rna examples/rna/rna.dat | $(plotter) \
	  );

precomputed: default
	@(echo '--- Description ---';                \
	  cat ./examples/precomputed/precomputed.md; \
	  echo '--- Running ---';                    \
	  echo 'Result is: ';                        \
	  ./bin/precomputed)

.PHONY: test

PYTHON ?= python
plotter := $(PYTHON) -c 'from pylab import*;X=loadtxt(sys.stdin);scatter(X[0],X[1]);title("Embedding");grid();show()'

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
	@(if ($(PYTHON) -c 'from modshogun import LocallyLinearEmbedding' > /dev/null 2>&1); \
	  then                                         \
	    echo '--- Description ---';                \
	    cat ./examples/langs/langs.md;     \
	    echo '--- Python example ---';  \
	    cat examples/langs/lle.py;    \
	    echo '--- Running ---';      \
	    $(PYTHON) examples/langs/lle.py;    \
	    echo '--- Octave example ---';  \
	    cat examples/langs/ltsa.m;    \
	    echo '--- Running ---';      \
	    octave examples/langs/ltsa.m;    \
	  else                                         \
	    echo 'Shogun machine learning toolbox is not installed or compiled without Tapkee (may lack some dependencies)' \
	         ' (https://github.com/shogun-toolbox/shogun)';     \
	  fi;)

promoters:
	@(if ($(PYTHON) -c 'from modshogun import LocallyLinearEmbedding' > /dev/null 2>&1); \
	  then                                         \
	    echo '--- Description ---';                \
	    cat ./examples/promoters/promoters.md;     \
	    echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';  \
	    $(PYTHON) examples/promoters/promoters.py data/mml.txt;    \
	  else                                         \
	    echo 'Shogun machine learning toolbox is not installed or compiled without Tapkee (may lack some dependencies)' \
	         ' (https://github.com/shogun-toolbox/shogun)';     \
	  fi;)

mnist: default
	@(echo '--- Description ---';               \
	  cat ./examples/mnist/mnist.md;            \
	  echo '--- Embedding and plotting (please wait, a window will appear in a minute) ---';    \
	  $(PYTHON) examples/mnist/mnist.py data/mnist.json)

faces: default
	@(echo '--- Description ---';               \
	  cat ./examples/faces/faces.md;              \
	  echo '--- Embedding and plotting (please wait, a window will appear in a few seconds) ---';    \
	  $(PYTHON) examples/faces/faces.py data/faces)

format: default
	@(find . -iname *.hpp -o -iname *.cpp -iname *.h | xargs clang-format -i)

VENV_BUILD = .venv-build
VENV_TEST = .venv-test
SMOKE_TEST = import tapkee; import numpy as np; r = tapkee.embed(np.random.randn(3, 50), method='pca'); assert r.shape == (50, 2); print('OK')

$(VENV_BUILD):
	$(PYTHON) -m venv $(VENV_BUILD)
	$(VENV_BUILD)/bin/pip install -q build

$(VENV_TEST):
	$(PYTHON) -m venv $(VENV_TEST)

pip-package:
	$(PYTHON) -m pip wheel packages/python -w dist

pip-sdist: $(VENV_BUILD)
	rm -rf .sdist-work dist/tapkee-*.tar.gz
	cp -rL packages/python .sdist-work
	cd .sdist-work && ../$(VENV_BUILD)/bin/python -m build --sdist -o ../dist
	rm -rf .sdist-work

test-pip-package: pip-package $(VENV_TEST)
	$(VENV_TEST)/bin/pip install --force-reinstall --no-index --find-links dist tapkee
	$(VENV_TEST)/bin/python -c "$(SMOKE_TEST)"

test-pip-sdist: pip-sdist $(VENV_TEST)
	$(VENV_TEST)/bin/pip install --force-reinstall dist/tapkee-*.tar.gz
	$(VENV_TEST)/bin/python -c "$(SMOKE_TEST)"

clean-venvs:
	rm -rf $(VENV_BUILD) $(VENV_TEST)

R_SMOKE_TEST = library(tapkee); X <- matrix(rnorm(300), 3, 100); r <- tapkee_embed(X, method='pca'); stopifnot(dim(r) == c(100, 2)); cat('OK\n')

r-package:
	R CMD build packages/r

r-check: r-package
	R CMD check --as-cran tapkee_*.tar.gz

r-install:
	R CMD INSTALL packages/r

test-r-package: r-install
	Rscript -e "$(R_SMOKE_TEST)"

.PHONY: test minimal rna precomputed promoters mnist faces pip-package test-pip-package pip-sdist test-pip-sdist clean-venvs r-package r-check r-install test-r-package

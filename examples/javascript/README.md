# JavaScript (WebAssembly) example

Tapkee compiles to WebAssembly with [Emscripten](https://emscripten.org). The
`BUILD_JS` CMake option builds `tapkee.js` + `tapkee.wasm` from the embind
wrapper in `src/js/tapkee_js.cpp`, and copies this demo page next to them.

ARPACK is not available in WebAssembly builds; tapkee automatically falls back
to its Dense (exact, Eigen) and Randomized eigensolvers.

## Build

```bash
mkdir build-js && cd build-js
emcmake cmake -DBUILD_JS=ON ..
cmake --build . --target tapkee_js
```

The output lands in `bin/`: `tapkee.js`, `tapkee.wasm`, and `index.html`
(this demo).

## Run the demo

```bash
python3 -m http.server -d ../bin
```

Open http://localhost:8000 — a swiss roll is generated in JavaScript and
embedded to 2D by tapkee running in the browser (LLE, Isomap, Hessian LLE,
Linear LTSA, Laplacian Eigenmaps, PCA, MDS, Diffusion Map).

## Use from JavaScript

The module works in browsers and Node alike:

```js
const createTapkee = require('./tapkee.js');

createTapkee().then((tapkee) => {
    // data is a flat array, point-major: [x0, y0, z0, x1, y1, z1, ...]
    const result = tapkee.embed(data, nPoints, nDims, {
        method: 'lle',       // same short names as the CLI (see tapkee -h)
        numNeighbors: 12,
        targetDimension: 2,
    });
    // result.embedding is flat point-major, result.rows x result.cols
});
```

Supported options mirror the Python bindings: `method`, `neighborsMethod`,
`eigenMethod`, `numNeighbors`, `targetDimension`, `gaussianKernelWidth`,
`landmarkRatio`, `maxIteration`, `diffusionMapTimesteps`, `snePerplexity`,
`sneTheta`, `squishingRate`, `speGlobalStrategy`, `speNumUpdates`,
`speTolerance`, `nullspaceShift`, `klleShift`, `faEpsilon`,
`checkConnectivity`.

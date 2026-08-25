# tapkee

[Tapkee](https://github.com/lisitsyn/tapkee) — a C++ dimensionality reduction
library — compiled to WebAssembly. Runs in browsers and in Node, with no
server and no native dependencies.

## Install

```bash
npm install tapkee
```

Or load it straight from a CDN, no build step required:

```html
<script src="https://cdn.jsdelivr.net/npm/tapkee/dist/tapkee.js"></script>
```

## Use

```js
const createTapkee = require('tapkee'); // or the global from the <script> tag

const tapkee = await createTapkee();

// data is a flat array, point-major: [x0, y0, z0, x1, y1, z1, ...]
const result = tapkee.embed(data, nPoints, nDims, {
    method: 'lle',
    numNeighbors: 12,
    targetDimension: 2,
});

// result.embedding is flat, result.rows x result.cols
```

Errors are thrown as ordinary JavaScript exceptions:

```js
tapkee.embed(data, 800, 3, { method: 'lle', numNeighbors: 5000 });
// Error: [3, 800) range check failed for number of neighbors, its value is 5000
```

## Methods

Pass any of these as `method` (the same short names the CLI accepts):

`lle`, `ltsa`, `hlle`, `lltsa`, `npe`, `lpp`, `isomap`, `l-isomap`, `mds`,
`l-mds`, `dm`, `la`, `pca`, `kpca`, `ra`, `fa`, `spe`, `t-sne`,
`manifold_sculpting`, `passthru`

## Options

`method`, `neighborsMethod`, `eigenMethod`, `numNeighbors`, `targetDimension`,
`gaussianKernelWidth`, `landmarkRatio`, `maxIteration`, `diffusionMapTimesteps`,
`snePerplexity`, `sneTheta`, `squishingRate`, `speGlobalStrategy`,
`speNumUpdates`, `speTolerance`, `nullspaceShift`, `klleShift`, `faEpsilon`,
`checkConnectivity`

ARPACK is not available in WebAssembly builds, so the `Dense` eigensolver is
the default; pass `eigenMethod: 'randomized'` for larger problems.

## Demo

`examples/javascript` in the repository embeds a swiss roll in the browser and
shows the call it makes for every run.

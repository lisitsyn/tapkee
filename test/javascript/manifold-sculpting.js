// Regression test and reproducible benchmark for the WebAssembly optimizer.
// node test/javascript/manifold-sculpting.js
// node test/javascript/manifold-sculpting.js --run /absolute/path/tapkee.js 800 100 123456789
const assert = require('node:assert/strict');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function swissRoll(n, seed) {
    const data = [], intrinsic = [];
    function random() {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 4294967296;
    }
    for (let i = 0; i < n; i++) {
        const t = 1.5 * Math.PI * (1 + 2 * random()), h = 20 * random();
        data.push(t * Math.cos(t), h, t * Math.sin(t));
        // Arc length along the spiral gives the flat sheet's true coordinates.
        intrinsic.push([0.5 * (t * Math.sqrt(1 + t * t) + Math.asinh(t)), h]);
    }
    return { data, intrinsic };
}

function quality(data, intrinsic, embedding, k) {
    const n = intrinsic.length;
    let localError = 0, localDistance = 0, trueSquared = 0, embeddedSquared = 0, cross = 0;
    for (let i = 0; i < n; i++) {
        const neighbors = [];
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const original = Math.hypot(...[0, 1, 2].map(d => data[3 * i + d] - data[3 * j + d]));
            const embedded = Math.hypot(embedding[2 * i] - embedding[2 * j],
                                        embedding[2 * i + 1] - embedding[2 * j + 1]);
            neighbors.push({ original, embedded });
            if (j > i) {
                const truth = Math.hypot(intrinsic[i][0] - intrinsic[j][0], intrinsic[i][1] - intrinsic[j][1]);
                trueSquared += truth * truth;
                embeddedSquared += embedded * embedded;
                cross += truth * embedded;
            }
        }
        neighbors.sort((a, b) => a.original - b.original);
        for (const { original, embedded } of neighbors.slice(0, k)) {
            localDistance += original;
            localError += (embedded - original) ** 2;
        }
    }
    return {
        localDistanceRmse: Math.sqrt(localError / (n * k)) / (localDistance / (n * k)),
        // Pairwise distance stress after fitting one global scale; lower is better.
        intrinsicDistanceStress: Math.sqrt(Math.max(0, 1 - cross * cross / (embeddedSquared * trueSquared))),
    };
}

async function run(modulePath, n, maxIteration, seed) {
    const tapkee = await require(modulePath)();
    const { data, intrinsic } = swissRoll(n, seed);
    const start = performance.now();
    const result = tapkee.embed(data, n, 3, {
        method: 'manifold_sculpting', numNeighbors: 12, targetDimension: 2, maxIteration,
    });
    const ms = performance.now() - start;
    assert.equal(result.rows, n);
    assert.equal(result.cols, 2);
    assert.equal(result.embedding.length, n * 2);
    assert(result.embedding.every(Number.isFinite), 'coordinates must be finite');
    assert(Math.max(...result.embedding) > Math.min(...result.embedding), 'embedding must not collapse');
    return { n, maxIteration, seed, ms, ...quality(data, intrinsic, result.embedding, 12) };
}

if (process.argv[2] === '--run') {
    run(path.resolve(process.argv[3]), Number(process.argv[4]), Number(process.argv[5]), Number(process.argv[6]))
        .then(result => console.log(JSON.stringify(result)))
        .catch(error => { console.error(error); process.exitCode = 1; });
} else {
    // A subprocess deadline can interrupt synchronous WASM; a JS timer cannot.
    // This is a runaway guard, deliberately generous rather than a speed benchmark.
    const child = spawnSync(process.execPath, [__filename, '--run',
        path.join(__dirname, '../../bin/tapkee.js'), '400', '100', '123456789'],
        { timeout: 30000, encoding: 'utf8' });
    assert.ifError(child.error);
    assert.equal(child.status, 0, child.stderr);
    const result = JSON.parse(child.stdout);
    assert(Number.isFinite(result.localDistanceRmse));
    assert(Number.isFinite(result.intrinsicDistanceStress));
    console.log('Manifold Sculpting regression passed:', result);
}

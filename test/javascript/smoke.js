// Smoke test for the WebAssembly module (bin/tapkee.js, built with BUILD_JS).
// Run with: node test/javascript/smoke.js

const path = require('path');
const createTapkee = require(path.join(__dirname, '..', '..', 'bin', 'tapkee.js'));

function swissRoll(n) {
    const data = [];
    for (let i = 0; i < n; i++) {
        const t = 1.5 * Math.PI * (1 + 2 * Math.random());
        const h = 20 * Math.random();
        data.push(t * Math.cos(t), h, t * Math.sin(t));
    }
    return data;
}

function assert(condition, message) {
    if (!condition) {
        throw new Error('FAILED: ' + message);
    }
}

createTapkee().then((tapkee) => {
    const n = 400;
    const data = swissRoll(n);

    for (const method of ['lle', 'isomap', 'hlle', 'lltsa', 'la', 'pca', 'mds', 'dm']) {
        const result = tapkee.embed(data, n, 3, { method, numNeighbors: 12, targetDimension: 2 });
        assert(result.rows === n && result.cols === 2, `${method}: expected ${n}x2, got ${result.rows}x${result.cols}`);
        assert(result.embedding.length === n * 2, `${method}: embedding length ${result.embedding.length}`);
        assert(result.embedding.every(Number.isFinite), `${method}: embedding contains non-finite values`);
        const spread = Math.max(...result.embedding) - Math.min(...result.embedding);
        assert(spread > 0, `${method}: embedding is degenerate (zero spread)`);
        console.log(`${method}: ok (${result.rows}x${result.cols})`);
    }

    const randomized = tapkee.embed(data, n, 3, { method: 'isomap', numNeighbors: 12, eigenMethod: 'randomized' });
    assert(randomized.rows === n && randomized.cols === 2, 'randomized eigen method');
    console.log('isomap with randomized eigensolver: ok');

    const defaults = tapkee.embed(data, n, 3, {});
    assert(defaults.rows === n, 'default options');
    console.log('default options: ok');

    for (const [label, fn] of [
        ['unknown method', () => tapkee.embed(data, n, 3, { method: 'nope' })],
        ['mismatched data length', () => tapkee.embed(data, n - 1, 3, { method: 'lle' })],
        ['out-of-range neighbors', () => tapkee.embed(data, n, 3, { method: 'lle', numNeighbors: n * 10 })],
    ]) {
        let thrown = false;
        try { fn(); } catch (e) { thrown = true; }
        assert(thrown, `${label} should throw`);
        console.log(`${label} throws: ok`);
    }

    console.log('All smoke tests passed');
}).catch((e) => {
    console.error(e.message || e);
    process.exit(1);
});

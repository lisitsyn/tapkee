// Exercise the real demo script with controlled workers and a minimal DOM.
// Run with: node test/javascript/demo.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../../examples/javascript/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function startDemo() {
    const elements = new Map();
    const workers = [];
    function element() {
        return {
            value: '', textContent: '', clientWidth: 400, clientHeight: 400,
            listeners: {},
            classList: { add() {}, remove() {} },
            addEventListener(name, callback) { this.listeners[name] = callback; },
            appendChild() {},
            getContext() { return new Proxy({}, { get: () => () => {} }); },
        };
    }
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, element());
            return elements.get(id);
        },
        querySelector: () => element(),
        createElement: () => element(),
    };
    class MockWorker {
        constructor() { this.messages = []; this.terminated = false; workers.push(this); }
        postMessage(message) { this.messages.push(message); }
        terminate() { this.terminated = true; }
        finish() {
            const request = this.messages.at(-1);
            this.onmessage({ data: {
                id: request.id, ok: true, ms: 1,
                embedding: Array.from({ length: request.n * 2 }, (_, i) => i % 13),
            } });
        }
    }
    const context = vm.createContext({
        document, Worker: MockWorker, Blob, URL,
        location: { href: 'http://localhost/index.html' },
        window: { devicePixelRatio: 1, addEventListener() {} },
        ResizeObserver: class { observe() {} },
        requestAnimationFrame() {},
    });
    vm.runInContext(script, context);
    return {
        workers, elements, workerScript: vm.runInContext('workerSrc', context),
        change(id, value) {
            const el = document.getElementById(id);
            el.value = value;
            el.listeners.change();
        },
        close() { URL.revokeObjectURL(vm.runInContext('workerURL', context)); },
    };
}

const demo = startDemo();
try {
    const first = demo.workers[0];
    assert.equal(first.messages.length, 1);
    demo.change('method', 'manifold_sculpting');
    const second = demo.workers.at(-1);
    assert.notEqual(first, second, 'changing settings must replace a busy worker');
    assert(first.terminated, 'obsolete computation must be terminated');
    assert.equal(second.messages[0].options.method, 'manifold_sculpting');

    demo.change('npoints', '400');
    const latest = demo.workers.at(-1);
    assert(second.terminated);
    assert.equal(latest.messages[0].n, 400);
    assert.equal(demo.workers.filter(w => !w.terminated).length, 1);
    first.finish();
    second.onerror({ message: 'obsolete failure' });
    assert.equal(demo.elements.get('status').textContent, 'Computing…', 'stale events must be ignored');
    latest.finish();
    assert.equal(demo.elements.get('status').textContent, 'Computed in 1 ms');

    demo.change('kslider', '15');
    assert.equal(demo.workers.at(-1), latest, 'completed workers should be reused');
    assert.equal(latest.messages.at(-1).options.numNeighbors, 15);
    latest.onerror({ message: 'worker failed' });
    assert.equal(demo.elements.get('status').textContent, 'worker failed');
    assert(latest.terminated);
    demo.change('method', 'pca');
    const recovered = demo.workers.at(-1);
    assert.notEqual(recovered, latest, 'the next request must recover from a worker failure');
    recovered.finish();
    assert.equal(demo.elements.get('status').textContent, 'Computed in 1 ms');

    demo.change('method', 'manifold_sculpting');
    const failedRequest = recovered.messages.at(-1);
    recovered.onmessage({ data: { id: failedRequest.id, ok: false, error: 'module failed to load' } });
    assert.equal(demo.elements.get('status').textContent, 'module failed to load');
    assert(recovered.terminated, 'a rejected initialization must not poison future requests');
    demo.change('method', 'pca');
    assert.notEqual(demo.workers.at(-1), recovered);
    demo.workers.at(-1).finish();
    assert.equal(demo.elements.get('status').textContent, 'Computed in 1 ms');
    console.log('Demo cancellation, stale events, reuse, and recovery: passed');
} finally {
    demo.close();
}

async function checkInitializationFailure() {
    const replies = [];
    const context = vm.createContext({
        importScripts() {},
        createTapkee: () => Promise.reject(new Error('module failed to load')),
        postMessage: message => replies.push(message),
    });
    vm.runInContext(demo.workerScript, context);
    await context.onmessage({ data: { id: 7 } });
    assert.equal(replies.length, 1);
    assert.equal(replies[0].id, 7);
    assert.equal(replies[0].ok, false);
    assert.equal(replies[0].error, 'module failed to load');
    console.log('Worker initialization failure: passed');
}
checkInitializationFailure().catch(error => { console.error(error); process.exitCode = 1; });

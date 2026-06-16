---
title: Content-Addressed Caching for Reactive Notebooks
short_title: Paper
---

```{marimo-config}
---
echo: false
output: true
header: |
  import sys, os
  sys.path.insert(0, '/Users/dmadisetti/src/scipy_proceedings/papers/madisetti_cache')
  os.chdir('/Users/dmadisetti/src/scipy_proceedings/papers/madisetti_cache')
pyproject: |
  requires-python = ">=3.12"
  dependencies = [
      "marimo>=0.23.8",
      "matplotlib==3.10.9",
      "numpy==2.4.6",
      "diskcache==5.6.3",
      "joblib>=1.4",
      "pymandala>=0.2.0a0",
      "pyarrow>=15",
      "tqdm",
      "prettytable",
      "graphviz",
      "ipython",
  ]
---
```



```{marimo} python
:name: setup
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
import platform
import tempfile
import diskcache
import mandala.imports  # noqa: F401 — submodule import; cells use `mandala.imports.{Storage, op}`

# Bench primitives, plot helpers, and `compute.claim(...)` used by the
# figure cells. Importing `lib` also applies a temporary monkeypatch
# for PR #8805 (`lib/_marimo_patch.py`); without it the decorator-path
# hit latency is fake-flat. Delete the patch file once the fix ships.
from lib import compute, save_fig
```

+++ {"part": "abstract"}

We describe the caching mechanism behind resumable sessions in marimo, a reactive Python notebook.
Cache keys are derived recursively from the reactive DAG: references are content-addressed where their bytes are reachable, and the producing cell's key is substituted where they are not, a Merkle-style recurrence that an edit invalidates at exactly the subtree it dominates.
Keys are computed over compiled bytecode rather than source text, and the notebook that builds this paper re-derives the resulting invariances — to comments, formatting, cell reordering, and cell identifiers — against the implementation on every render.
The same keys make results durable and portable: cached values cross process and session boundaries through a lazy stub mechanism and ship inside marimo's static WASM/HTML export, so a reader's browser rehydrates results — even values produced by libraries the browser cannot import — with no local Python installation.
Microbenchmarks against representative baselines validate that the overhead stays interactive-class: hits on exploratory-scale payloads remain under the 100 ms threshold, and measuring the write path yields a break-even rule for when caching pays.
The mechanism stays native to the reactive notebook and requires no per-call-site opt-in.

+++

(sec:intro)=
# Introduction

Notebooks underpin scientific Python, but most of them do not survive a clean rerun.
In a survey of 1.4 million public Jupyter notebooks, only about a quarter re-execute top-to-bottom without raising [@pimentel2019largescale].
The cells a reader sees record what the author once ran, not what the notebook does now; out-of-order execution and hidden state mean reproducing a result requires replaying an execution order the notebook does not record.

Reactive notebooks mostly address this gap by treating cells, a unit of source code, as nodes in a dataflow graph.
From the variables a cell reads (its *references*, `refs`) and the variables it binds (its *definitions*, `defs`), execution order is determined by data dependencies rather than source order, and editing a cell re-executes its dependents.
Because the relation is derived statically rather than from a runtime trace, hidden state is hard to sustain and out-of-order execution is impossible at the cell level.
Notable reactive notebooks include Pluto.jl [@vanderplas2020pluto], Observable [@bostock2017observable], and Livebook [@valim2020livebook], while IPyflow and nbsafety [@macke2021nbsafety;@ipyflow] retrofit something close onto Jupyter through runtime dependency inference and program slicing.
The lineage descends from a longer tradition of direct-manipulation programming environments [@victor2012inventing], in which editing the source is itself the act that updates the running program.
marimo [@marimo] reinvents the reactive notebook for Python and is the focus of this paper.

Reactivity has a cost: where Jupyter lets a user re-evaluate just enough cells to repopulate memory, a dataflow pipeline reruns from its roots, so every session pays for every expensive cell.
Yet a reactive notebook already knows what each cell depends on, and recomputation can be skipped wherever the cell body is deterministic.

Caching for notebooks and Python is not new.
IncPy modifies the CPython interpreter to memoize function calls automatically [@guo2011incpy]; knitr caches literate-document chunks with hand-declared dependency chains [@xie2015knitr]; jupyter-cache re-executes a notebook wholesale when any code cell changes [@jupytercache].
`mandala` memoizes decorated calls inside a `with storage:` block [@makelov2024mandala], Kishu checkpoints whole sessions for time travel [@li2025kishu], ElasticNotebook migrates live state across machines [@li2024elasticnotebook], and diskcache provides a byte-keyed store at the bottom of the stack [@diskcache].
Each asks the user to opt in at a different boundary.
Starting from marimo's reactive DAG produces a cache the user does not have to opt into per call site, because the same parse pass that derives a cell's references also derives the inputs to its key — knitr's hand-maintained dependency declarations arrive for free.

Concretely we target three properties.
(a) Skip expensive recomputation when references and source are unchanged.
(b) Preserve reactive determinism within a session under reordering and partial reruns.
(c) Make cached artifacts transportable through marimo's static WASM/HTML export.
Out of scope are full session restoration in the sense of [@li2025kishu], distributed execution, and reproducibility of arbitrary Python notebooks [@pimentel2019largescale]; we claim deterministic reuse within a reactive session, a strictly smaller and more defensible target.

(sec:background)=
# Background and Related Work

Michie's memo functions [@michie1968memo] are the canonical outline of "function caching": skip recomputation when an input-derived key matches a stored key.
Caching a *cell* forces the question of what a cell's "inputs" are.
A reactive notebook's static derivation gives a graph that is stable across runs, so the key must be a function of the graph the source defines — not of a particular execution trace — together with the values bound at evaluation time.
Cell-level dataflow tracking has an earlier antecedent in [@koop2017dataflow], and Rex [@zheng2025reactive] probes the boundaries of reactivity in marimo specifically.

To build a key from a cell's refs, we look to build systems.
Build Systems à la Carte [@mokhov2018build] decomposes a build system into a *rebuilder* (deciding when to rerun) and a *scheduler* (deciding the rebuild order); marimo's caching is a content-hash rebuilder paired with a reactive scheduler, with no persistent build trace.
We borrow the recursive-hash derivation lookup from Nix [@dolstra2004nix;@dolstra2006purely] and apply it to a reactive notebook instead of a static package graph — closer to Nix's input-addressed model than to Shake's verifying traces.
The data-engineering analog is Bauplan/Nessie's pipeline-stage hashing [@tagliabue2024bauplan], and workflow engines persist the same discipline at task granularity: Nextflow's `-resume` and Snakemake's `--cache` key results on code, parameters, and input hashes [@ditommaso2017nextflow;@molder2021snakemake].
Beneath the systems layer, self-adjusting computation and Adapton supply the change-propagation theory that DAG memoization specializes [@acar2002adaptive;@hammer2014adapton].

For existing scientific-Python memoization, mandala [@makelov2024mandala] is the closest analog at this venue (SciPy 2024); it memoizes calls inside a `with storage:` context using `joblib.hash` for content addressing and persists call provenance.
signac [@adorf2018signac] is an earlier parameter-hashed predecessor, diskcache [@diskcache] is the byte-keyed control where content addressing is the user's responsibility, and joblib's `Memory` [@joblib], scientific Python's standard persistent memoizer, keys on pickled arguments per decorated function.
Notebook-state work is complementary: Kishu [@li2025kishu] snapshots co-variable groups for session time travel, ElasticNotebook [@li2024elasticnotebook] migrates state live, and noWorkflow [@pimentel2017noworkflow] traces execution into a queryable provenance DAG.
Checkpoints store *state* where a cache stores *results*, and the two compose.
Two concurrent systems bracket ours: NBRewind checkpoints notebook cells incrementally to reproduce distributed workflows [@azaz2026nbrewind], and Chordata formalizes shortcut memoization for live re-execution at the interpreter level [@kirisame2026chordata].
Neither derives keys from a reactive graph — the property the rest of this paper exploits.

(sec:keys)=
# Cache Key Construction

For computational caching, false positive hits are unacceptable and false negatives merely undesirable: key derivation must be sensitive to value changes yet robust to superficial notebook edits.
A naive key would hash every ref's value directly.
This does not survive Python's exposure of mutation and FFI — pointers move under realloc, weakrefs report identity rather than content, and opaque C-extension objects expose neither a buffer nor a stable repr — so value-only keys generate false negatives as process state shifts beneath an unchanged notebook.
Hashing the cell's source alone fails in the other direction: cells that read the filesystem, network, or wall clock — and marimo's (heavily discouraged) in-place mutation — would be invisible to the key, generating false positives.
Build systems face the same dilemma and substitute the producer when the artifact is opaque [@dolstra2004nix]; marimo's cache key follows the same discipline, derived recursively over the reactive DAG.
Each cell's key depends on (a) the cell's compiled body — bytecode rather than source text, so comments and formatting do not participate — (b) a content address for every reference the cell reads, and (c) the keys of the parent cells that own those references.
Some refs cannot be addressed directly (mutable state, side-effecting refs, large values without a typed buffer protocol).
For those we substitute the parent-cell hash; when an input cannot be addressed directly, address the producer that defines it.

(sec:dispatch)=
## Key dispatch

:::{figure} figs/fig1_dispatch.png
:label: fig:dispatch
:width: 100%

Cache key construction.
Left: the per-ref dispatch, a three-way fallback into **Pure** (hash the value), **ContentAddressed** (hash the buffer), or **ExecutionPath** (substitute the producing cell's $H$); the first match emits the per-ref key, and every per-ref key feeds one sha256 combiner with the cell's compiled-body hash to produce $H(c)$.
Right: the derivation over the full cell, abridged from `BlockHasher.__init__` (`marimo/_save/hash.py`).
:::


The dispatch is a flat three-way fallback, and the listing beside it makes the recurrence over the full cell explicit ({ref}`fig:dispatch`).
**Pure** cells reference nothing outside their own body; the key reduces to the hash of the compiled cell.
**ContentAddressed** refs expose their bytes directly: Python primitives and frozen collections, values captured by marimo's intentional notebook state (UI elements and `mo.state`, hashed by current value), and buffer-protocol objects.
An important case for scientific computing is the numpy ndarray, hashed from its contiguous buffer without serialization; other objects advertising numpy's array interface normalize to the same byte view — an idiom borrowed from joblib [@joblib] and mandala [@makelov2024mandala].
**ExecutionPath** refs are not themselves directly hashable but are *cell-owned* — produced by an upstream cell with a known cache key — so we substitute the parent keys, recursively determined by the same dispatch.
A fourth outcome in the listing, **ContextExecutionPath**, covers references defined in the same cell as the cached block — the code preceding a `mo.persistent_cache` context manager — whose surrounding context is folded into the key.
The decorators `@mo.cache` / `@mo.persistent_cache` apply the same dispatch to function call arguments, resolved at call time.
The streamlit `cache_data` / `cache_resource` split [@streamlit2023caching] anticipates the data-vs-resource distinction that ContentAddressed and ExecutionPath resolve.

(sec:parent-hash)=
## Parent-hash substitution

For each cell $c$ we record a hash $H(c)$ at the end of the cell's execution.
When a downstream cell reads a ref $r$ produced by $c$ and $r$ is not directly addressable, we substitute $H(c)$ for $r$ in the downstream key — the `ExecutionPath` branch of {ref}`fig:dispatch`.
The downstream key then invalidates exactly when the producer's key does, which is exactly when the reactive scheduler would re-run the cell.
We do not require every value to be hashable in itself, only that its producing cell be hashable.
The substitution is a special case of Hughes's lazy memo function model [@hughes1985lazy], where equality is by stored location rather than by deep value.

The recurrence builds a Merkle DAG [@merkle1988protocols] over the notebook: each cell's hash commits to the hashes of its parents, so an edit to one cell invalidates exactly the subtree it dominates.
It gives the rebuilder its $O(|\text{changed subtree}|)$ rehash cost; we never re-derive a cell's hash if none of its inputs have moved.

(sec:invariances)=
## Worked example

These properties are asserted, not assumed: the notebook that builds this paper re-derives them with marimo's hasher (`BlockHasher`) on every render — invariance to comments, formatting, reordering, and cell identifiers; invalidation on public-name renames and upstream value changes; fixed keys under idempotent reruns — with one measured boundary: attribute mutation on a non-addressable object leaves every key fixed, a false positive the cache cannot detect ({ref}`sec:limitations`).

{ref}`fig:walked` visualizes the recurrence on the four-cell DAG codified below (abridged; the exact sources ship in the notebook as `compute.WALKED_SOURCES`).
A slider seeds a random input array, a small model class is constructed independently, and the forward pass binds them.
`TinyNet` stands in for any opaque value a cell can produce — a torch `nn.Module`, a database connection, a compiled kernel.
Each branch of the dispatch is exercised: `a` is Pure, `b` is ContentAddressed via the array's buffer, and `d` substitutes $H(c)$ for the unhashable model instance.
`c` itself keys cheaply as ContentAddressed — its only outside reference is the module-pinned `np` — even though its *product* cannot be hashed; substituting the producer spares every consumer.


```{marimo} python
:name: walked_example_slider
# Interactive slider — drives the seed of the live exploration
# below. In a rendered PDF this widget is inert; the multi-panel
# figure that follows shows three fixed seeds.
walked_slider = mo.ui.slider(0, 10, value=7, label="seed")
walked_slider
```

```{marimo} python
:name: walked_example_live
# Live, slider-driven hash readout, keyed by marimo's real
# BlockHasher on the compiled four-cell graph. Re-evaluates
# reactively every time `walked_slider.value` changes.
_walked_live = compute.compute_walked_state(walked_slider.value)
mo.md(
    "**Live hashes** (seed = {v}): "
    "`H(a)={a}`, `H(b)={b}`, `H(c)={c}`, `H(d)={d}`".format(
        v=walked_slider.value,
        a=_walked_live["a"]["h"], b=_walked_live["b"]["h"],
        c=_walked_live["c"]["h"], d=_walked_live["d"]["h"],
    )
)
```

```{marimo} python
:name: fig_walked_example
:eval: false
def walked_example_figure():
    # Three fixed seeds tell the story of the recurrence: initial
    # fill, cascading invalidation, idempotent rerun. Every hash and
    # branch label is produced by marimo's real BlockHasher over a
    # real compiled cell graph (compute.WALKED_SOURCES) — nothing in
    # this figure is simulated.
    seeds = (3, 7, 7)
    titles = (
        r"$t_0$: seed = 3 (initial)",
        r"$t_1$: seed = 7 (leaf changed)",
        r"$t_2$: seed = 7 (idempotent)",
    )
    states = [compute.compute_walked_state(s) for s in seeds]

    # Smoke claim: invalidation propagates a → b → d, leaves c (model)
    # alone, and an idempotent rerun changes nothing.
    assert states[0]["a"]["h"] != states[1]["a"]["h"], compute.claim(
        "seed hash changes when value moves", states=states)
    assert states[1]["b"]["h"] != states[0]["b"]["h"], compute.claim(
        "array hash cascades when seed changes", states=states)
    assert states[1]["d"]["h"] != states[0]["d"]["h"], compute.claim(
        "forward-pass hash cascades through the array", states=states)
    assert states[1]["c"]["h"] == states[0]["c"]["h"], compute.claim(
        "model hash is independent of the seed", states=states)
    assert states[2] == states[1], compute.claim(
        "idempotent rerun leaves every hash fixed", states=states)

    # Invariance battery: every robustness and boundary property
    # quoted in the prose is re-measured against the real hasher on
    # every build of this figure.
    inv = compute.hash_invariance_report()
    assert all(inv.values()), compute.claim(
        "hash invariance battery holds", **inv)

    fig = compute.plot_walked_example(states, titles)
    save_fig(fig, "fig2_walked_example")
    return fig


walked_example_figure()
```

:::{figure} figs/fig2_walked_example.svg
:label: fig:walked
:width: 78%

Worked example of the recurrence on the four-cell DAG codified above.
Every hash and branch label is emitted by marimo's hasher over a compiled cell graph at render time.
Moving the seed ($t_0 \to t_1$) invalidates `a`, `b`, and `d` (red edges); `c` stays cached because the seed is not among its refs.
Re-rendering with the same seed ($t_1 \to t_2$) leaves every hash fixed (green edges) and the rebuilder reuses every result.
The italic label under each box names the dispatch branch the cell exercises.
:::

(sec:storage)=
# Storage and Loading

Cache keys identify values, but a notebook does not always need the value itself: a downstream cell may forward a reference without inspecting it, and a static export may serialize a value for a reader who never executes the producing cell.
Storage and loading are therefore decoupled from lookup.
Of marimo's two persistent backends, the default `PickleLoader` serializes the full `Cache` envelope as one pickle blob, re-materializing every def eagerly on lookup; the opt-in `LazyLoader` writes a JSON manifest of per-def references alongside individual blob files and hydrates values on demand through a stub mechanism that crosses process and session boundaries — the same protocol whether the value comes from local disk or a sidecar inside a static HTML bundle.

(sec:store)=
## Store interface and LazyLoader

The store exposes a single `ReferenceStub` protocol with one `load()` method and codec-specific subclasses for pickle, joblib, numpy `.npy`, and Arrow; a lookup returns a stub, and deserialization happens on the stub's first access, not on lookup.
The split charges deserialization to the consumer that actually reads the value — a downstream cell may bind a stub by name without forcing it — and a single stub kind chooses its codec from the value type at save time.

(sec:wasm)=
## WASM portability

marimo exports notebooks to static HTML through a Pyodide WASM bundle.
Because the cache is content addressed and the loader is portable, cached values participate in that export: `--execute` runs the notebook once and bundles the resulting manifests and blobs, and a reader's browser re-derives the same keys and rehydrates each value on first access, exactly as in an interactive session.
Scientific articles (such as this one), general-audience blog posts, and educational materials all benefit, shipping heavy results inside the notebook itself.
Two parity constraints follow: the export runs under Pyodide's CPython release, since bytecode hashes are minor-version specific, and module versions stay out of the key (`pin_modules=False`), since host and browser environments never coincide.
Manifests are Ed25519-signed at export and verified before rehydration; a failed blob is treated as a miss, so tampered data is never silently served.

The export ships values rather than the code that produced them, so a cell may depend on packages the browser cannot import, as long as the values it defines serialize to a portable codec.
As a demonstration (`demo/` in this paper's artifact), the export host trains a PyTorch model, exports it to ONNX bytes behind a small runtime wrapper, and slices a prediction sweep into numpy arrays.
In the browser, where `import torch` is impossible, the arrays and the wrapper restore through their codecs — a custom stub rebinds the ONNX bytes to `onnxruntime-web` — and a slider drives live in-browser inference.
Values that resist serialization restore as named stubs that raise on first access, so partial restoration stays safe.
The benchmark data behind {ref}`fig:cache-eval` ships through the same export as a kilobyte-scale sidecar, because a cell's cache stores its defs — the measurement rows — while the multi-hundred-megabyte payloads they timed never leave the bench.

(sec:eval)=
# Evaluation

Caching is a conditional win: the cache pays key derivation plus value load on a hit, and key derivation plus value save on a miss.
When the goal is portability and provenance the time penalty is inconsequential; to skip recomputation, the cache must pay back its overhead by avoiding a more expensive cell body.
We validate the implementation by measuring three cost components (key derivation, value load, value save) separately and as end-to-end hit and miss paths on numpy `float64` payloads from 1 MB to 500 MB, bracketed by representative baselines: mandala's decorated-function memoization and diskcache's byte-keyed store (the no-derivation floor).
The baselines situate the overhead; they solve adjacent problems, not this one, and the contribution remains the derived key and what it unlocks.

(sec:e2e)=
## End-to-end cache evaluation

What notebook users observe is not per-call lookup latency but the wall time from "edit upstream" to "downstream value bound in Python."
Panel (a) of {ref}`fig:cache-eval` reports that composite metric across six strategies.
All cluster up to roughly 50 MB.
Above that, the spread is host-dependent: where disk dominates, strategies split along the in-memory / persistent boundary; where key derivation dominates, the persistent variants ride the same curve as the in-memory `mo.cache` bound.
The 100 ms dashed line marks the threshold below which a response reads as instantaneous [@card1991information] — worth defending, since added latency measurably suppresses exploration [@liu2014latency] — and every persistent method holds it across typical exploratory payloads.

The dotted curve in panel (a) prices the miss path.
With hit rate $p$, caching pays when the cell body costs more than $T_{hit} + \frac{1-p}{p}\,(T_{key} + T_{save})$.
Because the measured miss overhead is comparable to the hit cost on these payloads, the break-even body cost at $p = 0.9$ rides roughly 10% above the hit curve.

Panel (b) decomposes the largest-payload hit into key derivation and value load, alongside the overhead a miss adds (key derivation and value save).
marimo and mandala both content-address the value; they differ in the primitive.
mandala derives its key via `joblib.hash`, which serializes through pickle before hashing [@makelov2024mandala]; marimo's `data_to_buffer` views the ndarray's contiguous bytes through Python's buffer protocol and hashes them directly.
How much the pickle pass costs is strongly host-dependent: on the Apple M2 Pro used for the camera-ready figures, hashing is fast and the pickle pass costs mandala roughly 3× end to end; on a Linux x86-64 server, memory bandwidth makes the pickle pass nearly free, value load dominates instead, and the penalty compresses to roughly 1.2×.
The panel-(a) ordering — marimo at or below mandala at every payload — held on all three hosts measured (Apple M2 Pro, Linux server, Linux workstation); the figure carries its build host and ratio so any reproduction shows its own number.
marimo's value load tracks the `diskcache (fixed key)` floor, isolating the structural gap in key derivation rather than storage.
Above the primitive, the differentiator is ergonomic: mandala caches at decorated-function boundaries inside `with storage:` blocks, while marimo caches at cell boundaries derived from the reactive DAG.
The byte-keyed alternatives bracket the comparison: a pre-built byte key is the fastest possible lookup but cannot detect input changes, and `diskcache.memoize` pays pickle on both the key and its SQLite blob store.
Panel (c) exposes per-call variance: irregular fsyncs and page-cache turnover surface as a long tail, longest for the lazy variant's background-writer settle.

The key path generalizes beyond contiguous ndarrays: containers of arrays and Arrow tables normalize to byte views at comparable cost; a large homogeneous Python list pays an element-wise walk roughly an order of magnitude slower — numeric data belongs in arrays before it crosses a cached boundary.
Framework-native formats slot into the same split: torch tensors already key through the buffer protocol, and a `.pt` codec in the lazy registry round-trips them 1.4–1.7× faster than the joblib pairing at 50 and 500 MB (`lib/bench_torch.py` in this paper's artifact).

The sweep behind {ref}`fig:cache-eval` is itself served by the mechanism under test: `@mo.persistent_cache` wraps every measurement function, keyed additionally on a host fingerprint and a bench-version string; the sweep cost 91 s cold on the Linux server build, re-binds in under a millisecond warm, and editing any measurement function invalidates exactly the affected sweeps.

```{marimo} python
:name: fig_cache_eval
:eval: false
def _e2e_setup():
    """Wire every cache backend once and return ``(methods, cleanup)``.

    ``methods`` is ``{name: bind}``. ``bind(payload)`` does per-payload
    preparation (warm a cache, set a byte-key, prime a memoize wrapper)
    and returns the 0-arg callable the sweep will time. Setup does NO
    measurement — that's owned by ``compute.sweep_methods_samples`` so
    all timing lives in one place.

    ``cleanup()`` runs once after the sweep finishes. We use it to
    `__exit__` the mandala storage so its SQLite connection closes and
    to `rmtree` every temporary cache directory (the persistent caches
    can grow to several GB across a sweep).

    Has to live in the cell (not in ``compute.py``) so ``mo.cache`` /
    ``mo.persistent_cache`` capture the marimo runtime context.
    """
    import shutil
    pickle_dir = tempfile.mkdtemp()
    lazy_dir   = tempfile.mkdtemp()
    md_dir     = tempfile.mkdtemp()

    mo_cache_fn = mo.cache(compute.identity)
    mo_pickle_fn = mo.persistent_cache(
        save_path=pickle_dir, method="pickle",
    )(compute.identity)
    mo_lazy_fn = mo.persistent_cache(
        save_path=lazy_dir, method="lazy",
    )(compute.identity)

    md_storage = mandala.imports.Storage(
        db_path=f"{md_dir}/db.sqlite",
        overflow_dir=md_dir,
        overflow_threshold_MB=0,
    )
    md_identity = mandala.imports.op(compute.identity)
    # Activate the storage for the bench's lifetime. mandala's
    # context manager calls `commit()` (heavy fsync) on `__exit__`,
    # so per-call `with md_storage:` would dominate the timing.
    # We enter here and let `cleanup()` exit at sweep end.
    md_storage.__enter__()

    def bind_mandala(payload):
        md_identity(payload)  # warm
        def hit():
            return md_storage.unwrap(md_identity(payload))
        return hit

    def cleanup():
        # Close the SQLite handle first, then wipe every persistent
        # cache directory we allocated. Each holds the per-payload
        # ndarrays from the sweep — left behind, they accumulate
        # across reruns and exhaust /tmp.
        try:
            md_storage.__exit__(None, None, None)
        finally:
            for d in (pickle_dir, lazy_dir, md_dir):
                shutil.rmtree(d, ignore_errors=True)

    methods = {
        "mo.cache":                   compute.bind_warm(mo_cache_fn),
        "mo.persistent_cache":        compute.bind_warm(mo_pickle_fn),
        "mo.persistent_cache (lazy)": compute.bind_warm(mo_lazy_fn),
        "mandala":                    bind_mandala,
        "diskcache.memoize":          compute.bind_diskcache_memoize,
        "diskcache (fixed key)":      compute.bind_diskcache_fixed,
    }
    return methods, cleanup


# Bench-key components folded into every cached measurement: a host
# fingerprint (system + arch + Python version) and a methodology
# version string. Bumping `_BENCH_VERSION` invalidates every persisted
# sweep on every host, which is the right move whenever `_e2e_setup`,
# `compute.measure_hit`, `compute.measure_miss`, codec choices, or
# payload generation change. Without this, `@mo.persistent_cache`
# would key only on `sizes_mb` and silently reuse cross-host or
# pre-edit measurements.
_HOST_FINGERPRINT = (
    f"{platform.system()}|{platform.machine()}|"
    f"{platform.python_implementation()}|{platform.python_version()}"
)
_BENCH_VERSION = "2026-06-miss-overhead-and-cold-timing"


@mo.persistent_cache
def _e2e_sweep(sizes_mb, host=_HOST_FINGERPRINT, version=_BENCH_VERSION):
    # Times its own cold cost, so the figure can quote what the cache
    # saves on reopening this very notebook (the paper's own sweep is
    # the realistic workload the cache was built for).
    del host, version  # consumed by the cache key only
    import time
    _t0 = time.perf_counter()
    rows = compute.sweep_methods_samples(sizes_mb, _e2e_setup)
    return {"rows": rows, "cold_s": time.perf_counter() - _t0}


@mo.persistent_cache
def _stage_breakdown(size_mb, host=_HOST_FINGERPRINT, version=_BENCH_VERSION):
    # One-size primitive measurement so we can decompose cache-hit time
    # into key derivation and value load at the figure's largest
    # payload. The e2e sweep only times totals.
    del host, version
    return compute.measure_hit(compute.random_payload(size_mb))


@mo.persistent_cache
def _miss_breakdown(size_mb, host=_HOST_FINGERPRINT, version=_BENCH_VERSION):
    # Write-path mirror of `_stage_breakdown`: the key + save overhead
    # a miss adds on top of the cell body it had to run anyway.
    del host, version
    return compute.measure_miss(compute.random_payload(size_mb), runs=5)


@mo.persistent_cache
def _miss_sweep(sizes_mb, host=_HOST_FINGERPRINT, version=_BENCH_VERSION):
    # Total miss overhead (key + save) per size, for the break-even
    # curve in panel (a).
    del host, version
    return compute.sweep_write_overhead(sizes_mb)


def cache_eval():
    import time

    # Capped at 500 MB so the in-memory mo.cache series doesn't retain
    # the cumulative payloads across the sweep — past ~500 MB it pushes
    # commodity laptops into swap. The interactive threshold crossing is
    # already visible by this point.
    sizes_mb = (1, 10, 50, 100, 200, 500)
    _t0 = time.perf_counter()
    _sweep = _e2e_sweep(sizes_mb)
    _first_ms = (time.perf_counter() - _t0) * 1000
    rows, sweep_cold_s = _sweep["rows"], _sweep["cold_s"]
    # Self-demonstration: this notebook is its own workload. In a warm
    # session the first call above is a real cache hit and its timing
    # is the honest per-session rebind cost; in the cold session that
    # just measured the sweep, fall back to the median of further hits.
    if _first_ms < 0.5 * sweep_cold_s * 1000:
        sweep_warm_ms = _first_ms
    else:
        _warm = []
        for _ in range(3):
            _t0 = time.perf_counter()
            _e2e_sweep(sizes_mb)
            _warm.append((time.perf_counter() - _t0) * 1000)
        sweep_warm_ms = float(np.median(_warm))

    median_rows = compute.medians_from_samples_rows(rows)
    sizes_arr, e2e = compute.sweep_pivot(median_rows)
    threshold_ms = 100.0

    # Surface payload-size ceilings (largest size at which each
    # persistent method still beats the interactive threshold) for
    # prose to quote.
    def below_threshold_ceiling_mb(series):
        below = sizes_arr[series < threshold_ms]
        return float(below.max()) if below.size else float("nan")

    mp = e2e["mo.persistent_cache"]
    mp_lazy = e2e["mo.persistent_cache (lazy)"]
    interactive_ceiling_mb = {
        "mo.persistent_cache":        below_threshold_ceiling_mb(mp),
        "mo.persistent_cache (lazy)": below_threshold_ceiling_mb(mp_lazy),
    }
    assert any(c == c for c in interactive_ceiling_mb.values()), compute.claim(
        f"some mo.persistent_cache variant beats {threshold_ms:g} ms at some size",
        sizes_mb=sizes_arr,
        mo_persistent_cache_ms=mp,
        mo_persistent_cache_lazy_ms=mp_lazy,
    )

    # Abstract claim, asserted structurally: marimo at or below mandala
    # within measurement noise at every size where both survive (sub-ms
    # ties at small payloads flip under load — observed once on a loaded
    # build), and strictly faster at the largest payload, where the
    # pickle-pass gap is the claim.
    ma = e2e["mandala"]
    _both = ~(np.isnan(ma) | np.isnan(mp))
    assert (mp[_both] <= 1.10 * ma[_both]).all(), compute.claim(
        "marimo cache hit within noise of mandala at every measured size",
        sizes_mb=sizes_arr, mandala_ms=ma, mo_persistent_cache_ms=mp,
    )
    assert mp[_both][-1] < ma[_both][-1], compute.claim(
        "marimo cache hit strictly beats mandala at the largest payload",
        sizes_mb=sizes_arr, mandala_ms=ma, mo_persistent_cache_ms=mp,
    )

    # Miss overhead and the break-even body cost. Caching pays at hit
    # rate p when body > hit + (1-p)/p * (key + save); we plot p = 0.9.
    _wrows = _miss_sweep(sizes_mb)
    _, miss_total = compute.sweep_pivot(
        [{"size_mb": r["size_mb"], "method": r["method"], "ms": r["ms"]}
         for r in _wrows])
    p_hit = 0.9
    breakeven_ms = mp + (1 - p_hit) / p_hit * miss_total["mo.persistent_cache"]

    # Stage decomposition at the largest e2e payload — hit (key + load)
    # and miss (key + save) measured on the real disk-backed paths.
    box_size_mb = float(max(sizes_mb))
    stage_rows = [{**r, "size_mb": box_size_mb} for r in _stage_breakdown(box_size_mb)]
    stage = compute.stage_decomposition_at(stage_rows, box_size_mb)
    miss_rows = [{**r, "size_mb": box_size_mb} for r in _miss_breakdown(box_size_mb)]
    miss_stage = compute.stage_decomposition_at(miss_rows, box_size_mb)

    # Structural claim: mandala's end-to-end hit is slower than
    # mo.persistent_cache at every payload (asserted above). The
    # magnitude is strongly host-dependent — sha256 vs pickle
    # throughput — so the label carries this build's e2e ratio at
    # the largest size where both survive.
    if _both.any():
        mandala_slowdown_x = round(
            float(ma[_both][-1] / mp[_both][-1]), 1)
    else:
        mandala_slowdown_x = float("nan")

    # Observed (not asserted): on most builds diskcache.memoize hits its
    # SQLite blob ceiling somewhere in the sweep.
    memoize_ms = e2e.get("diskcache.memoize", np.full(len(sizes_arr), np.nan))
    fail_mask = np.isnan(memoize_ms)
    memoize_fail_threshold_mb = (
        float(sizes_arr[fail_mask].min()) if fail_mask.any() else float("nan")
    )

    host_label = (
        f"host: {platform.system()} {platform.machine()} "
        f"({platform.python_implementation()} {platform.python_version()}); "
        f"mandala/mo ratio: {mandala_slowdown_x}x; "
        f"this figure's sweep: {sweep_cold_s:.0f} s cold, "
        f"{sweep_warm_ms:.1f} ms cached"
    )

    samples_at_box = compute.samples_at_size(rows, box_size_mb)
    fig = plt.figure(figsize=(8.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0],
                          hspace=0.55, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    compute.plot_e2e(ax_a, e2e, sizes_arr, threshold_ms=threshold_ms,
                     host_label=host_label)
    _fin = ~np.isnan(breakeven_ms)
    ax_a.loglog(sizes_arr[_fin], breakeven_ms[_fin], ls=":", lw=1.6,
                color="#1f77b4", alpha=0.9,
                label="break-even body cost (90% hit rate)")
    ax_a.legend(loc="upper left", fontsize=6.5)
    compute.plot_stage_decomposition(
        ax_b, stage, miss_ms=miss_stage,
        title=f"(b) Hit vs miss decomposition at {box_size_mb:.0f} MB",
    )
    compute.plot_e2e_box(
        ax_c, samples_at_box,
        title=f"(c) Per-call distribution at {box_size_mb:.0f} MB",
    )
    save_fig(fig, "fig3_cache_eval")
    return fig


cache_eval()
```

:::{figure} figs/fig3_cache_eval.svg
:label: fig:cache-eval
:width: 90%

End-to-end cache evaluation on numpy `float64` payloads.
(a) Cache-hit latency vs payload size, log-log; the 100 ms dashed line marks the interactive threshold [@card1991information], the dotted curve the break-even body cost at a 90% hit rate.
(b) Hit (key derivation + value load) and miss (key derivation + value save) decomposition at the largest sweep size, on the real disk-backed paths.
(c) Per-method distribution of cache-hit samples at the largest size; `diskcache.memoize` fails outright past its SQLite blob ceiling, and failed sizes drop out of the plot.
The host label carries the build host, the mandala/marimo ratio, and the cold-vs-cached cost of the figure's own sweep.
:::

The camera-ready figures are built on a MacBook Pro with an Apple M2 Pro; the full sweep also ran on the two Linux hosts as a consistency check, not a portability study, with the decomposition shifting between key-derivation-dominant and load-dominant as described above.
The stage decomposition times the real disk-backed paths on both sides — `load_cache` for marimo, `joblib.load` on a temp-file blob for mandala — including the page-cache and envelope costs a primitive `pickle.loads` skips.
All marimo timings carry the fix for a hash-memo bug (marimo PR #8805) that otherwise leaves decorator-path hits unrealistically flat.

(sec:limitations)=
# Limitations and Discussion

We lead with limitations.
The cache does not round-trip cells whose body closes over a non-serializable handle such as an open socket, CUDA device, or captured callable; rather than corrupt downstream state, the system bails to recomputation.
Hash-memo flushing is coarse: a lifecycle event on a defining cell clears the whole memo, costing a redundant rehash on next access.
Renaming is the false-negative class: public renames invalidate consumers by design, and renaming a `@mo.persistent_cache` function orphans its on-disk namespace.
Library versions do not enter the key unless `pin_modules=True`, so an environment upgrade can serve stale hits; the portable WASM export must accept this gap, since pinning would bind keys to the export host.
Mutable refs that bypass the DAG — alias mutation through a closure, attribute writes on a non-addressable object — can still poison downstream cells; the invariance battery measures exactly this undetectable false positive (Rex [@zheng2025reactive] probes the same boundary).
The lazy codec registry is young: pandas and polars frames take the Arrow path, while a raw `pyarrow.Table` still falls back to pickle.
Signed manifests bind blobs to the exporting author, but signing attests provenance, not correctness: the reader still trusts that the author cached what the code computes.

Side effects — file reads, network calls, wall-clock queries — fold back into the key through hashable side-effect handles whose results contribute to the cell-level hash as if they were external references.
Two constructors are currently provided, `mo.watch.file` and `mo.watch.directory`, keyed on content and listing; randomness or wall-clock time could bind to similar lifetime-managed handles.
Outside that surface, side effects go unseen by the key.

(sec:future-work)=
# Future work

Three lines follow from the measured boundaries.
*Cost-aware policy*: with the write path measured, the break-even rule can run online — refuse to cache bodies cheaper than their own overhead, and track realized savings per cell.
*Key scope*: pinning module versions by default would close the stale-hit gap.
*Richer codecs*: landing the measured `.pt` tensor codec and a `pyarrow.Table` Arrow path, plus persistent cell-result reuse for agentic workflows driving long-running sessions [@manz2026beyond].
Storage policy (eviction, per-codec footprint) and provenance-aligned cross-session memoization [@pimentel2017noworkflow] remain open.

(sec:conclusion)=
# Conclusion

Given marimo's reactive DAG, content-addressed caching of cell results is a systems consequence rather than a novelty: the same parse that schedules a cell yields its cache key.
Every render of this paper re-derives the invariance battery and the worked example with marimo's hasher, and the benchmarks are served by the cache they evaluate.
On the payloads measured, hits track the byte-keyed floor within a small factor, and break-even sits roughly 10% above hit latency at a 90% hit rate.
Cached artifacts ride the WASM export to readers with only a browser, a torch-trained model restoring as a live ONNX session in a page that cannot import torch.
The keys that schedule a notebook thus carry its results across sessions, processes, and machines.

% --- Bibliography is rendered by mystmd from references.bib ---

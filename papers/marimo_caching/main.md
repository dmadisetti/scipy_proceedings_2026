---
title: 'Hash all the things: Caching for fast notebook restarts'
short_title: marimo Caching
marimo-version: 0.23.16
pyproject: |-
  requires-python = ">=3.12"
  # NB. Keep in sync with `pyproject.toml` in this directory; marimo's
  # `--sandbox` reads from here, the standalone file serves `uv sync`
  # and IDE tooling.
  dependencies = [
      # Pin to marimo 0.23.16 because the paper uses automatic
      # local-module packaging for the WASM export
      # (so the paper's own `lib` ships to the browser with no hand-hosted
      # wheel), the `cache_cells` runtime option, and the signed
      # LazyLoader. The `lib.compute` / `save_fig` helpers are imported
      # from this directory
      # for native runs (`make figures`, `make pdf`).
      "marimo==0.23.16",
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
```{raw:typst}
// Hide marimo cell source blocks from the PDF render.
#show raw.where(block: true, lang: "python"): _ => none
```

```python {.marimo hide_code="true" name="setup"}
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
import platform
import tempfile
import diskcache
import mandala.imports  # noqa: F401 — submodule import; cells use `mandala.imports.{Storage, op}`

# Bench primitives, plot helpers, and `compute.claim(...)` used by the
# figure cells.
from lib import compute, save_fig
```

+++ {"part": "abstract"}

We describe a caching mechanism that lets reactive notebooks restart without re-running their expensive cells.
The mechanism is built into marimo, a reactive Python notebook that models the notebook as a dataflow graph.
Each cell's cached result is identified by a key built from fingerprints (hashes) of the cell's code and inputs.
Input values whose bytes are accessible are hashed directly, while other inputs are represented by the key of the cell that produced them, computed the same way.
Because each cell's key folds in the keys of the cells it depends on, editing one cell invalidates the cached results of exactly the cells downstream of it.
Cached values are stored on disk and loaded only when accessed, so they can be reused across independent runs of the same notebook.
These values are also bundled into marimo's static export, a standalone web page written in HyperText Markup Language (HTML) that runs the notebook through WebAssembly (WASM), so readers whose only Python runtime is a browser can open a notebook with its expensive results and trained models already in place.
In microbenchmarks over payloads of varying size, a marimo cache hit is comparable to widely used Python caching libraries, with a speedup on certain hardware, while needing little user setup.

+++

(sec:intro)=
# Introduction

Notebooks underpin much of scientific Python, but most notebooks cannot be re-executed from scratch.
In a survey of 1.4 million public Jupyter notebooks, only about a quarter re-executed top to bottom without raising an error [@pimentel2019largescale].
Traditional notebooks like Jupyter, built on the imperative `ipykernel`, are read-evaluate-print loops (REPLs) in which running each block of source code, or *cell*, mutates shared global state.
As a result, these notebooks accumulate hidden state, and the outputs saved in the notebook file can differ from those of a fresh top-to-bottom run.

Reactive notebooks close this gap by treating each cell as a node in a dataflow graph.
The notebook determines, for each cell, which variables it reads (its *references*, or `refs`) and which variables it defines (its *definitions*, or `defs`).
From these, a reactive notebook derives a deterministic execution order based on data dependencies rather than on the order of cells on the page.
Running a cell removes the cell's previous variable bindings from memory, updates the dataflow graph if the cell's code changed, and re-runs the cells that depend on it, which minimizes hidden state.
Notable reactive notebooks include Pluto.jl [@vanderplas2020pluto], Observable [@bostock2017observable], and Livebook [@valim2020livebook].
Reactive notebooks descend from a longer tradition of direct-manipulation programming environments [@victor2012inventing], in which editing the source is itself the act that updates the running program.
marimo [@marimo] is a reactive notebook for Python, and its caching mechanism is the subject of this paper.

To mitigate unnecessary re-runs, reactive notebooks offer runtime configuration, such as lazy executors that mark cells stale instead of running them; however, these primitives require user intervention.
We exploit their deterministic execution order to build a caching mechanism that automatically skips recomputation of an expensive cell whose code and inputs are unchanged.

Caching for notebooks and Python is not new; we review prior systems in {ref}`sec:background`.
Each prior method asks the user to opt in at a boundary, such as a decorated function, document chunk, or session.
In contrast, reactive notebooks already draw a boundary around every cell, so the author never has to choose where caching applies.

The caching mechanism we propose, and whose implementation we share, was designed to satisfy three properties.

* Skip expensive recomputation when a cell's references and source are unchanged.
* Preserve reactive determinism by reusing a result only while it stays valid and never serving a stale (false-positive) hit.
* Make cached artifacts transportable through marimo's static WASM/HTML export.

Out of scope are full session restoration in the sense of Kishu [@li2025kishu], distributed execution, and reproducibility of arbitrary Python notebooks [@pimentel2019largescale].
Our contribution is deterministic reuse between potentially cross-platform notebook sessions that follow marimo's reactive principles.

(sec:background)=
# Background and Related Work

A memoized function, or *memo* function [@michie1968memo], stores the result of a call and returns that result when it receives the same inputs again.
To find the stored result, the function associates each set of inputs with a *cache key* as an identifier used to index the cache.
marimo constructs this key with a *hash*, a short, fixed-size fingerprint derived from data.
Matching hashes can stand in for matching data because cryptographic hash functions make it improbable that two different inputs have the same fingerprint.
When a system identifies a value by hashing the value's own bytes, the value is said to be *content-addressed*.
To reuse a cached cell result, marimo must derive the same key whenever its inputs are unchanged.

A cell's inputs are the variables it reads, which may be defined in other cells or in the global environment.
In a reactive notebook like marimo, these variables are statically known prior to execution, as the cell's "refs" derive a dependency graph from source.
At run time, marimo additionally knows the values currently bound in memory.
It can therefore derive a cell's key from the graph, the cell's code, and the current reference values, rather than from the history of which cells happened to run.
Cell-level dataflow tracking has an earlier antecedent in [@koop2017dataflow].

IPyflow and nbsafety [@macke2021nbsafety;@ipyflow] take a different route to reactivity.
They retrofit reactivity onto Jupyter by tracing execution to record which variables each cell reads and writes, then flag or re-run cells whose inputs have become stale.
Because the resulting graph contains only dependencies observed during execution, different execution histories can produce different graphs for the same notebook.
marimo instead derives the graph from source, so the same source produces the same dependencies before any cell runs.

The dependency graph identifies what contributes to a cell, but the cache must still reduce that information to a stable key.
For this step, marimo borrows from build systems.
Build Systems à la Carte [@mokhov2018build] separates a *scheduler*, which decides the order in which tasks run, from a *rebuilder*, which decides whether each task must run again.
marimo's reactive dataflow engine supplies the scheduler, while its cache acts as the rebuilder by comparing keys computed from a cell's code and inputs ({ref}`sec:keys`).

The Nix package manager provides the model for extending a key through the dependency graph [@dolstra2004nix;@dolstra2006purely].
Nix identifies a package with a hash of its inputs, including the hashes of the packages from which it was built.
This *recursive hash* makes the package's identity cover its dependency tree.
marimo applies the same construction to a reactive notebook: a cell's key incorporates the keys of upstream cells when their values cannot be content-addressed directly.
Data-engineering systems use related constructions at coarser granularity.
Bauplan and Nessie hash pipeline stages [@tagliabue2024bauplan], while the workflow engines Nextflow (`-resume`) and Snakemake (`--cache`) key task results on code, parameters, and input hashes [@ditommaso2017nextflow;@molder2021snakemake].

Existing caching systems for Python and notebooks differ in both their unit of reuse and the information they require from the author.
IncPy modifies CPython, the reference implementation of Python, to memoize function calls automatically [@guo2011incpy].
knitr caches chunks of literate documents whose dependencies the author declares by hand [@xie2015knitr].
jupyter-cache re-executes a notebook as a whole when any code cell changes [@jupytercache].
Streamlit asks the author to choose between `cache_data`, for values identified by content, and `cache_resource`, for values identified by what produced them [@streamlit2023caching].
marimo instead uses its key construction ({ref}`sec:dispatch`) to make that choice for each reference, while its reactive graph supplies a reuse boundary around every cell.

There are a few existing scientific-Python memoizers (for comparison to this work see {ref}`sec:eval`).
The most similar, mandala [@makelov2024mandala], provides the end-to-end comparison by memoizing execution inside a `with storage:` context, computing content addresses with `joblib.hash`, and recording which calls produced which values.
Alternatively, diskcache [@diskcache] provides the storage control by storing values under byte keys supplied by the caller, allowing the evaluation to hold key construction constant while measuring storage cost.
Other notable work includes Kishu [@li2025kishu] and ElasticNotebook [@li2024elasticnotebook], which checkpoint and migrate notebook state, but are not directly comparable to marimo's mechanism.

(sec:keys)=
# Cache Keys

In computational caching, a false positive — restoring a value that the code would not have produced — is unacceptable.
A false negative — failing to find a stored value and recomputing it — merely wastes time and is acceptable because the user can understand the caching criteria.
The key derivation must therefore change whenever a value the cell reads changes, and it should not change under superficial edits such as reformatting or comment changes.

Consider two obvious ways to derive a key.
The first is to hash the value of every reference the cell reads.
A marimo notebook could attempt this, since at runtime it exposes both the dataflow graph and the reference values bound in memory.
This derivation fails because some Python values expose nothing stable to hash.
An object's memory address is neither stable nor meaningful as an identity;
weak references report which object they point to, not what it contains;
opaque C-extension objects expose neither their underlying bytes nor a stable text representation.
The second derivation is to hash the cell's source bytes alone.
This fails in the opposite way: the key is computable but does not change with the cell's inputs, producing the false positives ruled out above.
It misses inputs that arrive through side effects such as the filesystem, the network, or the wall clock.

marimo combines the two derivations: a value that can be hashed is hashed, and a value that cannot is identified by the code that produced it.
Each derivation covers the other's failure: the first keeps the key sensitive to inputs and the second keeps it computable.
Build systems handle unhashable artifacts the same way: an artifact that exposes no content to hash is identified by the build step that produced it [@dolstra2004nix].
marimo applies this idea recursively over the dataflow graph.
{ref}`sec:dispatch` gives the precise construction.
{ref}`sec:invalidation` then examines invalidation: editing a cell invalidates only downstream cached results, and recomputing keys takes time proportional to affected cells.
Side effects, which neither derivation sees, remain only partially covered; we return to this limitation in {ref}`sec:limitations`.

(sec:dispatch)=
## Constructing the key

Here we describe a cell's cache key, computed from its source code and references. Throughout, $H(c)$ denotes the key of cell $c$.

### Hashing source code

A cell's references identify what flows into it, and its code determines what it computes from those inputs.
The code must therefore enter the key.
Without it, two cells that read the same input, such as `y = x + 1` and `y = x - 1`, would share a key, and one could be served the other's cached result.
Hashing code provides basic invalidation: editing a cell changes its key, so its stale result is never reused.

marimo hashes the compiled bytecode rather than the source text.
Compilation discards edits that cannot change behavior, such as comments and formatting, so they do not invalidate the cache.
The trade-off is that bytecode is specific to the Python version.
Cached results therefore do not transfer across interpreter upgrades, and marimo's browser export ({ref}`sec:wasm`) requires the exporting interpreter to match the browser's Python version.

### Hashing references

:::{figure} figs/fig1_dispatch.png
:label: fig:dispatch
:width: 100%

Hashing references.
Left: a cell with no references contributes no reference hashes.
For a cell with references, each value is content-addressed when possible; otherwise, the key of the cell that produced it is substituted.
The reference hashes and the hash of the cell's compiled body are then combined into a single hash: the cell's key, $H(c)$.
Right: the same derivation applied to a whole cell, classified as `Pure` when it has no references, `ContentAddressed` when every reference is hashed by content, and `ExecutionPath` when any reference requires producer substitution.
The listing also adds `ContextExecutionPath`, a special case described in the text.
:::

```python {.marimo hide_code="true" name="fig_dispatch_display"}
mo.image("figs/fig1_dispatch.png", width="100%")
```


The construction first checks whether the cell has references.
A cell with references tries to content-address each value and uses producer substitution only where content addressing fails; we call this decision the *key dispatch*.
The cases are described below and illustrated in {ref}`fig:dispatch`.

**No references.** A cell that references nothing outside its own body needs no reference hashes, so its key contains only the hash of its compiled code.
The figure labels the whole-cell key `Pure`.

**Content-addressed values.** A value that exposes its underlying bytes through Python's buffer protocol — the standard mechanism for exposing an object's raw bytes without copying — has those bytes hashed.
Immutable values, such as numbers, strings, and frozen collections, are also hashed directly.
Interactive inputs reach this case through normalization: before hashing, a reference to a user interface (UI) element such as a slider is replaced by its current value, so the key changes when the reader moves the slider.
Buffer hashing covers NumPy ndarrays and other objects advertising NumPy's array interface.
The hash is computed from the contiguous buffer without serialization, an approach borrowed from joblib [@joblib] and mandala [@makelov2024mandala].
The figure labels a whole-cell key `ContentAddressed` when every reference can be hashed by content.

**Producer substitution.** Any other value exposes nothing stable to hash, but a known upstream cell produced it.
Instead of hashing the value, marimo uses the producing cell's key as the reference's hash.
The producing cell's key uses this construction, covering its code, inputs, and everything upstream.
A change anywhere in the value's ancestry therefore changes the reference's hash.
The figure labels a whole-cell key `ExecutionPath` when any reference requires this fallback.

### Combining hashes

All hashing in this construction uses SHA-256, as {ref}`fig:dispatch` shows.
The hashes of the code and of the references are combined into one in the combine step at the bottom of the figure.
Every reference hash, in sorted reference-name order, is fed with the bytecode hash into one SHA-256 computation, so processing order cannot affect the result.
The resulting digest is the cell's key, $H(c)$.
Because SHA-256 is cryptographic ({ref}`sec:background`), keys match only when the code and every reference contribution match, so key equality safely stands in for "this cell would compute the same values."

### Caching blocks and functions

The cached unit is not always a whole cell: it can be a block of code inside a cell, or a function ({ref}`sec:api`).
A reference defined earlier in the same cell as a cached block has no parent cell whose key could stand in for it.
Instead, the code surrounding the block is folded into the key, a special case of producer substitution that the implementation calls `ContextExecutionPath`.
For a cached function, the same dispatch classifies the function's arguments at call time.

(sec:invalidation)=
## Invalidation

The construction of {ref}`sec:dispatch` is recursive: a cell's key can contain its producer's key in turn. Yet no key is derived by walking the whole ancestry.
When a cell $c$ finishes, marimo records $H(c)$; when a downstream key needs it for producer substitution, marimo reuses the recorded value.
Each cell's key is therefore computed at most once per execution.

A cached result is *invalidated* when its cell's key changes: the new key matches no stored entry, so the next lookup misses and the cell recomputes.
Invalidation is not an action marimo performs, and nothing is deleted.
The old entry simply stops being found.

{ref}`sec:keys` opened by ruling out false positives: the cache must never serve a value the code would not have produced.
For invalidation, that means never missing a change: whenever re-running a cell could produce a different value, the cell's key must have changed.
To keep false negatives rare, invalidation should be limited: an edit should not invalidate results it cannot affect, and updating keys should be cheap.
The two subsections below establish each property in turn.

### Invalidation never misses a change

For a reference hashed by content, the property is immediate: changing bytes changes its hash and every key built from it.
The case that needs an argument is producer substitution, which judges a value by its origin rather than its content.
Could the producing cell's key stay the same while the value it produced changes?
In a reactive notebook, no.
A value changes only if its producing cell re-runs, and a cell re-runs only when its code or inputs change — its key's ingredients.
An unchanged producer key therefore implies an unchanged value if cell bodies are deterministic; side effects are the exception ({ref}`sec:limitations`).
This establishes reactive determinism ({ref}`sec:intro`): a cached value is what re-running the cell would produce.
This argument is what requires a reactive notebook.

Caching never requires the value itself to be hashable, only that the producing cell have a key.
The substitution is a special case of Hughes's lazy memo functions [@hughes1985lazy]: two values are treated as equal because they come from the same execution of the same code, not because their contents were compared.

### Invalidation is limited and cheap

With producer substitution, a cell's key contains upstream cell keys.
The notebook's keys therefore form a Merkle directed acyclic graph (DAG) [@merkle1988protocols], a structure in which each node's fingerprint depends on the fingerprints of the nodes it builds on.
Git commits use the same construction.
Two properties follow.
First, editing a cell changes keys only downstream, so every other cached result remains valid.
Second, where a re-run cell produces byte-identical values, content-addressed consumers keep their old keys and propagation stops.
Recomputing keys after an edit takes time proportional to changed keys because every other recorded key is reused.

### A worked example

{ref}`fig:walked` traces the construction on the four-cell PyTorch graph below: a seed, a random tensor generated from it, an independently constructed small neural network, and a forward pass that applies the network to the tensor.
The cells exercise all three strategies: the seed is hashed as an immutable value, the tensor through its buffer, and the network (`TinyNet`), which exposes no bytes, by its producing cell's key.
The italic label under each cell in the figure classifies that cell's own key, as computed by marimo's hasher: `a` is `Pure` (no references), `b` and `c` are `ContentAddressed` (every reference hashed by value), and `d` is `ExecutionPath` (one reference substituted a producer's key).

```{raw:typst}
#text(size: 8.5pt)[#raw(lang: "pseudo", block: true,
"# a · seed        # b · random input            # c · model               # d · forward pass
seed = 7           x = rng(seed).normal(64)      model = TinyNet()         y = model(x)")]
```

```python {.marimo hide_code="true" name="walked_example_slider"}
# This slider sets the seed used by the hash readout in the next
# cell. In the rendered PDF the slider does nothing; the figure
# below shows three fixed seeds instead.
walked_slider = mo.ui.slider(0, 10, value=7, label="seed")
walked_slider
```

```python {.marimo hide_code="true" name="walked_example_live"}
# Compute H(a) through H(d) for the current slider value, using
# marimo's hasher on the compiled four-cell graph. When a reader
# moves the slider, this cell re-runs and the hashes update.
_walked_live = compute.compute_walked_state(walked_slider.value)
mo.md(
    "**Hashes** (seed = {v}): "
    "`H(a)={a}`, `H(b)={b}`, `H(c)={c}`, `H(d)={d}`".format(
        v=walked_slider.value,
        a=_walked_live["a"]["h"], b=_walked_live["b"]["h"],
        c=_walked_live["c"]["h"], d=_walked_live["d"]["h"],
    )
)
```

```python {.marimo hide_code="true" name="fig_walked_example"}
def walked_example_figure():
    # Compute the hashes at three seeds: 3, then 7, then 7 again.
    # The three panels show an initial state, the invalidation caused
    # by changing the seed, and a rerun in which nothing changed.
    # Every hash and branch label comes from marimo's hasher on a
    # compiled cell graph; nothing in this figure is simulated.
    seeds = (3, 7, 7)
    titles = (
        r"$t_0$: seed = 3 (initial)",
        r"$t_1$: seed = 7 (leaf changed)",
        r"$t_2$: seed = 7 (idempotent)",
    )
    states = [compute.compute_walked_state(s) for s in seeds]

    # Check that the figure shows what the prose claims: changing
    # the seed invalidates a, b, and d; c is unaffected; and a rerun
    # with the same seed changes nothing.
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

    # Re-check every robustness property quoted in the prose against
    # the hasher, on every build of this figure.
    inv = compute.hash_invariance_report()
    assert all(inv.values()), compute.claim(
        "hash invariance battery holds", **inv)

    fig = compute.plot_walked_example(states, titles)
    save_fig(fig, "fig2_walked_example")
    return fig
```

:::{figure} figs/fig2_walked_example.png
:label: fig:walked
:width: 100%

A worked example of the recurrence on the four-cell graph written out above.
Every hash and branch label in the figure is produced by marimo's real hasher on a compiled cell graph at render time.
Changing the seed ($t_0 \to t_1$) invalidates `a`, `b`, and `d` (red edges); `c` stays cached because the seed is not among its references.
Re-rendering with the same seed ($t_1 \to t_2$) leaves every hash unchanged (green edges), so the rebuilder reuses every result.
The italic label under each box names the dispatch branch that cell exercises.
:::

(sec:storage)=
# Storage and Loading

On a cache hit, marimo restores the variables the cached cell would have defined.
Restoring has two parts: *lookup* finds the stored entry matching $H(c)$, and *loading* deserializes its values into memory.
They need not happen together because the notebook does not always need a value's bytes.
For example, a downstream cell may pass a variable to a third cell without inspecting it.
marimo therefore lets loading lag lookup, and offers two loaders that differ in how far.

For `mo.persistent_cache`, the default loader, `PickleLoader`, does not lag at all.
It writes the full `Cache` envelope, the record holding every variable the cell defined, as a single blob using pickle, Python's built-in serializer.
On lookup, it loads every variable back immediately.

A second loader, `LazyLoader`, writes a JSON (JavaScript Object Notation) manifest listing each variable, alongside one blob file per value.
For each variable, the manifest records either the value itself (small primitives are inline) or its blob-file name.
Its `cache_type` field records which strategy of {ref}`sec:dispatch` produced the key.
Abridged, a manifest for a cell that defined a seed and a large array might read:

```json
{
  "hash": "9V5v6Cji…",
  "cache_type": "ContentAddressed",
  "defs": {
    "seed": {"primitive": 7},
    "x": {"reference": "blob-3fa9…"}
  },
  "stateful_refs": [],
  "meta": {"version": 4, "blob_hashes": {"blob-3fa9…": "e3b0c4…"}}
}
```

On lookup, `LazyLoader` binds each variable not to its value but to a *stub*: a small placeholder object that records where the value's bytes live and how to deserialize them.
All stubs share a `load()` method, with blob formats for pickle, NumPy `.npy`, Apache Arrow, PyTorch `.pt`, and binary media.
The stub loads the value on first use; unused variables are never loaded.

(sec:wasm)=
## WASM portability

Cached values also work in marimo's static WASM export: a standalone HTML file that runs the notebook in the reader's browser without a server on Pyodide, a Python runtime compiled to WebAssembly.
When automatic cell caching is enabled, exporting a notebook through marimo's command-line interface (`marimo export html-wasm --execute`) bundles the cache manifests and blobs into that file.
When a reader opens it, the browser session derives the same keys and loads matching cached values on first use; a missing or unverifiable entry is treated as a cache miss.
Scientific articles, blog posts, and educational materials can therefore include precomputed results and trained models.

The export contains only the cached values and the user code that produced them, not external libraries.
A cell may therefore depend on packages the browser cannot import, as long as the values it defines are stored in a portable format.
Our demonstration, published at <https://dmadisetti.github.io/scipy_proceedings/>, exercises exactly this case.
On the exporting machine, the notebook trains a PyTorch model, converts it to ONNX (Open Neural Network Exchange) bytes, defines a small wrapper class, and stores a sweep of predictions as NumPy arrays.
In the browser, where Pyodide cannot currently import PyTorch, the arrays and the wrapper load through their format stubs, and a custom stub feeds the ONNX bytes to `onnxruntime-web`, an ONNX runtime for the browser, restoring a model the notebook can call.

(sec:api)=
# Using marimo's cache

marimo provides three cache mechanisms: in-memory caching within a session, persistent caching across sessions, and a mode that caches every cell automatically.
None of the three asks the user to declare dependencies or construct keys.
Every key is built as described in {ref}`sec:keys`, so invalidation follows from the notebook's dataflow graph, including changes to UI elements and `mo.state`.

## In-memory caching

The decorator `@mo.cache` memoizes a function.
Through {ref}`sec:dispatch`, each call is keyed on the function's code, arguments, and variables it uses outside its body.
Results stay in kernel memory, so a hit reads nothing from disk, but all are lost when the session ends.
This form suits values cheap to hold but wasteful to recompute, such as a function result reused whenever a slider moves.
`@mo.cache` keeps every result; the variant `@mo.lru_cache` keeps a bounded number, 128 by default, evicting the least recently used when full.

**Comparison to `functools`.** Python's built-in `functools.cache` is not well-suited to reactive notebooks.
A reactive runtime re-runs a function's defining cell whenever its code or inputs change, recreating the function with an empty `functools` cache.
Every stored result is therefore thrown away on every re-run of the defining cell, even when the change did not affect the function's behavior.

## Persistent caching

The decorator `@mo.persistent_cache` memoizes a function with the same keys but writes results to disk through the loaders of {ref}`sec:storage`, so they survive kernel restarts.
This lets a notebook restart without re-running expensive cells.
Used as a context manager, `with mo.persistent_cache(name="training"):` caches a block of code inside a cell.
On a hit, the block does not execute: its variables are restored from disk, and its side effects do not happen.
Optional arguments choose the storage directory (`save_path`) and the loader (`method="pickle"` or `method="lazy"`).
A third option, `pin_modules`, opts library versions into the key.

## Automatic caching of every cell

The decorators and the context manager cache only what the author wraps.
marimo can also cache every cell automatically.
In this mode, the runtime computes each cell's hash and checks the cache: on a hit it skips the body and restores its variables, and on a miss it runs the cell and saves the results.
A stored entry occasionally cannot be used: if one of its variables could not be serialized, the entry holds only a placeholder, and the cell re-runs along with any upstream cells needed to rebuild the missing value.
Turning on the `cache_cells` runtime option yields the automatic, cell-granular caching that {ref}`sec:intro` argued reactive notebooks make possible: the whole notebook is cached and the author marks nothing.

(sec:eval)=
# Evaluation

Caching does not always save time.
Every hit pays a fixed overhead: deriving the key, then loading the value.
When the goal is portability, or a record of where results came from, rather than speed, that overhead does not matter.
When the goal is to skip expensive recomputation, the overhead must be smaller than the cell body it avoids.

We validate the implementation by measuring three cost components — key derivation, value load, and value save — both separately and as end-to-end hit and miss paths.
Payloads are NumPy `float64` arrays from 1 MB to 1 GB.
We compare six strategies: `mo.cache`, `mo.persistent_cache` with each of its two loaders, mandala's decorated-function memoization, and diskcache in two forms, a memoizing decorator and a plain store with a fixed key.
The fixed-key form performs no key derivation at all, so it is a lower bound on hit time.
Each cost is measured over 10 runs after priming and a warmup, and we report the median.
Every persisted measurement is keyed on a host fingerprint (operating system, architecture, and Python version) and a methodology-version string, so no result is reused across hosts or after the harness changes.

(sec:e2e)=
## End-to-end cache evaluation

A notebook user experiences the cache as a single delay: the time from editing one cell to the moment a downstream cell's value is available in Python again.
Panel (a) of {ref}`fig:cache-eval` reports that end-to-end time for six strategies.
All six cluster together up to roughly 49 MB.
The dashed line at 100 ms marks the threshold below which a response reads as instantaneous [@card1991information], and every persistent method stays under it across typical exploratory payload sizes.

The dotted curve in panel (a) shows when caching pays off.
With hit rate $p$, caching saves time when the cell body costs more than $T_{\textrm{hit}} + \frac{1-p}{p}\,(T_{\textrm{key}} + T_{\textrm{save}})$.
On these payloads the measured miss overhead is comparable to the hit cost, so at $p = 0.9$ the break-even body cost is roughly 10% above the hit curve.

Panel (b) decomposes the largest-payload hit into key derivation and value load, next to the overhead a miss adds (key derivation and value save).
mandala derives its key with `joblib.hash`, which serializes the value through pickle before hashing it [@makelov2024mandala].
marimo hashes the array's contiguous bytes directly through Python's buffer protocol, the content addressing of {ref}`sec:dispatch`, with no serialization step.
On the Apple M4 Max used for the camera-ready figures, hashing is fast, and the extra pickle pass costs mandala roughly 3× end to end.
On a Linux x86-64 server, where memory copies are cheap relative to hashing, the pickle pass costs little, value load dominates instead, and the penalty shrinks to roughly 1.2×.
marimo's value-load time matches that of the fixed-key diskcache form, which performs no key derivation, confirming that the gap comes from key derivation rather than from storage.
Panel (c) exposes per-call variance; it also shows `diskcache.memoize` failing once a payload exceeds its SQLite blob-size ceiling, so its largest sizes drop out of the sweep.


```python {.marimo hide_code="true" name="fig_cache_eval"}
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
_BENCH_VERSION = "2026-08-1gb-sweep"


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
    return compute.measure_miss(compute.random_payload(size_mb))


@mo.persistent_cache
def _miss_sweep(sizes_mb, host=_HOST_FINGERPRINT, version=_BENCH_VERSION):
    # Total miss overhead (key + save) per size, for the break-even
    # curve in panel (a).
    del host, version
    return compute.sweep_write_overhead(sizes_mb)


def cache_eval():
    import time

    # Sweep to 1 GB. The top size crosses diskcache.memoize's SQLite
    # single-blob ceiling (~1 GB, on its byte key), so memoize drops out
    # there while the others continue. The in-memory mo.cache series
    # retains the cumulative payloads across the sweep, so this needs a
    # workstation rather than a commodity laptop.
    sizes_mb = (1, 10, 50, 100, 200, 500, 1024)
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

:::{figure} figs/fig3_cache_eval.png
:label: fig:cache-eval
:width: 100%

End-to-end cache evaluation on NumPy `float64` payloads.
(a) Cache-hit latency versus payload size, on log-log axes.
The dashed line at 100 ms marks the interactive threshold [@card1991information], and the dotted curve is the break-even body cost at a 90% hit rate.
(b) Decomposition of a hit (key derivation plus value load) and a miss (key derivation plus value save) at the largest sweep size, measured on the real disk-backed paths.
(c) Per-method distribution of cache-hit samples at the largest size; `diskcache.memoize` drops out past its blob ceiling.
The host label reports the build host, the mandala-to-marimo ratio, and the cold-versus-cached cost of the figure's own sweep.
:::

The camera-ready version of this paper was evaluated on a MacBook Pro with an Apple M4 Max.
The measurements are produced by the paper's own build, and the figure's host label reports the measured mandala-to-marimo ratio, so a reproduction on different hardware shows its own number.
The stage-decomposition measurement times the real disk-backed paths on both sides: `PickleLoader.load_cache` and `LazyLoader.load_cache` for marimo, and `joblib.load` on a temp-file blob for mandala.
The load comparison therefore includes the cost of reading through the operating system's file cache and of reconstructing the cache envelope, both of which a bare `pickle.loads` would skip.


(sec:limitations)=
# Limitations and Discussion

The most salient limitations follow.

**Library versions.** The decorator and context-manager APIs do not take library versions into account by default; the user can include them with `pin_modules` ({ref}`sec:api`).
Without module pinning, upgrading a package can therefore serve stale hits.
Automatic cell caching includes library versions by default, so a version mismatch produces a cache miss instead.

**Python versions.** Because code is hashed as bytecode, keys are specific to the Python version ({ref}`sec:dispatch`).
Upgrading the interpreter therefore invalidates every cached result.
Unlike a library upgrade, this failure is safe: the keys miss and the cells recompute.
It is also why the browser export requires the exporting interpreter to match the browser's Python version ({ref}`sec:wasm`).

**Mutation outside the graph.** Mutations that bypass the dataflow graph — aliased mutation through a closure, or attribute writes on an object that cannot be content-addressed — can change a value without changing any key, so downstream cells can be served stale results.
A suite of checks that runs on every build of this paper measures exactly this class of undetectable false positive, and Rex [@zheng2025reactive] probes the same boundary.

**Cache tampering.** Unpickling can execute arbitrary code, so loading from a cache that an attacker has modified could run a malicious payload.
When cryptographic signing support is available, the lazy loader signs caches by default: each signed manifest carries an Ed25519 signature that also covers the SHA-256 hash of every blob, and signatures are checked before any bytes are deserialized.
The pickle loader has no such protection, and its caches should be loaded only from trusted sources.

**Side effects.** The cache is mostly blind to side effects such as file reads, network calls, and wall-clock queries.
marimo does provide a mechanism for tracking a side effect explicitly: the side effect is wrapped in a handle whose value is folded into the cell's hash as if it were an external reference.
Two such handles exist today, `mo.watch.file` and `mo.watch.directory`, keyed on file content and directory listing respectively.
Randomness and wall-clock time could bind to similar lifetime-managed handles.

None of these limitations is fundamental to the approach.
Future work can address each one by adding branches to the key dispatch.

(sec:future-work)=
# Future work

Our evaluation and limitations point to three directions.

**Cost-aware policy.** {ref}`sec:e2e` measured what a cache hit costs and what a cache miss adds. With those numbers available at runtime, marimo could apply the break-even rule automatically: decline to cache a cell whose body runs faster than the cache's own overhead, and report the time each cached cell actually saved.

**Expanded side effects.** {ref}`sec:limitations` described handles that fold a side effect's value into a cell's key. Handles for randomness (`mo.random`), network requests (`mo.request`), and time (`mo.clock`) would be straightforward additions. Whether tracking more side effects justifies the added interface surface remains an open question.

**Chain of trust for exports.** The signing described in {ref}`sec:limitations` protects a cache against tampering, but a shared or exported cache also requires the reader to know which public key to trust. Establishing that chain of trust, from the machine that produced an export to the reader's browser session, remains open.

Two further questions remain open: a storage policy (eviction, per-codec footprint), and cross-session memoization aligned with recorded provenance [@pimentel2017noworkflow].

(sec:conclusion)=
# Conclusion

We have presented a caching mechanism for marimo, a reactive notebook for Python.
The cache key for a cell is built from the cell's compiled body and the content-addressed values of its references, falling back to the producing cell's hash when a reference cannot be addressed by content.
This construction preserves reactive determinism, survives superficial edits, and supports portable exports.
Our evaluation shows that a marimo cache hit performs comparably to existing scientific-Python memoizers while requiring no annotations from the user.
Finally, the cache participates in marimo's static HTML/Pyodide WASM export, so users can share notebooks with precomputed results and models, as the export published at <https://dmadisetti.github.io/scipy_proceedings/> demonstrates.

## Acknowledgments

Portions of this work were assisted by a generative AI tool (Claude, Anthropic). Claude was used to help develop and run the benchmark harness reported in the Evaluation. All benchmark code and results were reviewed, verified, and revised by the authors, who take full responsibility for the accuracy and integrity of the final content.


% --- Bibliography is rendered by mystmd from references.bib ---

# Embedding GPU kernels from a library crate — the artifact-anchor fix

*A from-scratch explanation. Assumes you know Rust and roughly how a linker
works (it stitches compiled object files into one executable), but explains the
specific linker behavior this bug hinges on — archive-member selection — from
the ground up. This documents a small fix we want to contribute upstream to
cuda-oxide, and, more importantly, the alternatives we weighed, because the
obvious linker-level answers turn out not to work transparently.*

---

## 1. What we're proposing, in one paragraph

cuda-oxide lets you write a GPU kernel module in Rust (`#[cuda_module]`) and
**embed** the compiled GPU code directly inside your final executable, so the
program is self-contained — no loose `.ptx` files to ship alongside it. This
works when the module lives in the binary crate you're building. But when the
module lives in a **dependency library crate** — the natural way to ship a
reusable GPU-kernel library — the embedded GPU code silently fails to make it
into the final binary, and loading it at runtime fails with *"embedded CUDA
module '…' was not found."* The fix is ~90 lines: give the embedded payload a
named **anchor symbol**, and have the generated loader hold a reference to it, so
the linker is forced to pull the payload into the binary. This document explains
why the payload goes missing, exactly how the anchor fixes it, and — at length —
why the more obvious linker-flag solutions don't work from a library.

---

## 2. Three things you need to know first

### 2a. How a `#[cuda_module]` gets embedded

When cuda-oxide compiles a `#[cuda_module]`, it produces, alongside the normal
Rust code, a small **host object file** that carries the compiled GPU code (PTX
or cubin) as a blob inside a dedicated section named `.oxart` ("oxide artifact").
At runtime, a loader (`load_embedded_module`) walks the running executable's own
sections, finds the `.oxart` blob by the crate's name, and hands the GPU code to
the driver. That's what "embedded" means: the GPU code travels *inside* the
executable, in a section, rather than in a separate file.

The crate-name keying matters later: each crate's bundle is tagged with its
`CARGO_PKG_NAME`, and the generated `kernels::load(ctx)` asks the loader for the
bundle named after *its own* crate.

### 2b. What an rlib is, and archive-member selection

When you compile a Rust **library** crate, the compiler packages it as an
**rlib** — which is, mechanically, a Unix `ar` **archive**: a bundle of object
files (`.o`) glued together, much like a `.zip` of object files. A binary crate,
by contrast, hands its object files to the linker directly.

Here is the crucial linker rule, and the entire root of the bug:

> **Archive-member selection.** When the linker processes an archive (an rlib),
> it does **not** include every member. It includes a member **only if that
> member defines a symbol that something already in the link is looking for** (an
> *undefined reference*). Members that define nothing currently wanted are
> skipped entirely.

This is decades-old Unix linker behavior and it's usually exactly what you want:
it's how you link against a big library and only pay for the parts you use.

A **symbol** here is just the linker-level name of a function or a piece of data
(e.g. `kernel_lib::scale`). "Undefined reference" = some object says *"I call
symbol X but I don't define it; someone please provide it."* The linker then goes
looking for X in the archives — and pulling in the member that defines X.

### 2c. `--gc-sections` and `RETAIN` — a related rule that is *not* the one that bites

There's a second, separate cleanup the linker can do: `--gc-sections` drops
individual *sections* that nothing references, even from objects that *were*
included. The flag `SHF_GNU_RETAIN` (Rust's `#[used(linker)]`) marks a section as
"never garbage-collect me."

Keep these two mechanisms distinct, because conflating them is exactly how this
bug hides:

- **Archive-member selection** decides whether an object is pulled *into* the
  link at all. (Happens first.)
- **`--gc-sections` / `RETAIN`** decides whether an already-included section is
  kept or swept. (Happens later, and only to things already in.)

`RETAIN` protects a section *that made it in*. It does nothing if the object was
never selected from the archive in the first place.

---

## 3. The bug, precisely

The `.oxart` host object is **already** flagged `SHF_ALLOC | SHF_GNU_RETAIN` — so
the original author correctly guarded it against `--gc-sections`. But that object
defines **no symbol anything references.** It is pure passive data: a blob in a
section, with no functions, no exported statics, nothing the rest of the program
calls or names.

So when that object is an **archive member** of a dependency library's rlib,
archive-member selection (§2b) skips it — there's no undefined symbol it could
satisfy. The object is never pulled into the binary, the `.oxart` section never
arrives, and at runtime the loader finds no bundle:

```
embedded CUDA module 'kernel-lib' was not found
```

The `RETAIN` flag never gets a chance to matter, because the object never made it
past selection. (§2c is the trap: it *looks* guarded.)

**Why binary crates work and library crates don't.** A binary crate's `.oxart`
object is handed to the linker as a *root* input, not packed in an archive — root
inputs are always included, no selection applied. Library crates are rlibs
(archives), so their `.oxart` object is subject to selection, and loses.

---

## 4. The fix: a referenced anchor

The fix makes the payload object satisfy an undefined reference, so
archive-member selection has a reason to pull it in. Concretely, ~90 lines across
five files:

1. **A reserved symbol name** (`reserved-oxide-symbols`). A new helper,
   `artifact_anchor_symbol(crate_name)`, produces a per-crate symbol like
   `cuda_oxide_artifact_anchor_246e25db_kernel_lib`. (The `246e25db_` infix is the
   project's existing collision-avoidance convention; the crate name is sanitized
   so hyphens become underscores.) One function is the single source of truth, so
   the definition and the reference can never drift apart.

2. **Define the anchor in the payload** (`oxide-artifacts`). When the backend
   builds the `.oxart` host object, it now also defines a global symbol — the
   anchor — *located in the `.oxart` section*. The blob now has a name. The
   section is flagged `SHF_ALLOC` only — deliberately *not* `SHF_GNU_RETAIN` (see
   the two hops and §5/§6 for why retention is now the anchor's job, done
   precisely).

3. **Pass the crate's anchor name through** (`rustc-codegen-cuda`). The backend
   keys the anchor on the same `CARGO_PKG_NAME` it already keys the bundle on.

4. **Reference the anchor from the generated loader** (`cuda-macros`). The
   `#[cuda_module]` macro expands `kernels::load_named` to include:

   ```rust
   unsafe extern "C" {
       #[link_name = "cuda_oxide_artifact_anchor_246e25db_kernel_lib"]
       static __OXIDE_ARTIFACT_ANCHOR: u8;
   }
   core::hint::black_box(core::ptr::addr_of!(__OXIDE_ARTIFACT_ANCHOR));
   ```

   This is an *undefined reference* to the anchor: "I use this symbol; someone
   provide it." `black_box(addr_of!(…))` forces the address-take to survive
   optimization so it lowers to a real relocation.

**How it pulls the object in — the two hops.** The consumer's binary references
`kernels::load` → `load_named`. That drags `load_named`'s code object out of the
library's rlib (normal selection: the binary wanted it). That object now carries
an *undefined reference to the anchor*. To satisfy it, the linker goes back to the
rlib and pulls in **the member that defines the anchor — the `.oxart` payload
object.** The blob is now in the binary; the runtime loader finds it.

**Why we drop `RETAIN`, and the precise-liveness it buys.** It's tempting to also
keep the old `SHF_GNU_RETAIN` flag "to be safe" — but that flag is now both
redundant *and* harmful, and getting this right is the difference between the fix
including dead weight and not. The subtlety is that archive-member selection works
at **object-file granularity**, not per-function: `load_named` shares an object
file (codegen unit) with other library code, so a consumer that references
*anything* nearby — even a wholly unrelated, non-GPU function from the same
library — pulls `load_named`'s object in, and with it the anchor reference, and
with *that* the payload. If the `.oxart` section is `RETAIN`-pinned, it then
*stays* even though nothing live uses it — the embedded bundle becomes dead weight
in a binary that never loads it.

Dropping `RETAIN` and relying on the anchor reference alone fixes this, because
`--gc-sections` keeps a section only while something *live* references it:

- **Embedded path used:** `load_named` is live → it references the anchor → the
  `.oxart` section is kept. ✓
- **Embedded path unused** (consumer only used unrelated library code, so
  `load_named` got pulled in but is itself unreferenced): `--gc-sections` strips
  `load_named`, the anchor reference disappears, and the now-unreferenced `.oxart`
  section is reclaimed. ✓ No dead bundle.

So retention becomes *precise*: the embedded bundle is present **exactly when the
embedded loader survives the link**, and absent otherwise — for library *and*
binary crates alike. (This even removes a pre-existing inefficiency: a binary
crate that defines a module but loads its PTX from a file used to carry a dead
embedded copy, kept alive solely by `RETAIN`.) We verified all of this
empirically — see §7.

---

## 5. Alternatives we considered

The maintainers will reasonably ask: *can't the linker do this for us, without
emitting an anchor symbol from compiled code?* We looked hard. The short answer:
the obvious linker-native options can't be driven **transparently from a library
crate** through Cargo, and the one genuinely cleaner option is a deeper
refactor. Here is the full reasoning.

### Alt A — `--whole-archive` (link the entire rlib)

`--whole-archive` tells the linker to include *every* member of an archive,
bypassing selection. It would pull the payload in.

Rejected:
- **It's a consumer-side decision.** The flag is applied at the *final binary's*
  link, so the binary author would have to know that some dependency embeds GPU
  bundles and opt in per-dependency. That defeats the entire point — a library
  should be a transparent dependency.
- The mechanism is for *native* static libraries (`#[link(modifiers =
  "+whole-archive")]`); rustc owns how rlibs are fed to the linker, and there's no
  stable, library-author-controlled "link my rlib whole" knob.
- It's over-inclusive: it would drag in *all* of the library's unused code too,
  bloating the binary and defeating dead-code elimination.

### Alt B — force the symbol with a linker flag (`-u` / `--undefined`)

`-Wl,--undefined=SYM` injects an artificial undefined reference, forcing the
member that defines `SYM` to be selected — the exact effect of our anchor, but via
a link flag instead of compiled-in code. The natural place for a library to emit
it would be a build script: `cargo::rustc-link-arg=-Wl,--undefined=…`.

Rejected, and this one is decisive:
- **Cargo does not propagate a *dependency's* `rustc-link-arg` to the downstream
  binary's link.** `rustc-link-arg` (and its `-bins`/`-tests`/… variants) only
  affects the *emitting* crate's own targets. A library cannot inject link args
  into a consumer's binary — by design, so dependencies can't silently rewrite
  your link. (`rustc-link-lib`/`rustc-link-search` propagate; `rustc-link-arg`
  does not.)
- So the flag would have to be added by the *binary* author — back to the
  non-transparency of Alt A.

This is the crux of why we use a **source-level symbol reference**: a reference in
compiled code is a first-class part of the object graph and propagates through
normal linking automatically. It sidesteps Cargo's link-arg propagation rules
entirely, because it isn't a link arg — it's just a symbol the loader code uses.

### Alt C — `#[used]` / `#[used(linker)]` / `SHF_GNU_RETAIN`

These keep a symbol or section alive through *garbage collection* of things
already linked. They have **no effect on archive-member selection** (§2c). The
code already set `RETAIN`; that's precisely why the bug was subtle — the section
looked guarded but was never selected in the first place. This is the trap, not
the fix. Worse, as §4 explains, keeping `RETAIN` *alongside* the anchor causes the
dead-weight inclusion we want to avoid (it unconditionally pins the section once
selection drags the object in), so the fix actively **drops** it and lets the
anchor reference provide precise, `--gc-sections`-driven retention instead.

### Alt D — merge `.oxart` into an already-linked object (no anchor at all)

The cleanest *conceptual* fix: don't emit the payload as its own standalone
object that becomes an unreferenced archive member. Instead, attach the `.oxart`
section to an object that **already** defines symbols the consumer references —
e.g. the host object carrying `load`/`load_named` themselves. Then the section
rides in on normal selection, and no anchor is needed.

We didn't take this — but it's the alternative most worth flagging:
- The payload is produced at a *later* stage than host code generation: only after
  device codegen has produced PTX/cubin does the backend synthesize the artifact
  object (via `oxide_artifacts::build_host_object_for_target`) and append it to
  the rlib. By then, the "live" host objects are already emitted and sealed.
  Re-routing the bundle into a live object, or injecting a section into an
  already-emitted one, is a deeper change to the backend's object-emission flow
  and risks fighting rustc's own object management.
- **The anchor is the minimal, surgical realization of exactly this idea** —
  "make the payload reachable from a live symbol" — without restructuring *when or
  how* objects are emitted. If the maintainers prefer the deeper version, the
  anchor is a faithful stepping-stone to it and easy to remove later.

### Alt E — a cross-crate registry (`inventory`/`linkme`-style)

Distributed-slice crates collect items across crates into a startup registry.
They don't escape the problem: they still rely on the registering objects being
*selected into the link* in the first place (via referenced symbols or
consumer-named items). A passive blob in an otherwise-unreferenced archive member
still needs *something referenced* to drag it in — which is the anchor again, with
more machinery on top. No advantage.

### Alt F — give up embedding from libraries; load PTX from a file

This is the status-quo workaround (and what cuda-oxide's existing
`cross_crate_kernel` example does): emit a `.ptx` file and `load_module_from_file`
at runtime. It works, but it discards the value of embedding — a single
self-contained binary, with no loose `.ptx` to deploy, no working-directory or
path fragility. For a library meant to drop into someone else's binary, file
loading pushes a deployment burden onto every consumer. We want the embedded path
to actually work, not to route around it.

### Summary

| Option | Why not |
|---|---|
| A. `--whole-archive` | consumer-side, non-transparent, over-inclusive |
| B. `-u` link flag from a build script | Cargo won't propagate a dep's link-arg to the binary |
| C. `#[used]` / `RETAIN` | wrong layer — doesn't affect archive selection (the trap); and keeping it alongside the anchor reintroduces dead weight, so we drop it |
| D. merge into a live object | the principled fix, but a deeper backend refactor; the anchor is its minimal form |
| E. registry crate | reduces to the anchor with extra machinery |
| F. load PTX from a file | abandons embedding's self-contained-binary benefit |

The anchor wins because it is the *only* option that is (a) controlled entirely by
the library, transparently to the consumer, (b) precise — pulls the payload in
exactly when the embedded loader is used, and zero-cost otherwise, and (c) tiny
and additive, with no change to how objects are emitted.

---

## 6. Why the reference is robust

Maintainers will probe whether the optimizer can defeat the anchor. It can't:

- **Optimization.** `core::hint::black_box` plus `core::ptr::addr_of!` of an
  `extern` static with an explicit `#[link_name]` forces an address-taken use that
  survives optimization and lowers to a relocation against the anchor.
- **Fat LTO.** LTO operates on LLVM bitcode; the `.oxart` payload is a *synthesized
  ELF object* — final machine code, opaque to LTO. LTO cannot see "nothing really
  needs this" across that object boundary, so it cannot delete the cross-object
  relocation. The payload's not being LTO-visible is what makes the reference
  immune to whole-program optimization.
- **Symbol hygiene.** The anchor carries the reserved `246e25db_` infix and is
  keyed on the sanitized crate name, so it's unique per crate and won't collide
  with user symbols. A single `artifact_anchor_symbol` helper is the source of
  truth for both the definition and the reference.
- **No dead weight.** Because the section is `SHF_ALLOC`-only (not `RETAIN`), the
  bundle is subject to `--gc-sections` and survives only while the referencing
  loader code is live. A consumer that never calls the embedded loader carries no
  embedded bundle, even if it depends on the library for other reasons (§4, §7).

---

## 7. What we've proven

We re-derived the fix onto the current upstream tip and validated it end to end.

- **Compiles and unit-tests pass** on `upstream/main` (post the `pliron-llvm`
  migration): `reserved-oxide-symbols` (incl. the new anchor doctest),
  `oxide-artifacts` round-trip, and `cuda-core` embedded tests are all green.
- **A new example, `cross_crate_embedded`**, compiles clean through the CI's
  GPU-less compile gate (`cargo oxide build`). It is a library crate with a
  concrete `#[cuda_module]` kernel, loaded from a binary via
  `kernels::load(ctx)` — the embedded path, the one the fix repairs. (We chose a
  *separate* example rather than extending `cross_crate_kernel`, because that
  example loads PTX from a file and adding a concrete library kernel to its module
  breaks its file-load path.)
- **Hardware, with a before/after control.** On an sm_70 Titan V, the exact
  example loads the library's embedded bundle and runs correctly (all 1024
  elements `× 2`). With the anchor *reference disabled* in the macro — the negative
  control — it regresses to exactly `ModuleNotFound { name: "device-lib" }`. The
  anchor is demonstrably the load-bearing piece.
- **Precise liveness, measured.** We built a consumer that depends on the library
  but calls only an unrelated non-GPU function, never the embedded loader, and
  inspected the linked binary with `readelf`/`nm`:
  - With `RETAIN` kept: the `.oxart` section is **present** — the embedded bundle
    is dead weight in a binary that never loads it.
  - With `RETAIN` dropped (the shipped design): the `.oxart` section is **absent**
    — `--gc-sections` reclaims it.
  And in the *used* case (the embedded loader is called), `.oxart` is present and
  the kernel runs correctly either way — confirming the live anchor reference
  alone is sufficient retention. So the fix adds zero footprint to consumers that
  don't use the embedded path.

  (The Titan V is sm_70; upstream's PTX path currently floors at sm_80, a separate
  pre-Blackwell gap. The fix is architecture-independent, so we ran the hardware
  check through our sm_70-capable fork, which carries the byte-identical change.
  Independently, our own gale-gpu *is* a library crate loaded this way on sm_70
  daily.)

---

## 8. Scope and upstream framing

This is a focused, additive bug fix: a `#[cuda_module]` in a dependency library
crate should embed and load just like one in a binary crate, and today it
doesn't. The change is ~90 lines across five crates plus a demonstrating example.
It leaves binary-crate embedding, file loading, and non-embedded use working, and
replaces the old blunt `RETAIN` pin with a precise anchor reference, so embedded
bundles are retained exactly when used — a small improvement for binary crates
too. If the maintainers prefer the deeper "merge into a live object" approach
(Alt D), the anchor is a clean stepping-stone we'd be glad to evolve.

---

### Glossary

- **`.oxart` / artifact object** — the host object file cuda-oxide emits to carry
  a `#[cuda_module]`'s compiled GPU code, as a blob in an `.oxart` section, so it
  can be embedded in the executable.
- **rlib** — a compiled Rust library, mechanically a Unix `ar` archive of object
  files.
- **Archive-member selection** — the linker rule that an archive member is
  included only if it defines a symbol something references; the root of this bug.
- **Symbol** — the linker-level name of a function or datum; an *undefined
  reference* is a use of a symbol the object doesn't itself define.
- **`--gc-sections` / `SHF_GNU_RETAIN` (`#[used(linker)]`)** — a *separate*
  cleanup that drops unreferenced sections from already-included objects; `RETAIN`
  opts a section out. Distinct from selection.
- **Anchor symbol** — the per-crate global symbol this fix defines inside the
  `.oxart` section and references from the generated loader, so selection pulls the
  payload in.
- **PTX / cubin** — NVIDIA's GPU assembly and final binary formats; the GPU code
  carried in the bundle.
- **Primal / embedded path / `kernels::load`** — the generated runtime entry point
  that reads the embedded bundle (as opposed to `from_module`, which loads GPU code
  from an external PTX file).
```

# Differentiable GPU kernels in Rust — the `oxidiff` design

*A from-scratch explanation. Assumes you know roughly what a compiler does
(source code → machine code) but nothing about compiler "intermediate
representations." Every such concept is explained where it first comes up.*

---

## 1. What we're building, in one paragraph

We write GPU physics simulations in Rust — specifically a viscoelastic fluid
solver. For tasks like **fitting a material's parameters to experimental data**,
or shape/optimization problems, we need *derivatives* of those GPU computations:
"if I nudge this parameter a little, how does the result change?" Working those
derivatives out by hand is tedious and bug-prone. We want to **put one annotation
on a GPU kernel and automatically get a second GPU kernel that computes its
derivative** — no manual calculus, and it runs on the GPU just like the original.
This document explains how we do that, why the design looks the way it does, and
what we've actually proven works on real hardware.

The headline demo: we fit the two parameters of a viscoelastic material model to
synthetic "measured" data by gradient descent, where every gradient is computed by
automatically differentiating through the real GPU solver. It recovers the true
parameters to machine precision.

---

## 2. Three things you need to know first

### 2a. The GPU compiler: cuda-oxide

Normally GPU code is written in CUDA C++. We instead write it in Rust, using an
open-source project called **cuda-oxide**. It's a compiler that turns Rust
functions you mark as GPU "kernels" into GPU machine code. It works by plugging
into the normal Rust compiler (`rustc`) as a **backend**: `rustc` understands the
Rust language, and cuda-oxide takes over the final job of "turn this into
instructions a GPU can run." (For our purposes, "backend" = the part of a compiler
that emits the actual machine code.)

### 2b. Automatic differentiation, and Enzyme

**Automatic differentiation** ("autodiff") is a technique where a tool, given a
function, mechanically produces a *new* function that computes the original's
derivative — exactly (not an approximation), and without you doing any calculus.

**Enzyme** is a well-known autodiff tool. The unusual, powerful thing about it is
*where* it operates: not on your source code, and not on the final machine code,
but on the compiler's internal description of the program (next section). Because
it works at that level, it can differentiate almost anything the compiler can
compile — loops, calls into math libraries, and GPU code included.

A quick note on two "modes" of autodiff, because they come up later:
- **Forward mode** answers "I nudged *one input* — how did *all the outputs*
  move?" Cheap when you have few inputs.
- **Reverse mode** answers "here's how much I care about *one output* (a loss) —
  how should I nudge *every input* to improve it?" Cheap when you have one output
  and many inputs. This is the mode machine learning uses, and the one our
  parameter-fitting uses: a single reverse pass gives the gradient with respect to
  *all* the parameters at once.

### 2c. The "IR" — the idea this whole design hinges on

A compiler doesn't jump straight from source code to machine code in one leap. It
first rewrites your program into a simpler, standardized middle language called an
**Intermediate Representation**, or **IR**.

Think of the IR as a stripped-down, uniform, "assembly-like" description of *what
the program does*, with all the human-friendly surface syntax removed. The key
facts:

- Many different source languages (C, C++, Rust, …) all compile *down to the same
  IR*, called **LLVM IR**.
- Because it's a shared, standard format, there's a whole ecosystem of off-the-
  shelf tools that read and rewrite programs at the IR level. Enzyme is one of
  them.

So the mental model for the rest of this document is just this chain:

```
   Rust source  →  [ rustc + cuda-oxide ]  →  LLVM IR  →  GPU machine code
                                               ▲
                                   everything clever we do
                                       happens right here
```

Whenever this doc says "the IR," picture that middle box: a text-like description
of the program, after the language part is done but before it becomes GPU
instructions. (The GPU "machine code" at the end is actually two formats with
names you'll see in passing — *PTX*, a GPU assembly, and *cubin*, the final binary
the GPU loads. You can treat them both as "the GPU's machine code.")

---

## 3. The core idea: bridge Rust's built-in autodiff to the GPU

Rust recently gained a *built-in* autodiff feature called `std::autodiff`. You
annotate a function:

```rust
#[autodiff_forward(d_f, ...)]   // "also make me the derivative, called d_f"
fn f(x: f64) -> f64 { ... }
```

…and the compiler is *supposed* to generate `d_f`. Under the hood, the annotation
is a note that tells `rustc`: "when you compile this, hand it to Enzyme."

**The catch:** `rustc` only performs that hand-off on its *normal CPU backend*. For
GPU code, cuda-oxide *replaces* that backend — so the annotation reaches a dead
end. Nobody runs Enzyme; the derivative is never produced.

**Our bridge:** keep the nice built-in annotation as the "front door," but perform
the Enzyme step *ourselves*, at the IR level, after cuda-oxide has produced the IR.
The user writes the familiar annotation; we make sure Enzyme actually runs on the
GPU's IR, and a real derivative kernel comes out the other side. We never touch
the calculus; Enzyme does that. We just arrange for Enzyme to be *invoked* in a
place the language designers didn't anticipate (GPU code).

---

## 4. The architecture we converged on

A guiding goal: keep almost everything *outside* cuda-oxide. We don't want to
maintain a permanent private fork of someone else's compiler, and we'd like the
design to be reusable by others. It comes down to three pieces.

### Piece 1 — `oxidiff`: a standalone crate (all the Enzyme-specific machinery)

`oxidiff` is a normal Rust package we own. It provides:

- **The annotation macros.** We reuse the familiar names `autodiff_forward` /
  `autodiff_reverse`, but as our crate's own versions. A user importing them from
  `oxidiff` gets GPU behavior; the only line that changes versus standard Rust
  autodiff is the import.
- **A small build-time tool.** This is the thing that does the real work: it reads
  the IR, figures out what to differentiate, runs Enzyme, and produces the GPU
  machine code.

### Piece 2 — one tiny, general addition to cuda-oxide: a "post-IR hook"

The *only* thing we'd ask the cuda-oxide project itself to add is a small, general
extension point: **after cuda-oxide produces the IR, if a hook is configured, hand
the IR to an external program and let that program finish the job.**

That's the whole change — roughly thirty lines. cuda-oxide doesn't need to know
anything about Enzyme, derivatives, or our project. It just grows a sanctioned
"pause point" near the end of compilation where an outside tool can step in. (The
same hook is useful for completely unrelated things — running a custom
optimization, adding profiling instrumentation, or swapping in a different GPU
code generator — which is exactly why it's a reasonable thing to ask for.) We call
the hook `CUDA_OXIDE_POST_IR`.

The important architectural consequence: **everything opinionated, fragile, and
Enzyme-specific lives in `oxidiff`**, on the far side of that hook. cuda-oxide
stays clean.

### Piece 3 — the "marker" trick (how the instructions travel)

There's a subtle gap to bridge. Our macro knows *what* to differentiate — which
function, forward or reverse, which inputs matter — but it knows this at *annotation
time*, while the program is still being compiled. The external tool runs *later*
and sees only the IR. How does the "differentiate this, this way" instruction get
from the first place to the second, **without** us reaching into the compiler's
private internals (the thing we're trying to avoid)?

Our answer: **hide the instructions inside a function name.**

When you annotate a kernel, our macro quietly emits one extra, tiny GPU function
whose *name* spells out the request — something like:

```
__oxidiff_k__d_relax__implicit_relax__forward__Const_Const_Dual_Dual
            └ derivative └ original      └ mode    └ which inputs are "active"
              name          function
```

Function names are guaranteed to survive into the IR — the compiler keeps them as
**symbols** (the labels by which functions are referred to). So later, the tool
just scans the IR for any function whose name starts with `__oxidiff_`, and reads
the entire instruction straight out of the name. It's like a labeled sticky note
that rides along with the program and is still readable at the end. No compiler
internals, no side files, no extra coordination.

This marker function does a second quiet job: it *calls* the original function.
That matters because compilers throw away code that nothing uses, and a function
you only intend to differentiate might otherwise look unused and get deleted before
the tool ever sees it. By having the marker call it, we guarantee the original
function is present in the IR for Enzyme to work on. ("The original function" — the
thing being differentiated — is called the **primal**, a term you'll see below.)

---

## 5. How it works, end to end

Putting the pieces together, here's the whole journey:

1. **You write** a GPU kernel and annotate it with `oxidiff`'s `autodiff_forward`
   / `autodiff_reverse`. You also keep writing normal kernels as usual.
2. **The macro expands** at compile time: it leaves your function in place (the
   *primal*) and adds the tiny marker function whose name encodes the request and
   which calls the primal.
3. **cuda-oxide compiles** everything to IR, exactly as it always does. The primal
   and the marker are now both in the IR; the marker's encoded name is intact.
4. **The post-IR hook fires:** cuda-oxide hands the IR to `oxidiff`'s tool.
5. **The tool does the work:** it scans the IR for `__oxidiff_` markers, reads each
   request, runs **Enzyme** on the corresponding primal to synthesize the
   derivative, and then finishes turning the IR into GPU machine code.
6. **You run it:** the derivative kernel is now an ordinary GPU function you can
   launch by name, sitting right alongside your normal kernels.

From the user's point of view it's just: *annotate, build, launch the derivative.*
Everything in steps 2–5 is invisible.

---

## 6. What we've actually proven

This isn't a paper design. We built it end to end and tested it on a real GPU (an
NVIDIA Titan V), and then deliberately tried to *break* it.

**It computes correct derivatives of the real solver, on the GPU.** We
differentiated our actual viscoelastic relaxation kernel — which includes a
data-dependent iterative solver and calls into the GPU math library — with respect
to a material parameter. The automatically-produced gradient matches a
finite-difference sanity check to about seven decimal places.

**The headline use case works.** Using *reverse-mode* autodiff (one pass gives the
gradient for all parameters at once), we fit the two parameters of the material
model to synthetic measured data by gradient descent. It recovers the true values
to machine precision.

**We red-teamed the risky design choices** — the ones where, if they failed, we'd
have been forced back into hacking the compiler internals:

- *Does the "marker in a function name" actually survive compilation?* Yes —
  marker functions land in the IR with their names intact, even long names that
  encode a dozen activity flags, and the compiler's "delete unused code" step
  can't remove them.
- *Does the primal survive even if nothing else calls it?* Yes — the marker
  calling it is enough to keep it present, as a full standalone function Enzyme can
  differentiate.
- *Compatibility wrinkle.* The IR comes in two slightly different "dialects," and
  Enzyme prefers the newer one, while our older GPU target emits the older one. We
  confirmed that the standard LLVM tooling *transparently converts* the old dialect
  to the new one on the way in, with no special handling from us — and that the
  full real kernel, differentiated through that conversion, still produces the
  correct gradient on the GPU (the same seven-decimal match). This was the scariest
  unknown, and it held.

The net result: the design needs **no changes inside cuda-oxide beyond the small,
general hook** — validated all the way to correct numbers on real hardware.

---

## 7. Should this go upstream into cuda-oxide?

Short version: **propose only the small, general hook — not "add autodiff."**

cuda-oxide is an actively maintained, contribution-friendly project, but it's
explicitly early/experimental, and automatic differentiation is nowhere on its
roadmap. Asking them to adopt Enzyme, track Rust's still-unstable autodiff feature,
and support our particular GPU target would be a big, speculative ask on top of a
codebase they're actively reshaping.

By contrast, the `CUDA_OXIDE_POST_IR` hook is a tiny, broadly-useful extension
point (custom passes, instrumentation, alternative backends — autodiff is just one
example), with a precedent already in the project. That's a realistic contribution.
Everything Enzyme-shaped then lives in our `oxidiff` crate, entirely on the far
side of that hook. If Rust's autodiff feature later stabilizes, *that's* the moment
to revisit pushing more of it upstream — it becomes a much easier conversation.

So the plan is: contribute the general hook upstream; keep `oxidiff` as our own
crate that anyone using cuda-oxide can add for differentiable GPU kernels.

---

## 8. Status and what's next

- **Proven on hardware:** forward- and reverse-mode autodiff of the real GPU
  kernel; the inverse-rheology parameter fit; the marker-and-hook design with no
  in-tree compiler changes required.
- **Still design work (not correctness risk):** the exact contract for the hook —
  in particular, supporting both "tweak the IR and let cuda-oxide finish" (the
  lightweight case most users want) and "take over and produce the final GPU
  binary yourself" (what Enzyme needs). We'd lead the upstream proposal with the
  lightweight mode.
- **Next steps:** draft the hook proposal for the cuda-oxide maintainers; build out
  the `oxidiff` crate (the macros + the tool) as a clean standalone package.

---

### Glossary

- **Backend** — the part of a compiler that emits the actual machine code.
- **IR (Intermediate Representation) / LLVM IR** — the simplified, standardized
  "middle language" a program is rewritten into during compilation; the shared
  level where tools like Enzyme operate.
- **Enzyme** — an automatic-differentiation tool that operates on LLVM IR.
- **Autodiff (forward / reverse mode)** — automatically producing a function's exact
  derivative; forward is cheap for few-inputs, reverse for few-outputs (e.g. a loss).
- **Primal** — the original function being differentiated (as opposed to its derivative).
- **Kernel** — a function that runs on the GPU.
- **Symbol** — the name a function is referred to by inside the IR / machine code;
  the thing our "marker" trick rides on.
- **PTX / cubin** — the GPU's assembly and final binary formats; together, "GPU
  machine code."
- **The post-IR hook (`CUDA_OXIDE_POST_IR`)** — the one small, general extension
  point we'd add to cuda-oxide so an external tool can step in after the IR is
  produced.

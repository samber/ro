# Vectorize API feasibility — probe results (2026-09-10)

Run from `plugins/exp/simd/` with `GOWORK=off GOEXPERIMENT=simd go test ./zzscratchprobe/... -v`
on go1.27.0 darwin/arm64 (Apple M3, NEON, 16 int8 lanes).

Methodology: one probe enabled at a time (`mv qN_test.go qN_test.go.disabled`), full
output read, verdict written here immediately. A failing file breaks the whole package
build, so a passing sibling proves nothing unless it ran in the same green build.

## Summary

The **flat generic-stage** shape (Q3/Q4/Q5) works and is strictly better than the
nested-callback bridge an earlier session recommended. The previously "confirmed"
`Vectorize[R](pipeline func(Observable[Masked]) Observable[R])` bridge is **not viable**
for the real API — it only works when `R` is itself a simd type, which no reduction is.

## P7 control — reproduces the earlier session's passing probe

**PASS.** `VectorizeInt8P7[R any](pipeline func(ro.Observable[maskedInt8sP7])
ro.Observable[R])` instantiated at `R = maskedInt8sP7`. Confirms the earlier result is
real and the environment is sound.

## P7-native — the same bridge, returning a NATIVE type

**FAIL.**

```
in call to VectorizeInt8P7N@simd0, type func(ro.Observable[maskedP7N]) ro.Observable[R]
of pipeline does not match func(ro.Observable[maskedP7N@simd0]) ro.Observable[R]
(cannot infer R)
```

Identical to P7 in every respect except the inner pipeline returns `ro.Observable[int64]`
instead of the masked type. Explicitly writing `VectorizeInt8P7N[int64](...)` does not
help — this is specialization, not inference.

**Why P7 passed and this does not:** when `R` is itself simd-containing, the whole call
specializes together — bridge, parameter type, and the closure literal all become
`@simd128` and match. When `R` is native, the parameter type is still cloned to
`maskedP7N@simd0` but the caller's closure stays at plain `maskedP7N`, so they no longer
match.

**Consequence: the nested-callback bridge can never return to scalar space.** Any
reduction (`ReduceSum -> int8`) is impossible in it.

## Q2 — non-generic bridge owning both vectorize and devectorize

**FAIL.**

```
cannot use pipeline (variable of type func(ro.Observable[maskedQ2]) ro.Observable[maskedQ2])
as func(ro.Observable[maskedQ2@simd0]) ro.Observable[maskedQ2@simd0] value in argument to
vectorizeQ2@simd0
```

Dropping the type parameter (`func vectorizeQ2(pipeline func(Observable[maskedQ2])
Observable[maskedQ2]) func(Observable[int8]) Observable[int8]`) fails at the compiler's own
synthesized dispatcher, which forwards the callback argument without converting it to the
clone's parameter type. A non-generic function taking a simd-typed callback is unusable.

## Q3 — flat pipe, every stage generic over the masked type

**PASS**, numerically correct.

```go
ro.Pipe3(src, vectorizeQ3[maskedQ3], addQ3[maskedQ3](42), devectorizeQ3[maskedQ3])
// [43 44 45 46 47 48 49 50 51 52]
```

No callback, no nesting. Every declaration is generic over `S` and never names a concrete
simd type; the concrete type appears **only as an explicit type argument at the call site**.

Shapes proven safe here:

- `func vectorize[S vec[S]](source ro.Observable[int8]) ro.Observable[S]`
- `func devectorize[S vec[S]](source ro.Observable[S]) ro.Observable[int8]` — masked in,
  **native out**, which is exactly what P7-native could not do
- `func add[S vec[S]](operand int8) func(ro.Observable[S]) ro.Observable[S]`

10 items with 16 lanes exercises the partial-vector path end to end.

## Q4 — core `ro.Map` over the masked type, and reduce to scalar

**PASS**, both.

```go
ro.Map(func(x maskedQ3) maskedQ3 { return x.add(42).min(50) })
// [43 44 45 46 47 48 49 50 50 50]  — min(50) clamps the last two lanes
```

Core `ro.Map` carries a concrete simd-containing struct with no plugin-specific operator
involved, and the closure fully infers its types. Method chaining works.

`reduceSumQ4[S vec[S]](source ro.Observable[S]) ro.Observable[int64]` emits `[55]` for
`1..10`. Reductions exiting to native scalars are fine in this shape.

### File-level requirement discovered here

A file containing simd-dependent code must `import "simd"` **and touch a simd type inside a
function body**. Without the import: `undefined: simd` plus `"simd/internal/bridge" imported
and not used`. With only a package-level `var _ = simd.BroadcastInt8s`: still
`"simd/internal/bridge" imported and not used`. A function body reference
(`func lanes() int { return simd.BroadcastInt8s(0).Len() }`) satisfies it.

This matches the known rule that the midway pass hard-codes the identifier `simd` when
injecting `simd/archsimd`, and extends it: a package-level var is not enough. (A file whose
functions never touch a simd type — pure generic plumbing — is exempt; the plain var
reference only has to satisfy the ordinary unused-import check there.)

### Where the boundary actually falls (observed 2026-09-11, regrouping the package)

Splitting the operators out of the per-type files put the rule to a real test, and it
draws the line at **naming a concrete simd-containing type**, not at driving vector work:

- `arithmetic.go`, `bounds.go`, `contains.go`, `reduce.go` and `vectorize.go` hold all 160
  operators and **need no `simd` import at all** — not even the blank var. Every
  declaration is generic over `V`, and the only way they reach simd is through methods on
  that type parameter.
- The test files that name `PartialInt8s` and friends **do** need it, and fail with
  `undefined: simd` pointing at the `package` clause until the import plus a function-body
  reference is added. That includes external test files in `package rosimd_test`.

So the practical test when adding a file is not "does this file do SIMD work" but "does
this file write out a concrete Partial type anywhere".

## Q5 — pinning type arguments once on `ro.Pipe`

**PASS.**

```go
ro.Pipe3[int8, maskedQ3, maskedQ3, int8](src, vectorizeQ3, addQ3[maskedQ3](42), devectorizeQ3)
```

Non-curried stages (`vectorizeQ3`, `devectorizeQ3` — those taking `source` directly) infer
their type argument from the pinned `Pipe3` type arguments.

## Q5b — can curried operators infer too?

**FAIL.** `addQ3(42)` in the same pinned `Pipe3` gives:

```
in call to addQ3, cannot infer S
```

A curried operator's type parameter appears only in the type of the func it returns, and Go
does not infer type parameters from the expected type of a call's result. This is a plain
Go generics limitation, unrelated to simd.

**Consequence:** curried operators with a non-`V` operand always need an explicit type
argument. Method chaining through `ro.Map` needs none, and a curried operator whose operand
is itself a `V` (see Q6) infers from the operand.

## Q6 — one operator over BOTH the stdlib vector type and a partial type

**PASS**, both instantiations.

```go
type int8Vector[V any] interface { Add(V) V; Min(V) V }
func addWith[V int8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V]
```

- Instantiated at `simd.Int8s`: `[101 ... 116]`.
- Instantiated at a partial type with a 3-lane mask: `[101 102 103 4 5 6 ... 16]` — the
  mask is respected, padded lanes keep their pre-op value.

The type argument is **inferred** from the operand (`addWith(simd.BroadcastInt8s(100))`),
because the operand is itself a `V`. Vector-operand operators therefore need no explicit
type argument, unlike the scalar-operand form in Q5b.

## Q7 — scalar operand, and where the broadcast may happen

SIMD has no scalar-operand add: `simd.Int8s.Add` takes another vector, so a scalar must
first be broadcast across all lanes.

**FAIL — broadcasting inside the generic operator's body.**

```
in call to addInt8Q7@simd0, cannot infer V (declared at q7_test.go:44:16)
```

The error points at the operator's own declaration line, i.e. the compiler's synthesized
dispatcher between the function and its width clones. Calling `simd.BroadcastInt8s` in the
body makes the generic operator simd-dependent and breaks that dispatcher.

**FAIL — laundering through a non-generic helper.** Moving the call into
`func broadcastInt8Q7(operand int8) simd.Int8s` changes nothing, same error. **A generic
operator's body may not reach `simd.*` at all, directly or transitively.** Q3's `addQ3` and
Q6's `addWith` passed precisely because their bodies only called methods on the opaque type
parameter.

**PASS — broadcasting at the call site**, with the constraint's method taking a concrete
`simd.Int8s` operand:

```go
type int8Vector[V any] interface { Add(simd.Int8s) V; Min(simd.Int8s) V }
func addInt8[V int8Vector[V]](broadcast simd.Int8s) func(ro.Observable[V]) ro.Observable[V]

addInt8[simd.Int8s](simd.BroadcastInt8s(100))       // [101 ... 116]
addInt8[partialInt8sQ7](simd.BroadcastInt8s(100))   // [101 102 103 4 5 ... 16]
```

`simd.Int8s` satisfies the constraint natively; the partial type implements
`Add(simd.Int8s) V` and applies `IfElse` inside. Note `V` is **not** inferred here (the
operand is not a `V`), so an explicit type argument is required — unlike Q6's `Add(V) V`
shape, where it is inferred. The shipped API uses Q6's shape for exactly that reason.

### Hard constraint: scalar operands cannot be generic over stdlib types

`go doc simd.Int8s` shows **no scalar-operand method and no constructor method** — every
binary op takes another vector, and the only non-vector-arg methods are `Len`, `Store`,
`StorePart`, `String`, `ToArch`, `ToBits`, `ToMask`. Broadcasting is a package _function_
(`simd.BroadcastInt8s`), not reachable through a type parameter.

So an operator taking a plain `int8` cannot build the operand vector for an opaque `V`, and
no interface satisfied by `simd.Int8s` can supply one. Scalar-operand operators are
therefore possible only over rosimd's own partial types; vector-operand operators work for
both.

## Design rules this establishes

1. Never declare a function that puts a concrete simd-containing type inside another
   package's generic type — `ro.Observable[PartialInt8s]` fails both as a return type
   (P7-native) and as a callback parameter (Q2). Keep those declarations generic over `S`.

   The hazard is the foreign generic container, not the simd type by itself: the shipped
   package declares `fullMaskInt8() simd.Mask8s` and `prefixMaskInt8(n int) simd.Mask8s`
   as plain functions, and they compile. Do not read this rule more broadly than the
   probes support.

2. The concrete type belongs at the call site as an explicit type argument, nowhere else.
3. `simd.*` calls and struct-literal construction live in **methods** on the concrete type,
   reached only through the constraint interface.
4. Curried operators (`func() func(source ro.Observable[S]) ro.Observable[R]`) do not
   infer their type argument, unless their operand is itself a `V` — so a non-curried
   stage saves an explicit type argument at every call site. The shipped package took the
   opposite trade-off for `ReduceSum`/`ReduceMin`/`ReduceMax`: uniformity with every other
   operator in the package (all curried) won over that inference cost.
5. Every file with simd-dependent code imports `simd` and touches it in a function body.

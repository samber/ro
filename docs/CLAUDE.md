# Adding New Method Documentation

When adding a new method to the `ro` library, follow these guidelines to create or update documentation files in the `docs/data/` directory.

## File Structure

Documentation files follow the naming pattern `core-{method-name}.md` for core methods and `plugin-{plugin-name}-{method-name}.md` for plugins.

### Frontmatter Format

Each documentation file must start with a frontmatter section:

```yaml
---
name: methodName
slug: methodname
sourceRef: file.go#L123
type: core
category: filtering
signatures:
  - "func MethodName(params)"
  - "func (receiver *Type) MethodName(params)"
  - "func MethodNameI(params)"
  - "func MethodNameWithContext(params)"
playUrl: https://go.dev/play/p/EXAMPLE
variantHelpers:
  - core#slice#methodname
  - core#slice#methodnamei
  - core#slice#methodnamewithcontext
similarHelpers:
  - core#slice#filtermethodname
  - plugin#encoding-json#othermethodname
position: 0
---
```

### Frontmatter Fields

- **name**: The display name of the helper (PascalCase)
- **slug**: URL-friendly name (kebab-case, matches filename without `core-` prefix)
- **sourceRef**: Source file reference with line number (format: `operator_conditional.go#L123`)
- **type**: `core`, `plugin`. The category must match the file name.
- **category**: The functional category. For plugins, the category must match the file name. Some plugins might have a sub-sub-category: in that case use "-" (eg: plugin > `encoding-json` or plugin > `samber-hot` or plugin > `logger-logrus`).
- **signatures**: Array of function signatures as strings. Do not list signatures from other type/category.
- **playUrl**: Go Playground URL with working example
- **variantHelpers**: Array of variant method names. Must contain at least the default method of named above. Variation XxxxWithContext or XxxxF or Xxxx2/3/4/5/... or XxxxI must be listed here. Don't list methods from other packages (type/category) of this library (must be similarHelpers instead).
- **similarHelpers**: Array of related helper names (leave empty if none). Eg: equivalent in other package/category, or helper composition (map vs filtermap), or method with callback (Last vs LastBy).
- **position**: Position in the list (0, 10, 20, 30...). Order must follow the order in source code. Helpers are grouped by type+category and displayed on a page. Position number is reset for each page.

## Content Structure

After the frontmatter, include:

1. **Brief description**: One sentence explaining what the helper does
2. **Code example**: Working Go code demonstrating usage
3. **Expected output**: Comment showing the result

```markdown
Brief description of what this helper does.

```go
obs := ro.Pipe(
    ro.Just(1, 2, 3, 4),
    ro.MethodName(example),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// expected result
```
```

## Code Style Guidelines

- Use `Pipe` instead of `PipeX` variants.
- Use `NewObserver` or `NewObserverWithContext` instead of `Observer{...}`. Prefer NewObserver when no context is used.
- Example: `NewObserver(next, error, complete)` instead of `Observer{Next: next, Error: error, Complete: complete}`
- Use OnNext/OnNextWithContext/OnError/OnErrorWithContext/OnComplete/OnCompleteWithContext instead of NewObserver/NewObserverWithContext, when only a single callback is needed.

Multiple examples can be used for demonstration the method, such as edge cases. If multiple signatures/variants are grouped under this documentation, it could be useful to describe some (all?) of them. You can create examples by yourself or read _example_test.go files.

## Grouping Related Methods

**IMPORTANT: Distinguish between variants and similar helpers**

### Variants vs Similar Helpers

**Variants** should be grouped in a single file when:
- They are true variations of the same base functionality with **identical core behavior**
- They only differ by adding context (`WithContext`), index (`I`), or both (`IWithContext`) parameters
- They have the same fundamental purpose and behavior pattern
- Examples: `Filter()`, `FilterI()`, `FilterWithContext()`, `FilterIWithContext()`
- Examples: `CombineLatest()`, `CombineLatest2()`, `CombineLatest3()`, `CombineLatest4()`

**Similar helpers** should be documented as separate files when:
- They have **different core behavior** or functionality
- They solve different problems or use different algorithms
- They are composed differently (e.g., `Map` vs `FilterMap` vs `MapErr`)
- Examples: `BufferWhen` vs `BufferWithCount` (different buffering strategies), `CombineLatest` vs `CombineLatestWith` (different calling patterns)

### Guidelines for Method Grouping

**Group together as variants (single file):**
- Base method + context/index variants: `Method()`, `MethodI()`, `MethodWithContext()`, `MethodIWithContext()`
- Simple parameter variations of the exact same behavior

**Create separate files for similar helpers:**
- Different base functionality: `BufferWhen` (boundary-based) vs `BufferWithCount` (count-based)
- Different calling patterns: `CombineLatest2` (static) vs `CombineLatestWith2` (pipe operator)
- Different algorithms: `Merge` (interleave) vs `Concat` (sequential)
- Composed methods: `MergeMap` (merge + projection) vs `Merge` (simple merge)

### Variant Helper Naming

When methods are true variants that belong in the same file:
- Use consistent suffixes:
  - `I` suffix for variants having index argument in predicate callback
  - `WithContext` suffix when `context.Context` is provided
  - `IWithContext` suffix when both index and context are provided

Don't invent variants. They must exist in the source code.

### Examples of Correct Grouping

**✅ Correct - Variants in single file:**
```yaml
---
name: Filter
slug: filter
signatures:
  - "func Filter[T any](predicate func(item T) bool)"
  - "func FilterI[T any](predicate func(item T, index int64) bool)"
  - "func FilterWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool))"
  - "func FilterIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool))"
variantHelpers:
  - core#filtering#filter
  - core#filtering#filteri
  - core#filtering#filterwithcontext
  - core#filtering#filteriwithcontext
similarHelpers: []
---
```

**❌ Incorrect - Similar helpers grouped:**
```yaml
# WRONG: BufferWhen and BufferWithCount are different strategies
---
name: BufferWhen
slug: bufferwhen
signatures:
  - "func BufferWhen[T any, B any](boundary Observable[B])"
  - "func BufferWithCount[T any](size int)"
# ...
---
```

**✅ Correct - Similar helpers in separate files:**
```yaml
# File: core-map.md
---
name: Map
slug: map
signatures:
  - "func Map[T any, R any](project func(item T) R)"
  - "func MapWithContext[T any, R any](project func(ctx context.Context, item T) (context.Context, R))"
  - "func MapI[T any, R any](project func(item T, index int64) R)"
  - "func MapIWithContext[T any, R any](project func(ctx context.Context, item T, index int64) (context.Context, R))"
similarHelpers: [core#transformation#mapto, core#transformation#maperr, core#transformation#flatmap]
---

# File: core-maperr.md
---
name: MapErr
slug: maperr
signatures:
signatures:
  - "func MapErr[T any, R any](project func(item T) (R, error))"
  - "func MapErrWithContext[T any, R any](project func(ctx context.Context, item T) (R, error))"
  - "func MapErrI[T any, R any](project func(item T, index int64) (R, error))"
  - "func MapErrIWithContext[T any, R any](project func(ctx context.Context, item T, index int64) (R, error))"
similarHelpers:
  - core#transformation#map
---
```

When multiple methods operate on the same struct or serve similar purposes, consolidate them into a single file:

**Example**: Map methods:
- `Map()` base method
- `MapI()` add index to predicate callback
- `MapWithContext()` add context to predicate callback
- `MapIWithContext()` add index and context to predicate callback

In such cases:
1. Use the primary method name in the filename (e.g., `core-map.md`)
2. Include all related signatures in the `signatures` array
3. List all related methods in `variantHelpers` array
4. Document each method in its own section with `### MethodName` headers

### Content structure for Plugin

When the method is part of a plugin, add the `import` section on top of code block. Example:

```go
import (
  "github.com/samber/ro"
  rostrings "github.com/samber/ro/plugin/strings"
)

obs := ro.Pipe(
    ro.Just("hello world"),
    rostrings.Capitalize[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Hello world
// Completed
```

### Signatures

Real method signature will have the following form: `func All[T any](predicate func(T) bool) func(Observable[T]) Observable[bool]`. Please remove the returned `func(Observable[X]) Observable[Y]` and just write `func All[T any](predicate func(T) bool)` to the markdown file, because the developer already knows implicitly a function is returned.

## Naming Conventions

### Categories

Available categories for core methods: `combining`, `conditional`, `connectable`, `context`, `creation`, `error-handling`, `filtering`, `math`, `sink`, `transformation`, `utility`...

For plugins, categories are actually the name of the plugin. Each category or plugin must have its dedicated markdown file in `/docs/docs/operator/` or `/docs/docs/plugins/` directories.

### Helper Names
- Follow Go naming conventions (PascalCase for exported)
- Use descriptive names that clearly indicate purpose
- For function variants, use consistent suffixes:
  - `F` suffix for function-based versions (lazy evaluation)
  - `I` suffix for variants having `index int` argument in predicate callback
  - `WithContext` suffix when context.Context is provided
  - `X` suffix for helpers with varying arguments (eg: MustX: Must2, Must3, Must4...)

### Description and examples

Be concise and descriptive, for explain what the method does. Also describe variants.

We need at least 1 example, but more example is good, especially to describe edge cases. You can find existing code examples in xxxx_example_test.go files.

Don't be too descriptive: obvious examples don't need to be repeated. Example: for the `FromSlice` operator, we don't need example for empty slice.

## Go Playground Examples

Every helper must have a working Go Playground example:
1. Create a minimal, self-contained example
2. Use realistic but simple data
3. Include the expected result as a comment
4. Test the example to ensure it works

When creating the go playground example, please run it to be sure it compiles and returns the expected output. If invalid, loop until it works.

Add these examples in the source code comments, on top of methods, with a syntax like `// Play: <url>`.

If the documentation is created at the same time of the helper source code, then the Go playground execution might fail, since we need to merge+release the source code first to make this new helper available to Go playground compiler. In that case, skip the creation of the example and set no URL.

## Validation Scripts

Run these scripts to validate your documentation:

```bash
# Check cross-references
node scripts/check-cross-references.js

# Check filename matches frontmatter
node scripts/check-filename-matches-frontmatter.js

# Check for similar existing helpers
node scripts/check-similar-exists.js

# Check for similar keys in directory
node scripts/check-similar-keys-exist-in-directory.js
```

## Examples: Complete Files

```yaml
---
name: All
slug: all
sourceRef: operator_conditional.go#L24
type: core
category: conditional
signatures:
  - "func All[T any](predicate func(T) bool)"
  - "func AllWithContext[T any](predicate func(context.Context, T) bool)"
  - "func AllI[T any](predicate func(T, int) bool)"
  - "func AllIWithContext[T any](predicate func(context.Context, T, int) bool)"
playUrl: https://go.dev/play/p/EXAMPLE
variantHelpers:
  - core#conditional#all
  - core#conditional#allwithcontext
  - core#conditional#alli
  - core#conditional#alliwithcontext
similarHelpers: []
position: 0
---

Determines whether all elements of an observable sequence satisfy a condition.

```go
obs := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5),
    ro.All(func(i int) bool {
        return i > 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```

## With index and context

```go
obs := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5),
    ro.AllIWithContext(func(ctx context.Context, n int, index int64) bool {
        return n > 0
    }),
)

sub := obs.Subscribe(ro.NewObserver[string](
    func(value string) {
        fmt.Printf("Next: %s\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: true
// Completed
```
```

## Checklist

Before submitting:

- [ ] Frontmatter is complete and correctly formatted
- [ ] Filename matches slug (with `core-` or `plugin-` prefix)
- [ ] Source reference points to correct line number
- [ ] Type and category are appropriate
- [ ] All signatures are included and properly formatted
- [ ] Go Playground example works and demonstrates usage
- [ ] Expected output is shown as a comment
- [ ] Similar helpers are listed if applicable
- [ ] Related helpers are consolidated into single file when appropriate
- [ ] All validation scripts pass without errors
- [ ] Helper is added to llms.txt

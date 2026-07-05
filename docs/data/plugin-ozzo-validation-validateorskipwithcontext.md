---
name: ValidateOrSkipWithContext
slug: validateorskipwithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L161
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrSkipWithContext[T any](rules ...ozzo.Rule)"
playUrl: https://go.dev/play/p/I1RNFYzHvS8
variantHelpers:
  - plugin#ozzo-validation#validateorskipwithcontext
similarHelpers:
  - plugin#ozzo-validation#validateorskip
  - plugin#ozzo-validation#validatestructorskipwithcontext
position: 10
---

Validates values with rules using context propagation, skipping items that fail validation and forwarding valid items unchanged.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

obs := ro.Pipe1(
    ro.Just("hello", "", "world", "x-too-long-string"),
    roozzo.ValidateOrSkipWithContext[string](
        validation.Required,
        validation.Length(1, 10),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello
// Next: world
// Completed
```
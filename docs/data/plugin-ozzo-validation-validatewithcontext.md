---
name: ValidateWithContext
slug: validatewithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L63
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateWithContext[T any](rules ...ozzo.Rule)"
playUrl: https://go.dev/play/p/uKrqQrP6TAD
variantHelpers:
  - plugin#ozzo-validation#validatewithcontext
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validatestructwithcontext
  - plugin#ozzo-validation#validateorerrorwithcontext
position: 10
---

Validates values with rules using context propagation, emitting a `Result[T]` for each item.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

obs := ro.Pipe1(
    ro.Just("hello", "", "world"),
    roozzo.ValidateWithContext[string](
        validation.Required,
        validation.Length(1, 10),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[roozzo.Result[string]]())
defer sub.Unsubscribe()

// Next: {false hello <nil>}
// Next: {true  {validation_required cannot be blank map[]}}
// Next: {false world <nil>}
// Completed
```
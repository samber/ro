---
name: Validate
slug: validate
sourceRef: plugins/ozzo/operator.go#L32
type: plugin
category: ozzo-validation
signatures:
  - "func Validate[T any](rules ...ozzo.Rule)"
playUrl: https://go.dev/play/p/6kgTXIAhCt0
variantHelpers:
  - plugin#ozzo-validation#validate
similarHelpers:
  - plugin#ozzo-validation#validatestruct
  - plugin#ozzo-validation#validateorerror
position: 0
---

Validates values with rules, emitting a `Result[T]` that is either ok (valid) or err (invalid).

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

obs := ro.Pipe1(
    ro.Just("hello", "", "world"),
    roozzo.Validate[string](
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
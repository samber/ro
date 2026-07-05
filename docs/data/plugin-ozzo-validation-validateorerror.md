---
name: ValidateOrError
slug: validateorerror
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L92
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrError[T any](rules ...ozzo.Rule)"
playUrl: https://go.dev/play/p/uRAnClXZKQF
variantHelpers:
  - plugin#ozzo-validation#validateorerror
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validateorerrorwithcontext
position: 80
---

Validates values with rules. Valid items are forwarded unchanged; the first invalid item terminates the stream with an error.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

obs := ro.Pipe1(
    ro.Just("hello", "", "world"),
    roozzo.ValidateOrError[string](
        validation.Required,
        validation.Length(1, 10),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello
// Error: cannot be blank
```
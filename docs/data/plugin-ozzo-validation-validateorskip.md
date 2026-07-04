---
name: ValidateOrSkip
slug: validateorskip
sourceRef: plugins/ozzo/operator.go#L115
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrSkip[T any](rules ...ozzo.Rule)"
playUrl: https://go.dev/play/p/vPj3PElqahA
variantHelpers:
  - plugin#ozzo-validation#validateorskip
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validatestructorskip
position: 8
---

Validates values with rules, skipping items that fail validation and forwarding valid items unchanged.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

obs := ro.Pipe1(
    ro.Just("hello", "", "world", "x-too-long-string"),
    roozzo.ValidateOrSkip[string](
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
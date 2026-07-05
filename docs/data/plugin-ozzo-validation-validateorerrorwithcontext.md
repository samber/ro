---
name: ValidateOrErrorWithContext
slug: validateorerrorwithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L101
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrErrorWithContext[T any](rules ...ozzo.Rule)"
playUrl: https://go.dev/play/p/vLdYa6s0cTQ
variantHelpers:
  - plugin#ozzo-validation#validateorerrorwithcontext
similarHelpers:
  - plugin#ozzo-validation#validateorerror
  - plugin#ozzo-validation#validateorskipwithcontext
position: 82
---

Validates each item using context-aware ozzo-validation rules. Items that fail validation cause the stream to error; valid items are forwarded unchanged.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

obs := ro.Pipe[int, int](
    ro.Just(5, 15, 25),
    roozzo.ValidateOrErrorWithContext[int](
        validation.Min(10),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error: must be no less than 10
```

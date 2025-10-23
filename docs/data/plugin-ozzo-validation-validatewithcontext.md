---
name: ValidateWithContext
slug: validatewithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L57
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateWithContext[T any](rules ...ozzo.Rule)"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validatewithcontext
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validatestructwithcontext
  - plugin#ozzo-validation#validateorerrorwithcontext
position: 10
---

Validates values with rules using context.

```go
import (
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo-validation"
    "github.com/go-ozzo/ozzo-validation/v4"
)

type User struct {
    Name string
    Age  int
}

obs := ro.Pipe[User, Result[User]](
    ro.Just(User{Name: "Alice", Age: 30}),
    roozzo.ValidateWithContext[User](
        ozzo.Rule{Name: "name", Required: true},
        ozzo.Rule{Name: "age", Required: true, Min: 18},
    ),
)

sub := obs.Subscribe(ro.PrintObserver[Result[User]]())
defer sub.Unsubscribe()

// Next: {Value: {Name: "Alice", Age: 30}, Error: nil}
// Completed
```
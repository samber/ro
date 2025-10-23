---
name: Validate
slug: validate
sourceRef: plugins/ozzo/operator.go#L32
type: plugin
category: ozzo-validation
signatures:
  - "func Validate[T any](rules ...ozzo.Rule)"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validate
similarHelpers:
  - plugin#ozzo-validation#validatestruct
  - plugin#ozzo-validation#validateorerror
position: 0
---

Validates values with rules.

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

obs := ro.Pipe[User, roozzovalidation.Result[User]](
    ro.Just(User{Name: "Alice", Age: 30}),
    roozzo.Validate[User](
        validation.Rule{Name: "name", Required: true},
        validation.Rule{Name: "age", Required: true, Min: 18},
    ),
)

sub := obs.Subscribe(ro.PrintObserver[roozzovalidation.Result[User]]())
defer sub.Unsubscribe()

// Next: {Value: {Name: "Alice", Age: 30}, Error: nil}
// Completed
```
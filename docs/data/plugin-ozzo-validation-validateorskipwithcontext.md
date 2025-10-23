---
name: ValidateOrSkipWithContext
slug: validateorskipwithcontext
sourceRef: plugins/ozzo/operator.go#L131
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrSkipWithContext[T any](rules ...ozzo.Rule)"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validateorskipwithcontext
similarHelpers:
  - plugin#ozzo-validation#validateorskip
  - plugin#ozzo-validation#validatestructorskipwithcontext
position: 10
---

Validates observable values with context and skips invalid ones.

```go
import (
    "context"

    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo-validation"
    "github.com/go-ozzo/ozzo-validation/v4"
)

type User struct {
    Name string
    Age  int
}

obs := ro.Pipe[User, User](
    ro.Just(
        User{Name: "Alice", Age: 30}, // valid
        User{Name: "", Age: 15},      // invalid
        User{Name: "Bob", Age: 25},   // valid
    ),
    roozzo.ValidateOrSkipWithContext[User](
        validation.Rule{Name: "name", Required: true},
        validation.Rule{Name: "age", Required: true, Min: 18},
    ),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Name: "Alice", Age: 30}
// Next: {Name: "Bob", Age: 25}
// Completed (invalid entry skipped with context support)
```
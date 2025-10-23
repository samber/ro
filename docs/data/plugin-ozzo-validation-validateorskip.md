---
name: ValidateOrSkip
slug: validateorskip
sourceRef: plugins/ozzo/operator.go#L115
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrSkip[T any](rules ...ozzo.Rule)"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validateorskip
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validatestructorskip
position: 8
---

Validates observable values and skips invalid ones.

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

obs := ro.Pipe[User, User](
    ro.Just(
        User{Name: "Alice", Age: 30}, // valid
        User{Name: "", Age: 15},      // invalid (empty name, too young)
        User{Name: "Bob", Age: 25},   // valid
    ),
    roozzo.ValidateOrSkip[User](
        validation.Rule{Name: "name", Required: true},
        validation.Rule{Name: "age", Required: true, Min: 18},
    ),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Name: "Alice", Age: 30}
// Next: {Name: "Bob", Age: 25}
// Completed (invalid entry skipped)
```
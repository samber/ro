---
name: ValidateOrError
slug: validateorerror
sourceRef: plugins/ozzo/operator.go#L82
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateOrError[T any](rules ...ozzo.Rule)"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validateorerror
similarHelpers:
  - plugin#ozzo-validation#validate
position: 80
---

Validates or emits error.

```go
import (
    "fmt"
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
        User{Name: "Alice", Age: 30},  // valid
        User{Name: "", Age: 15},        // invalid
    ),
    roozzo.ValidateOrError[User](
        ozzo.Rule{Name: "name", Required: true},
        ozzo.Rule{Name: "age", Required: true, Min: 18},
    ),
)

sub := obs.Subscribe(
    ro.NewObserver(
        func(user User) { fmt.Printf("Valid: %+v\n", user) },
        func(err error) { fmt.Printf("Error: %v\n", err) },
        func() { fmt.Println("Completed") },
    ),
)
defer sub.Unsubscribe()

// Valid: {Name:Alice Age:30}
// Error: validation failed
// Completed
```
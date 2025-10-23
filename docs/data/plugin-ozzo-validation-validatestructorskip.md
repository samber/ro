---
name: ValidateStructOrSkip
slug: validatestructorskip
sourceRef: plugins/ozzo/operator.go#L123
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructOrSkip[T any]()"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validatestructorskip
similarHelpers:
  - plugin#ozzo-validation#validateorskip
  - plugin#ozzo-validation#validatestruct
position: 9
---

Validates struct observables and skips invalid ones.

```go
import (
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo-validation"
    "github.com/go-ozzo/ozzo-validation/v4"
)

type User struct {
    Name string `validate:"required"`
    Age  int    `validate:"required,min=18"`
}

func (u User) Validate() error {
    return validation.ValidateStruct(&u,
        validation.Field(&u.Name, validation.Required),
        validation.Field(&u.Age, validation.Required, validation.Min(18)),
    )
}

obs := ro.Pipe[User, User](
    ro.Just(
        User{Name: "Alice", Age: 30}, // valid
        User{Name: "", Age: 15},      // invalid
        User{Name: "Bob", Age: 25},   // valid
    ),
    roozzo.ValidateStructOrSkip[User](),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Name: "Alice", Age: 30}
// Next: {Name: "Bob", Age: 25}
// Completed (invalid entry skipped)
```
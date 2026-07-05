---
name: ValidateStructOrSkip
slug: validatestructorskip
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L147
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructOrSkip[T any]()"
playUrl: https://go.dev/play/p/9tuhes1uG2q
variantHelpers:
  - plugin#ozzo-validation#validatestructorskip
similarHelpers:
  - plugin#ozzo-validation#validateorskip
  - plugin#ozzo-validation#validatestruct
position: 9
---

Validates struct items using the `ozzo.Validatable` interface, skipping items that fail validation and forwarding valid items unchanged.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

type User struct {
    Name string
    Age  int
}

func (u User) Validate() error {
    return validation.ValidateStruct(&u,
        validation.Field(&u.Name, validation.Required),
        validation.Field(&u.Age, validation.Min(18)),
    )
}

obs := ro.Pipe1(
    ro.Just(
        User{Name: "Alice", Age: 30},
        User{Name: "", Age: 15},
        User{Name: "Bob", Age: 25},
    ),
    roozzo.ValidateStructOrSkip[User](),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Alice 30}
// Next: {Bob 25}
// Completed
```
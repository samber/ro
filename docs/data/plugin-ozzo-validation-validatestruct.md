---
name: ValidateStruct
slug: validatestruct
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L46
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStruct[T any]()"
playUrl: https://go.dev/play/p/s-raE2CUPEx
variantHelpers:
  - plugin#ozzo-validation#validatestruct
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validatestructwithcontext
  - plugin#ozzo-validation#validatestructorerror
position: 20
---

Validates struct values that implement the `ozzo.Validatable` interface, emitting a `Result[T]` for each item.

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
    ),
    roozzo.ValidateStruct[User](),
)

sub := obs.Subscribe(ro.PrintObserver[roozzo.Result[User]]())
defer sub.Unsubscribe()

// Next: {false {Alice 30} <nil>}
// Next: {true { 0} map[Age:{validation_min_greater_equal_than_required must be no less than {{.threshold}} map[threshold:18]} Name:{validation_required cannot be blank map[]}]}
// Completed
```
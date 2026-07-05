---
name: ValidateStructWithContext
slug: validatestructwithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L75
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructWithContext[T any]()"
playUrl: https://go.dev/play/p/bhu5ThhGqaN
variantHelpers:
  - plugin#ozzo-validation#validatestructwithcontext
similarHelpers:
  - plugin#ozzo-validation#validatestruct
  - plugin#ozzo-validation#validatewithcontext
  - plugin#ozzo-validation#validatestructorerrorwithcontext
position: 30
---

Validates struct values that implement the `ozzo.ValidatableWithContext` interface using context propagation.

```go
import (
    "context"

    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

type User struct {
    Name string
    Age  int
}

func (u User) ValidateWithContext(ctx context.Context) error {
    return validation.ValidateStructWithContext(ctx, &u,
        validation.Field(&u.Name, validation.Required),
        validation.Field(&u.Age, validation.Min(18)),
    )
}

obs := ro.Pipe1(
    ro.Just(
        User{Name: "Alice", Age: 30},
        User{Name: "", Age: 15},
    ),
    roozzo.ValidateStructWithContext[User](),
)

sub := obs.Subscribe(ro.PrintObserver[roozzo.Result[User]]())
defer sub.Unsubscribe()

// Next: {false {Alice 30} <nil>}
// Next: {true { 0} map[Age:{validation_min_greater_equal_than_required must be no less than {{.threshold}} map[threshold:18]} Name:{validation_required cannot be blank map[]}]}
// Completed
```
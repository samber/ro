---
name: ValidateStruct
slug: validatestruct
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L42
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStruct[T any]()"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validatestruct
similarHelpers:
  - plugin#ozzo-validation#validate
  - plugin#ozzo-validation#validatestructwithcontext
  - plugin#ozzo-validation#validatestructorerror
position: 20
---

Validates struct values that implement ozzo.Validatable interface.

```go
import (
    "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo-validation"
)

type User struct {
    Name string `validate:"required"`
    Age  int    `validate:"required,min=18"`
}

func (u User) Validate() error {
    return ozzo.ValidateStruct(&u,
        ozzo.Field(&u.Name, ozzo.Required),
        ozzo.Field(&u.Age, ozzo.Required, ozzo.Min(18)),
    )
}

obs := ro.Pipe[User, Result[User]](
    ro.Just(User{Name: "Alice", Age: 30}),
    roozzo.ValidateStruct[User](),
)

sub := obs.Subscribe(ro.PrintObserver[Result[User]]())
defer sub.Unsubscribe()

// Next: {Value: {Name: "Alice", Age: 30}, Error: nil}
// Completed
```
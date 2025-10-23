---
name: ValidateStructWithContext
slug: validatestructwithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L67
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructWithContext[T any]()"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validatestructwithcontext
similarHelpers:
  - plugin#ozzo-validation#validatestruct
  - plugin#ozzo-validation#validatewithcontext
  - plugin#ozzo-validation#validatestructorerrorwithcontext
position: 30
---

Validates struct values that implement ozzo.ValidatableWithContext interface using context.

```go
import (
    "context"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo-validation"
    "github.com/go-ozzo/ozzo-validation/v4"
)

type User struct {
    Name string `validate:"required"`
    Age  int    `validate:"required,min=18"`
}

func (u User) ValidateWithContext(ctx context.Context) error {
    return ozzo.ValidateStructWithContext(ctx, &u,
        ozzo.Field(&u.Name, ozzo.Required),
        ozzo.Field(&u.Age, ozzo.Required, ozzo.Min(18)),
    )
}

obs := ro.Pipe[User, Result[User]](
    ro.Just(User{Name: "Alice", Age: 30}),
    roozzo.ValidateStructWithContext[User](),
)

sub := obs.Subscribe(ro.PrintObserver[Result[User]]())
defer sub.Unsubscribe()

// Next: {Value: {Name: "Alice", Age: 30}, Error: nil}
// Completed
```
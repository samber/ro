---
name: ValidateStructOrSkipWithContext
slug: validatestructorskipwithcontext
sourceRef: plugins/ozzo/operator.go#L139
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructOrSkipWithContext[T any]()"
playUrl: https://go.dev/play/p/7KjElY-ytIH
variantHelpers:
  - plugin#ozzo-validation#validatestructorskipwithcontext
similarHelpers:
  - plugin#ozzo-validation#validatestructorskip
  - plugin#ozzo-validation#validateorskipwithcontext
position: 11
---

Validates struct items using the `ozzo.ValidatableWithContext` interface with context propagation, skipping items that fail validation and forwarding valid items unchanged.

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
        User{Name: "Bob", Age: 25},
    ),
    roozzo.ValidateStructOrSkipWithContext[User](),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Alice 30}
// Next: {Bob 25}
// Completed
```
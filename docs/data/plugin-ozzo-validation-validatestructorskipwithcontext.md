---
name: ValidateStructOrSkipWithContext
slug: validatestructorskipwithcontext
sourceRef: plugins/ozzo/operator.go#L139
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructOrSkipWithContext[T any]()"
playUrl: ""
variantHelpers:
  - plugin#ozzo-validation#validatestructorskipwithcontext
similarHelpers:
  - plugin#ozzo-validation#validatestructorskip
  - plugin#ozzo-validation#validateorskipwithcontext
position: 11
---

Validates struct observables with context and skips invalid ones.

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
    return validation.ValidateStructWithContext(ctx, &u,
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
    roozzo.ValidateStructOrSkipWithContext[User](),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Name: "Alice", Age: 30}
// Next: {Name: "Bob", Age: 25}
// Completed (invalid entry skipped with context support)
```
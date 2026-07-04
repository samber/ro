---
name: ValidateStructOrErrorWithContext
slug: validatestructorerrorwithcontext
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L108
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructOrErrorWithContext[T any]()"
playUrl: https://go.dev/play/p/iKmYApNt7Cm
variantHelpers:
  - plugin#ozzo-validation#validatestructorerrorwithcontext
similarHelpers:
  - plugin#ozzo-validation#validatestructorerror
  - plugin#ozzo-validation#validatestructorskipwithcontext
  - plugin#ozzo-validation#validateorerrorwithcontext
position: 92
---

Validates each struct item using context-aware ozzo-validation. Items that fail validation cause the stream to error; valid items are forwarded unchanged. The struct must implement `validation.ValidatableWithContext`.

```go
import (
    "context"

    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

type User struct {
    Name  string
    Email string
}

func (u User) ValidateWithContext(ctx context.Context) error {
    return validation.ValidateStructWithContext(ctx, &u,
        validation.Field(&u.Name, validation.Required),
        validation.Field(&u.Email, validation.Required),
    )
}

obs := ro.Pipe[User, User](
    ro.Just(
        User{Name: "Alice", Email: "alice@example.com"},
        User{Name: "", Email: ""},
    ),
    roozzo.ValidateStructOrErrorWithContext[User](),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Alice alice@example.com}
// Error: Email: cannot be blank; Name: cannot be blank.
```

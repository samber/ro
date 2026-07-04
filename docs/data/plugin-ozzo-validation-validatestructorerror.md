---
name: ValidateStructOrError
slug: validatestructorerror
sourceRef: plugins/ozzo/ozzo-validation/operator.go#L89
type: plugin
category: ozzo-validation
signatures:
  - "func ValidateStructOrError[T any]()"
playUrl: https://go.dev/play/p/FTmqek_eT7e
variantHelpers:
  - plugin#ozzo-validation#validatestructorerror
similarHelpers:
  - plugin#ozzo-validation#validatestruct
  - plugin#ozzo-validation#validatestructorskip
  - plugin#ozzo-validation#validateorerror
position: 91
---

Validates each struct item using ozzo-validation. Items that fail validation cause the stream to error; valid items are forwarded unchanged. The struct must implement `validation.Validatable`.

```go
import (
    validation "github.com/go-ozzo/ozzo-validation/v4"
    "github.com/samber/ro"
    roozzo "github.com/samber/ro/plugins/ozzo/ozzo-validation"
)

type User struct {
    Name  string
    Email string
}

func (u User) Validate() error {
    return validation.ValidateStruct(&u,
        validation.Field(&u.Name, validation.Required),
        validation.Field(&u.Email, validation.Required),
    )
}

obs := ro.Pipe[User, User](
    ro.Just(
        User{Name: "Alice", Email: "alice@example.com"},
        User{Name: "", Email: ""},
    ),
    roozzo.ValidateStructOrError[User](),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Alice alice@example.com}
// Error: Email: cannot be blank; Name: cannot be blank.
```

---
name: Fake
slug: fake
sourceRef: plugins/faker/operator.go#L96
type: plugin
category: faker
signatures:
  - "func Fake[T any](count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#fake
similarHelpers:
  - plugin#faker#generate
  - plugin#faker#generaten
position: 20
---

Emits `count` values of type `T` filled with random data by go-faker, driven by the `faker` struct tags of `T`. A tag error, such as an unknown tag, is forwarded as an error notification.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

type User struct {
    Name  string `faker:"name"`
    Email string `faker:"email"`
    Age   int    `faker:"boundary_start=18, boundary_end=65"`
}

obs := rofaker.Fake[User](2)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {Alice Johnson alice.johnson@example.org 42}
// Next: {Bob Smith bsmith@example.com 27}
// Completed
```

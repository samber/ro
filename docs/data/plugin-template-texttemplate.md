---
name: TextTemplate
slug: texttemplate
sourceRef: plugins/template/operator.go#L26
type: plugin
category: template
signatures:
  - "func TextTemplate[T any](template string)"
playUrl: ""
variantHelpers:
  - plugin#template#texttemplate
similarHelpers:
  - plugin#template#htmltemplate
position: 0
---

Applies text template to values.

```go
import (
    "github.com/samber/ro"
    rotemplate "github.com/samber/ro/plugins/template"
)

type User struct {
    Name string
    Age  int
}

obs := ro.Pipe(
    ro.Just(User{Name: "Alice", Age: 30}),
    rotemplate.TextTemplate[User]("Hello {{.Name}}, you are {{.Age}} years old"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Hello Alice, you are 30 years old
// Completed
```
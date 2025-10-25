---
name: HTMLTemplate
slug: htmltemplate
sourceRef: plugins/template/operator.go#L36
type: plugin
category: template
signatures:
  - "func HTMLTemplate[T any](template string)"
playUrl: https://go.dev/play/p/aKQUYjcte-Z
variantHelpers:
  - plugin#template#htmltemplate
similarHelpers:
  - plugin#template#texttemplate
position: 10
---

Applies HTML template to values.

```go
import (
    "github.com/samber/ro"
    rotemplate "github.com/samber/ro/plugins/template"
)

type Item struct {
    Name  string
    Price float64
}

obs := ro.Pipe[Item, string](
    ro.Just(Item{Name: "Apple", Price: 1.99}),
    rotemplate.HTMLTemplate[Item]("<h1>{{.Name}}</h1><p>Price: ${{.Price}}</p>"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: <h1>Apple</h1><p>Price: $1.99</p>
// Completed
```
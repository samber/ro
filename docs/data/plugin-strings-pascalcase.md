---
name: PascalCase
slug: pascalcase
sourceRef: plugins/strings/operator_pascalcase.go#L33
type: plugin
category: strings
signatures:
  - "func PascalCase[T ~string]()"
playUrl: ""
variantHelpers:
  - plugin#strings#pascalcase
similarHelpers:
  - plugin#strings#camelcase
  - plugin#strings#snakecase
  - plugin#strings#kebabcase
  - plugin#bytes#pascalcase
position: 5
---

Converts each string emitted by the source Observable to PascalCase (UpperCamelCase).

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("hello_world", "foo-bar", "some string"),
    rostrings.PascalCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: HelloWorld
// Next: FooBar
// Next: SomeString
// Completed
```

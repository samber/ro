---
name: CamelCase
slug: camelcase
sourceRef: plugins/strings/operator_camelcase.go#L54
type: plugin
category: strings
signatures:
  - "func CamelCase[T ~string]()"
  - "func CamelCaseWithLanguage[T ~string](tag language.Tag)"
playUrl: https://go.dev/play/p/65rbW1kFxhF
variantHelpers:
  - plugin#strings#camelcase
  - plugin#strings#camelcasewithlanguage
similarHelpers:
  - plugin#bytes#camelcase
position: 0
---

Converts string to camel case.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("hello_world_world"),
    rostrings.CamelCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: helloWorldWorld
// Completed
```

### CamelCaseWithLanguage

Converts the string to camel case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
    "golang.org/x/text/language"
)

obs := ro.Pipe[string, string](
    ro.Just("Istanbul city"),
    rostrings.CamelCaseWithLanguage[string](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: ıstanbulCity
// Completed
```
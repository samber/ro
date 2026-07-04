---
name: Capitalize
slug: capitalize
sourceRef: plugins/strings/operator_capitalize.go#L29
type: plugin
category: strings
signatures:
  - "func Capitalize[T ~string]()"
  - "func CapitalizeWithLanguage[T ~string](tag language.Tag)"
playUrl: https://go.dev/play/p/Q9lZAav_ETm
variantHelpers:
  - plugin#strings#capitalize
  - plugin#strings#capitalizewithlanguage
similarHelpers:
  - plugin#bytes#capitalize
position: 10
---

Capitalizes first letter of string.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("hello world"),
    rostrings.Capitalize[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Hello world
// Completed
```

### CapitalizeWithLanguage

Capitalizes the first letter using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
    "golang.org/x/text/language"
)

obs := ro.Pipe[string, string](
    ro.Just("istanbul"),
    rostrings.CapitalizeWithLanguage[string](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: İstanbul
// Completed
```
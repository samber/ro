---
name: SnakeCase
slug: snakecase
sourceRef: plugins/strings/operator_snakecase.go#L47
type: plugin
category: strings
signatures:
  - "func SnakeCase[T ~string]()"
  - "func SnakeCaseWithLanguage[T ~string](tag language.Tag)"
playUrl: https://go.dev/play/p/zHCGH586_X3
variantHelpers:
  - plugin#strings#snakecase
  - plugin#strings#snakecasewithlanguage
similarHelpers:
  - plugin#bytes#snakecase
position: 40
---

Converts string to snake case.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("HelloWorldWorld"),
    rostrings.SnakeCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello_world_world
// Completed
```

### SnakeCaseWithLanguage

Converts the string to snake case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
    "golang.org/x/text/language"
)

obs := ro.Pipe[string, string](
    ro.Just("IstanbulCity"),
    rostrings.SnakeCaseWithLanguage[string](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: ıstanbul_city
// Completed
```
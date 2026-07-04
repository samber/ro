---
name: SnakeCase
slug: snakecase
sourceRef: plugins/bytes/operator_snakecase.go#L33
type: plugin
category: bytes
signatures:
  - "func SnakeCase[T ~[]byte]()"
  - "func SnakeCaseWithLanguage[T ~[]byte](tag language.Tag)"
playUrl: https://go.dev/play/p/JSlMzOne811
variantHelpers:
  - plugin#bytes#snakecase
  - plugin#bytes#snakecasewithlanguage
similarHelpers:
  - plugin#strings#snakecase
position: 60
---

Converts the string to snake case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("HelloWorldWorld")),
    robytes.SnakeCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 95 119 111 114 108 100 95 119 111 114 108 100]
// Completed
```

### SnakeCaseWithLanguage

Converts the string to snake case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
    "golang.org/x/text/language"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("IstanbulCity")),
    robytes.SnakeCaseWithLanguage[[]byte](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [196 177 115 116 97 110 98 117 108 95 99 105 116 121]
// Completed
```
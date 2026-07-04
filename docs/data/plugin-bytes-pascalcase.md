---
name: PascalCase
slug: pascalcase
sourceRef: plugins/bytes/operator_pascalcase.go#L33
type: plugin
category: bytes
signatures:
  - "func PascalCase[T ~[]byte]()"
  - "func PascalCaseWithLanguage[T ~[]byte](tag language.Tag)"
playUrl: https://go.dev/play/p/ToPLTi_lCXI
variantHelpers:
  - plugin#bytes#pascalcase
  - plugin#bytes#pascalcasewithlanguage
similarHelpers:
  - plugin#strings#pascalcase
position: 40
---

Converts the string to pascal case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello_world_world")),
    robytes.PascalCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [72 101 108 108 111 87 111 114 108 100 87 111 114 108 100]
// Completed
```

### PascalCaseWithLanguage

Converts the string to pascal case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
    "golang.org/x/text/language"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("istanbul city")),
    robytes.PascalCaseWithLanguage[[]byte](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [196 176 115 116 97 110 98 117 108 67 105 116 121]
// Completed
```
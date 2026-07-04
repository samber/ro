---
name: Capitalize
slug: capitalize
sourceRef: plugins/bytes/operator_capitalize.go#L29
type: plugin
category: bytes
signatures:
  - "func Capitalize[T ~[]byte]()"
  - "func CapitalizeWithLanguage[T ~[]byte](tag language.Tag)"
playUrl: https://go.dev/play/p/qc7UDCtJM0n
variantHelpers:
  - plugin#bytes#capitalize
  - plugin#bytes#capitalizewithlanguage
similarHelpers:
  - plugin#strings#capitalize
position: 20
---

Capitalizes the first letter of the string.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello world")),
    robytes.Capitalize[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [72 101 108 108 111 32 119 111 114 108 100]
// Completed
```

### CapitalizeWithLanguage

Capitalizes the first letter using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
    "golang.org/x/text/language"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("istanbul")),
    robytes.CapitalizeWithLanguage[[]byte](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [196 176 115 116 97 110 98 117 108]
// Completed
```
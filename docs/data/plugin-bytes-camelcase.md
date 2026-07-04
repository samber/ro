---
name: CamelCase
slug: camelcase
sourceRef: plugins/bytes/operator_camelcase.go#L37
type: plugin
category: bytes
signatures:
  - "func CamelCase[T ~[]byte]()"
  - "func CamelCaseWithLanguage[T ~[]byte](tag language.Tag)"
playUrl: https://go.dev/play/p/RCL_Z45aIQC
variantHelpers:
  - plugin#bytes#camelcase
  - plugin#bytes#camelcasewithlanguage
similarHelpers:
  - plugin#strings#camelcase
position: 0
---

Converts the string to camel case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello_world_world")),
    robytes.CamelCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 87 111 114 108 100 87 111 114 108 100]
// Completed
```

### CamelCaseWithLanguage

Converts the string to camel case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
    "golang.org/x/text/language"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("Istanbul city")),
    robytes.CamelCaseWithLanguage[[]byte](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [196 177 115 116 97 110 98 117 108 67 105 116 121]
// Completed
```
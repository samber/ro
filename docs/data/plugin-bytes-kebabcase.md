---
name: KebabCase
slug: kebabcase
sourceRef: plugins/bytes/operator_kebabcase.go#L47
type: plugin
category: bytes
signatures:
  - "func KebabCase[T ~[]byte]()"
  - "func KebabCaseWithLanguage[T ~[]byte](tag language.Tag)"
playUrl: https://go.dev/play/p/86V3xKuLykG
variantHelpers:
  - plugin#bytes#kebabcase
  - plugin#bytes#kebabcasewithlanguage
similarHelpers: []
position: 10
---

Converts the string to kebab case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("HelloWorldWorld")),
    robytes.KebabCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 45 119 111 114 108 100 45 119 111 114 108 100]
// Completed
```

### KebabCaseWithLanguage

Converts the string to kebab case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
    "golang.org/x/text/language"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("IstanbulCity")),
    robytes.KebabCaseWithLanguage[[]byte](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [196 177 115 116 97 110 98 117 108 45 99 105 116 121]
// Completed
```

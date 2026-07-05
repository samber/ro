---
name: KebabCase
slug: kebabcase
sourceRef: plugins/strings/operator_kebabcase.go#L47
type: plugin
category: strings
signatures:
  - "func KebabCase[T ~string]()"
  - "func KebabCaseWithLanguage[T ~string](tag language.Tag)"
playUrl: https://go.dev/play/p/yAbSRKFl4pS
variantHelpers:
  - plugin#strings#kebabcase
  - plugin#strings#kebabcasewithlanguage
similarHelpers:
  - plugin#strings#snakecase
  - plugin#strings#camelcase
position: 20
---

Converts string to kebab case.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("HelloWorldTest"),
    rostrings.KebabCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello-world-test
// Completed
```

### KebabCaseWithLanguage

Converts the string to kebab case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
    "golang.org/x/text/language"
)

obs := ro.Pipe[string, string](
    ro.Just("IstanbulCity"),
    rostrings.KebabCaseWithLanguage[string](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: ıstanbul-city
// Completed
```
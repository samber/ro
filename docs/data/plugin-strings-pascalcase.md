---
name: PascalCase
slug: pascalcase
sourceRef: plugins/strings/operator_pascalcase.go#L47
type: plugin
category: strings
signatures:
  - "func PascalCase[T ~string]()"
  - "func PascalCaseWithLanguage[T ~string](tag language.Tag)"
playUrl: https://go.dev/play/p/107SvPGvHAK
variantHelpers:
  - plugin#strings#pascalcase
  - plugin#strings#pascalcasewithlanguage
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

### PascalCaseWithLanguage

Converts the string to pascal case using locale-aware casing.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
    "golang.org/x/text/language"
)

obs := ro.Pipe[string, string](
    ro.Just("istanbul city"),
    rostrings.PascalCaseWithLanguage[string](language.Turkish),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: İstanbulCity
// Completed
```

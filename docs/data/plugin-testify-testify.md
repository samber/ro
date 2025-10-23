---
name: Testify
slug: testify
sourceRef: plugins/testify/testify.go#L42
type: plugin
category: testify
signatures:
  - "func Testify[T any](is *assert.Assertions)"
playUrl: ""
variantHelpers:
  - plugin#testify#testify
similarHelpers: []
position: 0
---

Creates test assertions for observables.

```go
import (
    "testing"

    "github.com/samber/ro"
    rotestify "github.com/samber/ro/plugins/testify"
    "github.com/stretchr/testify/assert"
)

func TestMyObservable(t *testing.T) {
    is := assert.New(t)

    obs := ro.Pipe[int, int](
        ro.Just(1, 2, 3),
        ro.Map(func(x int) int { return x * 2 }),
    )

    spec := rotestify.Testify[int](is)
    obs.Subscribe(spec.Expect(2, 4, 6))

    // Test will pass if values match expected
}
```
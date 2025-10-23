---
name: Random
slug: random
sourceRef: plugins/bytes/operator_random.go#L89
type: plugin
category: bytes
signatures:
  - "func Random[T any](size int, charset []rune)"
playUrl: ""
variantHelpers:
  - plugin#bytes#random
similarHelpers:
  - plugin#strings#random
position: 50
---

Generates a random string of specified size using charset.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[int, []byte](
    ro.Just(1, 2, 3),
    robytes.Random[int](10, []rune("abcdefghijklmnopqrstuvwxyz")),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [some random bytes]
// Next: [some random bytes]
// Next: [some random bytes]
// Completed
```
---
name: CountDistinct
slug: countdistinct
sourceRef: plugins/hyperloglog/operator.go#L28
type: plugin
category: hyperloglog
signatures:
  - "func CountDistinct[T comparable](precision uint8, sparse bool, hashFunc func(input T) uint64)"
playUrl: ""
variantHelpers:
  - plugin#hyperloglog#countdistinct
similarHelpers:
  - plugin#hyperloglog#countdistinctreduce
position: 0
---

Estimates the number of distinct items in a stream using HyperLogLog algorithm.

```go
import (
    "github.com/samber/ro"
    rohyperloglog "github.com/samber/ro/plugins/hyperloglog"
    "github.com/cloudfoundry/gosigar"
)

obs := ro.Pipe[string, uint64](
    ro.Just(
        "apple", "banana", "apple", "orange", "banana",
        "grape", "apple", "kiwi", "orange", "mango",
    ),
    rohyperloglog.CountDistinct(14, false, func(s string) uint64 {
        return gosigar.Sum64(s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[uint64]())
defer sub.Unsubscribe()

// Next: 6
// Completed
```
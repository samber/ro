---
name: CountDistinctReduce
slug: countdistinctreduce
sourceRef: plugins/hyperloglog/operator.go#L57
type: plugin
category: hyperloglog
signatures:
  - "func CountDistinctReduce[T comparable](precision uint8, sparse bool, hashFunc func(input T) uint64)"
playUrl: https://go.dev/play/p/GrfnG0rq4Rq
variantHelpers:
  - plugin#hyperloglog#countdistinctreduce
similarHelpers:
  - plugin#hyperloglog#countdistinct
position: 10
---

Emits running distinct count estimates for each item in the stream.

```go
import (
	"hash/fnv"
    "github.com/samber/ro"
    rohyperloglog "github.com/samber/ro/plugins/hyperloglog"
)

obs := ro.Pipe[string, uint64](
    ro.Just(
        "apple", "banana", "apple", "orange", "banana",
        "grape", "apple", "kiwi", "orange", "mango",
    ),
    rohyperloglog.CountDistinctReduce(14, false, func(s string) uint64 {
        h := fnv.New64a()
        h.Write([]byte(s))
        return h.Sum64()
    }),
)

sub := obs.Subscribe(ro.PrintObserver[uint64]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 2
// Next: 3
// Next: 3
// Next: 4
// Next: 4
// Next: 5
// Next: 5
// Next: 6
// Completed
```
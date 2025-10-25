---
name: NewCSVWriter
slug: newcsvwriter
sourceRef: plugins/encoding/csv/sink.go#L25
type: plugin
category: encoding-csv
signatures:
  - "func NewCSVWriter(writer *csv.Writer)"
playUrl: https://go.dev/play/p/Mz_EgW5bBT7
variantHelpers:
  - plugin#encoding-csv#newcsvwriter
similarHelpers: []
position: 10
---

Creates an operator that writes string arrays to a CSV writer and returns the count of written records.

```go
import (
    "bytes"
    "encoding/csv"

    "github.com/samber/ro"
    rocsv "github.com/samber/ro/plugins/encoding/csv"
)

var buf bytes.Buffer
writer := csv.NewWriter(&buf)

obs := ro.Pipe[[]string, int](
    ro.Just(
        []string{"name", "age", "city"},
        []string{"Alice", "30", "New York"},
        []string{"Bob", "25", "Los Angeles"},
        []string{"Charlie", "35", "Chicago"},
    ),
    rocsv.NewCSVWriter(writer),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Completed
```
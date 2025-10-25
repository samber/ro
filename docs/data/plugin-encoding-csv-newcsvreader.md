---
name: NewCSVReader
slug: newcsvreader
sourceRef: plugins/encoding/csv/source.go#L26
type: plugin
category: encoding-csv
signatures:
  - "func NewCSVReader(reader *csv.Reader)"
playUrl: https://go.dev/play/p/lmL054evzfS
variantHelpers:
  - plugin#encoding-csv#newcsvreader
similarHelpers: []
position: 0
---

Creates an observable that reads records from a CSV reader.

```go
import (
    "encoding/csv"
    "strings"

    "github.com/samber/ro"
    rocsv "github.com/samber/ro/plugins/encoding/csv"
)

csvData := `name,age,city
Alice,30,New York
Bob,25,Los Angeles
Charlie,35,Chicago`

reader := csv.NewReader(strings.NewReader(csvData))
obs := rocsv.NewCSVReader(reader)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Next: [name age city]
// Next: [Alice 30 New York]
// Next: [Bob 25 Los Angeles]
// Next: [Charlie 35 Chicago]
// Completed
```
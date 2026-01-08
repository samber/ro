module github.com/samber/ro/examples/sql-to-csv

go 1.18

require github.com/samber/lo v1.51.0

require github.com/samber/ro v0.0.0

require github.com/samber/ro/plugins/encoding/csv v0.0.0

require (
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace github.com/samber/ro => ../..

replace github.com/samber/ro/plugins/encoding/csv => ../../plugins/encoding/csv

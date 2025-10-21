module github.com/samber/ro/examples/ics-to-csv

go 1.20

require github.com/samber/lo v1.52.0 // indirect

require (
	github.com/arran4/golang-ical v0.3.2
	github.com/samber/ro v0.0.0
	github.com/samber/ro/plugins/encoding/csv v0.0.0
	github.com/samber/ro/plugins/ics v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/sort v0.0.0-00010101000000-000000000000
)

require (
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace (
	github.com/samber/ro => ../..
	github.com/samber/ro/plugins/encoding/csv => ../../plugins/encoding/csv
	github.com/samber/ro/plugins/ics => ../../plugins/ics
	github.com/samber/ro/plugins/sort => ../../plugins/sort
)

module github.com/samber/ro/examples/ee-prometheus

go 1.25.0

require github.com/samber/lo v1.53.0

require github.com/samber/ro v0.0.0

require (
	github.com/prometheus/client_golang v1.24.1
	github.com/samber/ro/ee v0.0.0
	github.com/samber/ro/ee/plugins/prometheus v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/encoding/csv v0.0.0-00010101000000-000000000000
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.70.1 // indirect
	github.com/prometheus/procfs v0.21.1 // indirect
	golang.org/x/exp v0.0.0-20240613232115-7f521ea00fb8 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace (
	github.com/samber/ro => ../..
	github.com/samber/ro/ee => ../../ee
	github.com/samber/ro/ee/plugins/prometheus => ../../ee/plugins/prometheus
	github.com/samber/ro/plugins/encoding/csv => ../../plugins/encoding/csv
)

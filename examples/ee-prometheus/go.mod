module github.com/samber/ro/examples/ee-prometheus

go 1.18

require github.com/samber/lo v1.51.0

require github.com/samber/ro v0.0.0

require (
	github.com/prometheus/client_golang v1.16.0
	github.com/samber/ro/ee v0.0.0
	github.com/samber/ro/ee/plugins/prometheus v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/encoding/csv v0.0.0-00010101000000-000000000000
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/golang/protobuf v1.5.3 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.4 // indirect
	github.com/prometheus/client_model v0.6.1 // indirect
	github.com/prometheus/common v0.44.0 // indirect
	github.com/prometheus/procfs v0.15.1 // indirect
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/sys v0.22.0 // indirect
	golang.org/x/text v0.22.0 // indirect
	google.golang.org/protobuf v1.34.2 // indirect
)

replace (
	github.com/samber/ro => ../..
	github.com/samber/ro/ee => ../../ee
	github.com/samber/ro/ee/plugins/prometheus => ../../ee/plugins/prometheus
	github.com/samber/ro/plugins/encoding/csv => ../../plugins/encoding/csv
)

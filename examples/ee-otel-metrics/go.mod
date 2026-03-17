module github.com/samber/ro/examples/ee-otel-metrics

go 1.23.0

toolchain go1.24

require (
	github.com/prometheus/client_golang v1.18.0
	github.com/samber/lo v1.53.0
	github.com/samber/ro v0.0.0
	github.com/samber/ro/ee v0.0.0
	github.com/samber/ro/ee/plugins/otel v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/encoding/csv v0.0.0-00010101000000-000000000000
	go.opentelemetry.io/otel v1.37.0
	go.opentelemetry.io/otel/exporters/prometheus v0.46.0
	go.opentelemetry.io/otel/sdk v1.37.0
	go.opentelemetry.io/otel/sdk/metric v1.37.0
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/matttproud/golang_protobuf_extensions/v2 v2.0.0 // indirect
	github.com/prometheus/client_model v0.6.0 // indirect
	github.com/prometheus/common v0.45.0 // indirect
	github.com/prometheus/procfs v0.12.0 // indirect
	go.opentelemetry.io/auto/sdk v1.1.0 // indirect
	go.opentelemetry.io/otel/log v0.13.0 // indirect
	go.opentelemetry.io/otel/metric v1.37.0 // indirect
	go.opentelemetry.io/otel/trace v1.37.0 // indirect
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/sys v0.33.0 // indirect
	golang.org/x/text v0.22.0 // indirect
	google.golang.org/protobuf v1.33.0 // indirect
)

replace (
	github.com/samber/ro => ../..
	github.com/samber/ro/ee => ../../ee
	github.com/samber/ro/ee/plugins/otel => ../../ee/plugins/otel
	github.com/samber/ro/plugins/encoding/csv => ../../plugins/encoding/csv
)

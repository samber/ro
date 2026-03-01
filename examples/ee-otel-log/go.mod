module github.com/samber/ro/examples/ee-otel-log

go 1.24.0

toolchain go1.24.2

require (
	github.com/samber/lo v1.52.0
	github.com/samber/ro v0.0.0
	github.com/samber/ro/ee v0.0.0
	github.com/samber/ro/ee/plugins/otel v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/encoding/csv v0.0.0
	go.opentelemetry.io/otel v1.40.0
	go.opentelemetry.io/otel/exporters/stdout/stdoutlog v0.13.0
	go.opentelemetry.io/otel/log v0.13.0
	go.opentelemetry.io/otel/sdk v1.40.0
	go.opentelemetry.io/otel/sdk/log v0.13.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/metric v1.40.0 // indirect
	go.opentelemetry.io/otel/trace v1.40.0 // indirect
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/sys v0.40.0 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace (
	github.com/samber/ro => ../../
	github.com/samber/ro/ee => ../../ee
	github.com/samber/ro/ee/plugins/otel => ../../ee/plugins/otel
	github.com/samber/ro/plugins/encoding/csv => ../../plugins/encoding/csv
)

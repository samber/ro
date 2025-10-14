module github.com/samber/ro/ee/plugins/otel

go 1.23.0

toolchain go1.24

require (
	github.com/samber/ro v0.0.0
	github.com/samber/ro/ee v0.0.0
	go.opentelemetry.io/otel v1.37.0
	go.opentelemetry.io/otel/log v0.13.0
	go.opentelemetry.io/otel/metric v1.37.0
	go.opentelemetry.io/otel/trace v1.37.0
)

require (
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/samber/lo v1.51.0 // indirect
	go.opentelemetry.io/auto/sdk v1.1.0 // indirect
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace (
	github.com/samber/ro => ../../..
	github.com/samber/ro/ee => ../..
)

module github.com/samber/ro/plugins/cron

go 1.24.0

require (
	github.com/go-co-op/gocron/v2 v2.21.2
	github.com/samber/ro v0.0.0
	github.com/stretchr/testify v1.12.1
)

require (
	github.com/google/uuid v1.6.0 // indirect
	github.com/jonboulle/clockwork v0.5.0 // indirect
	github.com/robfig/cron/v3 v3.0.1 // indirect
	github.com/samber/lo v1.53.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/exp v0.0.0-20240613232115-7f521ea00fb8 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace github.com/samber/ro => ../..

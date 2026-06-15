module github.com/samber/ro/examples/distributed-websocket-gateway

go 1.24

require github.com/samber/lo v1.53.0

require github.com/samber/ro v0.0.0

require (
	github.com/gorilla/websocket v1.5.3
	github.com/redis/go-redis/v9 v9.20.1
	github.com/samber/ro/plugins/signal v0.0.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	go.uber.org/atomic v1.11.0 // indirect
	golang.org/x/exp v0.0.0-20240613232115-7f521ea00fb8 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace github.com/samber/ro => ../..

replace github.com/samber/ro/plugins/signal => ../../plugins/signal

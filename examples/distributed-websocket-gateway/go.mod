module github.com/samber/ro/examples/distributed-websocket-gateway

go 1.23.0

require github.com/samber/lo v1.52.0

require github.com/samber/ro v0.0.0

require (
	github.com/gorilla/websocket v1.5.3
	github.com/redis/go-redis/v9 v9.7.3
	github.com/samber/ro/plugins/signal v0.0.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	golang.org/x/exp v0.0.0-20240613232115-7f521ea00fb8 // indirect
	golang.org/x/text v0.28.0 // indirect
)

replace github.com/samber/ro => ../..

replace github.com/samber/ro/plugins/signal => ../../plugins/signal

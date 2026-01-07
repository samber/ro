module github.com/samber/ro/examples/stock-price-enrichment

go 1.23.0

require (
	github.com/samber/ro v0.0.0
	github.com/samber/ro/plugins/stdio v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/websocket/client v0.0.0
)

require (
	github.com/gorilla/websocket v1.5.3 // indirect
	github.com/samber/lo v1.52.0 // indirec
	golang.org/x/exp v0.0.0-20240613232115-7f521ea00fb8 // indirect
	golang.org/x/text v0.28.0 // indirect
)

replace (
	github.com/samber/ro => ../..
	github.com/samber/ro/plugins/stdio => ../../plugins/stdio
	github.com/samber/ro/plugins/websocket/client => ../../plugins/websocket/client
)

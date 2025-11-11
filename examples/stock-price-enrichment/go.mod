module github.com/samber/ro/examples/stock-price-enrichment

go 1.18

require (
	github.com/samber/ro v0.0.0
	github.com/samber/ro/plugins/stdio v0.0.0-00010101000000-000000000000
	github.com/samber/ro/plugins/websocket/client v0.0.0
)

require (
	github.com/gorilla/websocket v1.5.1 // indirect
	github.com/samber/lo v1.51.0 // indirec
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	golang.org/x/net v0.17.0 // indirect
	golang.org/x/text v0.16.0 // indirect
)

replace (
	github.com/samber/ro => ../..
	github.com/samber/ro/plugins/stdio => ../../plugins/stdio
	github.com/samber/ro/plugins/websocket/client => ../../plugins/websocket/client
)

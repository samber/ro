// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/samber/lo"
	"github.com/samber/ro"
	rostdio "github.com/samber/ro/plugins/stdio"
	rowebsocketclient "github.com/samber/ro/plugins/websocket/client"
)

type Subscription struct {
	Type   string `json:"type"`
	Symbol string `json:"symbol"`
}

type TickerData struct {
	Type string            `json:"type"`
	Data []*TickerDataItem `json:"data"`
}

type TickerDataItem struct {
	Symbol    string  `json:"s"`
	Price     float64 `json:"p"`
	Timestamp int64   `json:"t"`
	Volume    float64 `json:"v"`
}

type Minute struct {
	Symbol    string
	Time      time.Time
	LastPrice float64
	Volume    float64

	High60s float64
	Low60s  float64
}

var downstreamPipeline = ro.PipeOp[*TickerData, *Minute](
	// Prepare the input data
	ro.Map(func(value *TickerData) []*TickerDataItem {
		return value.Data
	}),
	// Flatten the items from []*TickerDataItem to *TickerDataItem
	ro.Flatten[*TickerDataItem](),
	// Split the stream into a stream of streams, one for each symbol
	ro.GroupBy(func(value *TickerDataItem) string {
		return value.Symbol
	}),
	ro.Map(func(symbolObservable ro.Observable[*TickerDataItem]) ro.Observable[*Minute] {
		windowPeriod := 1 * time.Minute

		// The first window will end at the end of the current minute, not in 1 minute.
		// It permits to align window start with the exact clock minutes.
		now := time.Now()
		firstWindow := windowPeriod - now.Sub(now.Truncate(windowPeriod))

		// Process the data for each symbol
		return ro.Pipe[*TickerDataItem, *Minute](
			symbolObservable,
			// Buffer 1min of data, to measure the high, low, last price and volume.
			ro.BufferWhen[*TickerDataItem](ro.IntervalWithInitial(firstWindow, windowPeriod)),
			ro.Map(func(values []*TickerDataItem) *Minute {
				// Measure the high, low, last price and volume
				timestamp := time.UnixMilli(values[0].Timestamp).Truncate(time.Minute)
				maxPrice := lo.MaxBy(values, func(a, b *TickerDataItem) bool {
					return a.Price > b.Price
				})
				minPrice := lo.MinBy(values, func(a, b *TickerDataItem) bool {
					return a.Price < b.Price
				})
				lastPrice := lo.MaxBy(values, func(a, b *TickerDataItem) bool {
					return a.Timestamp > b.Timestamp
				})
				volume := lo.SumBy(values, func(a *TickerDataItem) float64 {
					return a.Volume
				})

				// Return 1min aggregated values for the symbol
				return &Minute{
					Symbol:    values[0].Symbol,
					Time:      timestamp,
					LastPrice: lastPrice.Price,
					Volume:    volume,
					High60s:   maxPrice.Price,
					Low60s:    minPrice.Price,
				}
			}),
		)
	}),
	// Merge all the symbols into a single stream
	ro.MergeAll[*Minute](),
	// Adjust precision
	ro.Map(func(value *Minute) *Minute {
		truncate := func(value float64) float64 {
			switch {
			case value < 1:
				return math.Round(value*100000) / 100000
			case value < 10:
				return math.Round(value*10000) / 10000
			case value < 100:
				return math.Round(value*1000) / 1000
			case value < 1000:
				return math.Round(value*100) / 100
			case value < 10000:
				return math.Round(value*10) / 10
			default:
				return math.Round(value)
			}
		}
		value.LastPrice = truncate(value.LastPrice)
		value.High60s = truncate(value.High60s)
		value.Low60s = truncate(value.Low60s)
		value.Volume = truncate(value.Volume)
		return value
	}),
)

func main() {
	apiKey := os.Getenv("FINNHUB_API_KEY")
	if apiKey == "" {
		log.Fatal("FINNHUB_API_KEY is not set")
	}

	wsSubject := rowebsocketclient.NewWebsocketSubject(rowebsocketclient.WebsocketSubjectConfig[Subscription, *TickerData]{
		URL: "wss://ws.finnhub.io?token=" + apiKey,
		Serializer: func(value Subscription) ([]byte, error) {
			return json.Marshal(value)
		},
		Deserializer: func(data []byte) (*TickerData, error) {
			var d TickerData
			return &d, json.Unmarshal(data, &d)
		},
	})

	downstream := downstreamPipeline(wsSubject.AsObservable())
	upstream := ro.Pipe1(
		rostdio.NewPrompt("Subscribe to a stock symbol: "),
		ro.Map(func(value []byte) Subscription {
			return Subscription{
				Type:   "subscribe",
				Symbol: string(value),
			}
		}),
	)

	sub1 := downstream.Subscribe(ro.OnNext(func(value *Minute) {
		fmt.Printf("[%s] -> Last:%g Low:%g High:%g Volume:%g\n", value.Symbol, value.LastPrice, value.Low60s, value.High60s, value.Volume)
	}))
	defer sub1.Unsubscribe()

	sub2 := upstream.Subscribe(wsSubject)
	defer sub2.Unsubscribe()
}

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
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strings"

	ics "github.com/arran4/golang-ical"
	"github.com/samber/lo"
	"github.com/samber/ro"
	rocsv "github.com/samber/ro/plugins/encoding/csv"
	roics "github.com/samber/ro/plugins/ics"
	rosort "github.com/samber/ro/plugins/sort"
)

type Event struct {
	Summary      string
	Description  string
	Participants []string
	StartTime    string
	EndTime      string
}

// Define a pipeline to query users from a database and write them to stdout as csv.
var pipeline = ro.PipeOp6(
	// Retry on parsing error.
	ro.RetryWithConfig[*ics.VEvent](ro.RetryConfig{
		MaxRetries: 2,
	}),
	// Convert ics.VEvent to Event.
	ro.Map(func(event *ics.VEvent) *Event {
		participants := []string{}
		if org := event.GetProperty(ics.ComponentPropertyOrganizer); org != nil {
			participants = append(participants, org.Value)
		}
		if att := event.GetProperty(ics.ComponentPropertyAttendee); att != nil {
			participants = append(participants, strings.Split(att.Value, ",")...)
		}
		for i := range participants {
			participants[i] = strings.TrimSpace(participants[i])
			participants[i] = strings.TrimPrefix(participants[i], "mailto:")
		}
		sort.Strings(participants)
		participants = lo.Uniq(participants)

		startTime := ""
		endTime := ""
		if start := event.GetProperty(ics.ComponentPropertyDtStart); start != nil {
			startTime = start.Value
		}
		if end := event.GetProperty(ics.ComponentPropertyDtEnd); end != nil {
			endTime = end.Value
		}

		return &Event{
			Summary:      lo.FromPtr(event.GetProperty(ics.ComponentPropertySummary)).Value,
			Description:  lo.FromPtr(event.GetProperty(ics.ComponentPropertyDescription)).Value,
			Participants: participants,
			StartTime:    startTime,
			EndTime:      endTime,
		}
	}),
	// Sort events by start time.
	rosort.SortFunc(func(a, b *Event) int {
		return strings.Compare(a.StartTime, b.StartTime)
	}),
	// Convert Event to csv row.
	ro.Map(func(event *Event) []string {
		return []string{
			event.Summary,
			// event.Description,
			event.StartTime,
			event.EndTime,
			strings.Join(event.Participants, ","),
		}
	}),
	ro.DistinctBy(func(cols []string) string {
		// for very large files, we should use a custom DistinctBy implementation that
		// a bloom filter or a hash table to limit the memory footprint.
		return strings.Join(cols, ",")
	}),
	// Write events to stdout as csv.
	rocsv.NewCSVWriter(
		csv.NewWriter(os.Stdout),
	),
)

// go run main.go ../../plugins/ics/testdata/fr-public-holidays-*.ics
func main() {
	paths := os.Args[2:]

	var reader ro.Observable[*ics.VEvent]
	if len(paths) > 0 && strings.HasPrefix(paths[0], "https://") {
		reader = roics.NewICSURLReader(paths...)
	} else {
		reader = roics.NewICSFileReader(paths...)
	}

	subscription := pipeline(reader).
		Subscribe(
			ro.OnError[int](func(err error) {
				fmt.Println(err)
			}),
		)

	// Optional, since the pipeline will complete itself, in a blocking way.
	defer subscription.Unsubscribe()
}

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

package roics

import (
	"context"
	"os"

	ics "github.com/arran4/golang-ical"
	"github.com/samber/ro"
)

// NewICSFileReader reads events from one or more ICS files.
// @TODO: add glob support
func NewICSFileReader(paths ...string) ro.Observable[*ics.VEvent] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[*ics.VEvent]) ro.Teardown {
		for _, path := range paths {
			reader, err := os.Open(path)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				break
			}

			defer reader.Close()

			events, err := ics.ParseCalendar(reader)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				break
			}

			for _, event := range events.Events() {
				destination.NextWithContext(ctx, event)
			}
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// NewICSURLReader reads events from one or more ICS URLs.
func NewICSURLReader(urls ...string) ro.Observable[*ics.VEvent] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[*ics.VEvent]) ro.Teardown {
		for _, url := range urls {
			events, err := ics.ParseCalendarFromUrl(url)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				break
			}

			for _, event := range events.Events() {
				destination.NextWithContext(ctx, event)
			}
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

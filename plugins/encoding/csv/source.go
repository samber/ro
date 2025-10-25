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


package rocsv

import (
	"context"
	"encoding/csv"
	"io"

	"github.com/samber/ro"
)

// NewCSVReader creates an observable that reads CSV records from a csv.Reader.
// Play: https://go.dev/play/p/ZB3apy60Ujv
func NewCSVReader(reader *csv.Reader) ro.Observable[[]string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[[]string]) ro.Teardown {
		for {
			records, err := reader.Read()
			if err != nil {
				if err == io.EOF {
					destination.CompleteWithContext(ctx)
				} else {
					destination.ErrorWithContext(ctx, err)
				}
				break
			}
			destination.NextWithContext(ctx, records)
		}

		return nil
	})
}

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

package rohttpclient

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/samber/ro"
)

// HTTPRequest sends a http request and returns the response. It's a pull-based operator.
//
// A http status code >= 400 is not considered an error.
//
// Don't forget to call resp.Body.Close() when you're done with the response.
func HTTPRequest(req *http.Request, client *http.Client) ro.Observable[*http.Response] {
	if client == nil {
		client = http.DefaultClient
	}

	return ro.NewSafeObservableWithContext(func(_ context.Context, destination ro.Observer[*http.Response]) ro.Teardown {
		ctx, cancel := context.WithCancel(req.Context())

		go func() {
			req = req.WithContext(ctx)

			res, err := client.Do(req)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				return
			}

			destination.NextWithContext(ctx, res)
			destination.CompleteWithContext(ctx)
		}()

		return (func())(cancel)
	})
}

func HTTPRequestJSON[T any](req *http.Request, client *http.Client) ro.Observable[T] {
	return ro.MapErr(func(res *http.Response) (T, error) {
		defer res.Body.Close()

		var t T
		err := json.NewDecoder(res.Body).Decode(&t)
		return t, err
	})(HTTPRequest(req, client))
}

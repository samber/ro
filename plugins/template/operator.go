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


package rotemplate

import (
	"bytes"
	htmltemplate "html/template"
	texttemplate "text/template"

	"github.com/samber/ro"
)

// TextTemplate creates a text template operator that renders templates with input data.
// Play: https://go.dev/play/p/06cCGj34vLo
func TextTemplate[T any](template string) func(ro.Observable[T]) ro.Observable[string] {
	tpl := texttemplate.Must(texttemplate.New(template).Parse(template))

	return ro.MapErr(func(v T) (string, error) {
		var buf bytes.Buffer
		err := tpl.Execute(&buf, v)
		return buf.String(), err
	})
}

// HTMLTemplate creates an HTML template operator that renders templates with input data.
// Play: https://go.dev/play/p/emlON8wyaXx
func HTMLTemplate[T any](template string) func(ro.Observable[T]) ro.Observable[string] {
	tpl := htmltemplate.Must(htmltemplate.New(template).Parse(template))

	return ro.MapErr(func(v T) (string, error) {
		var buf bytes.Buffer
		err := tpl.Execute(&buf, v)
		return buf.String(), err
	})
}

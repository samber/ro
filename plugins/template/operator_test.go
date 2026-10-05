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
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

type testStruct struct {
	Name  string `json:"name"`
	Age   int    `json:"age"`
	Email string `json:"email,omitempty"`
}

type nestedStruct struct {
	ID       int         `json:"id"`
	Data     testStruct  `json:"data"`
	Metadata interface{} `json:"metadata,omitempty"`
}

func TestTextTemplate(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	t.Run("simple string template", func(t *testing.T) {
		input := []string{"Alice", "Bob", "Charlie"}
		template := "Hello, {{.}}!"
		expected := []string{"Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[string](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("struct template", func(t *testing.T) {
		input := []testStruct{
			{Name: "Alice", Age: 30, Email: "alice@example.com"},
			{Name: "Bob", Age: 25},
			{Name: "Charlie", Age: 35, Email: "charlie@example.com"},
		}
		template := "{{.Name}} is {{.Age}} years old{{if .Email}} ({{.Email}}){{end}}"
		expected := []string{
			"Alice is 30 years old (alice@example.com)",
			"Bob is 25 years old",
			"Charlie is 35 years old (charlie@example.com)",
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[testStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("nested struct template", func(t *testing.T) {
		input := []nestedStruct{
			{
				ID: 1,
				Data: testStruct{
					Name: "Alice", Age: 30, Email: "alice@example.com",
				},
				Metadata: map[string]interface{}{"role": "admin"},
			},
			{
				ID: 2,
				Data: testStruct{
					Name: "Bob", Age: 25,
				},
			},
		}
		template := "ID: {{.ID}}, Name: {{.Data.Name}}, Age: {{.Data.Age}}{{if .Metadata}} ({{.Metadata.role}}){{end}}"
		expected := []string{
			"ID: 1, Name: Alice, Age: 30 (admin)",
			"ID: 2, Name: Bob, Age: 25",
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[nestedStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("number template", func(t *testing.T) {
		input := []int{1, 2, 3, 42}
		template := "Number: {{.}}"
		expected := []string{"Number: 1", "Number: 2", "Number: 3", "Number: 42"}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[int](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("empty input", func(t *testing.T) {
		template := "Hello, {{.}}!"
		expected := []string{}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[string](),
				TextTemplate[string](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("complex template with conditionals", func(t *testing.T) {
		input := []testStruct{
			{Name: "Alice", Age: 30, Email: "alice@example.com"},
			{Name: "Bob", Age: 17},
			{Name: "Charlie", Age: 35, Email: "charlie@example.com"},
		}
		template := `{{.Name}} ({{.Age}} years old){{if .Email}}
Email: {{.Email}}{{end}}{{if gt .Age 18}}
Status: Adult{{else}}
Status: Minor{{end}}`
		expected := []string{
			"Alice (30 years old)\nEmail: alice@example.com\nStatus: Adult",
			"Bob (17 years old)\nStatus: Minor",
			"Charlie (35 years old)\nEmail: charlie@example.com\nStatus: Adult",
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[testStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("template with range", func(t *testing.T) {
		type listStruct struct {
			Name  string   `json:"name"`
			Items []string `json:"items"`
		}

		input := []listStruct{
			{Name: "Fruits", Items: []string{"apple", "banana", "orange"}},
			{Name: "Colors", Items: []string{"red", "blue"}},
		}
		template := `{{.Name}}:
{{range .Items}}- {{.}}
{{end}}`
		expected := []string{
			"Fruits:\n- apple\n- banana\n- orange\n",
			"Colors:\n- red\n- blue\n",
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[listStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})
}

func TestHTMLTemplate(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	t.Run("simple HTML template", func(t *testing.T) {
		input := []string{"Alice", "Bob", "Charlie"}
		template := "<h1>Hello, {{.}}!</h1>"
		expected := []string{"<h1>Hello, Alice!</h1>", "<h1>Hello, Bob!</h1>", "<h1>Hello, Charlie!</h1>"}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				HTMLTemplate[string](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("HTML template with struct", func(t *testing.T) {
		input := []testStruct{
			{Name: "Alice", Age: 30, Email: "alice@example.com"},
			{Name: "Bob", Age: 25},
			{Name: "Charlie", Age: 35, Email: "charlie@example.com"},
		}
		template := `<div class="user">
  <h2>{{.Name}}</h2>
  <p>Age: {{.Age}}</p>
  {{if .Email}}<p>Email: <a href="mailto:{{.Email}}">{{.Email}}</a></p>{{end}}
</div>`
		expected := []string{
			`<div class="user">
  <h2>Alice</h2>
  <p>Age: 30</p>
  <p>Email: <a href="mailto:alice@example.com">alice@example.com</a></p>
</div>`,
			`<div class="user">
  <h2>Bob</h2>
  <p>Age: 25</p>
  
</div>`,
			`<div class="user">
  <h2>Charlie</h2>
  <p>Age: 35</p>
  <p>Email: <a href="mailto:charlie@example.com">charlie@example.com</a></p>
</div>`,
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				HTMLTemplate[testStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("HTML template with nested struct", func(t *testing.T) {
		input := []nestedStruct{
			{
				ID: 1,
				Data: testStruct{
					Name: "Alice", Age: 30, Email: "alice@example.com",
				},
				Metadata: map[string]interface{}{"role": "admin"},
			},
			{
				ID: 2,
				Data: testStruct{
					Name: "Bob", Age: 25,
				},
			},
		}
		template := `<div class="record">
  <span class="id">ID: {{.ID}}</span>
  <div class="data">
    <strong>{{.Data.Name}}</strong> ({{.Data.Age}} years old)
    {{if .Data.Email}}<br>Email: {{.Data.Email}}{{end}}
  </div>
  {{if .Metadata}}<div class="metadata">Role: {{.Metadata.role}}</div>{{end}}
</div>`
		expected := []string{
			`<div class="record">
  <span class="id">ID: 1</span>
  <div class="data">
    <strong>Alice</strong> (30 years old)
    <br>Email: alice@example.com
  </div>
  <div class="metadata">Role: admin</div>
</div>`,
			`<div class="record">
  <span class="id">ID: 2</span>
  <div class="data">
    <strong>Bob</strong> (25 years old)
    
  </div>
  
</div>`,
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				HTMLTemplate[nestedStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("HTML template with list", func(t *testing.T) {
		type listStruct struct {
			Title string   `json:"title"`
			Items []string `json:"items"`
		}

		input := []listStruct{
			{Title: "Fruits", Items: []string{"apple", "banana", "orange"}},
			{Title: "Colors", Items: []string{"red", "blue"}},
		}
		template := `<div class="list">
  <h3>{{.Title}}</h3>
  <ul>
    {{range .Items}}<li>{{.}}</li>{{end}}
  </ul>
</div>`
		expected := []string{
			`<div class="list">
  <h3>Fruits</h3>
  <ul>
    <li>apple</li><li>banana</li><li>orange</li>
  </ul>
</div>`,
			`<div class="list">
  <h3>Colors</h3>
  <ul>
    <li>red</li><li>blue</li>
  </ul>
</div>`,
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				HTMLTemplate[listStruct](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("empty input", func(t *testing.T) {
		template := "<h1>Hello, {{.}}!</h1>"
		expected := []string{}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[string](),
				HTMLTemplate[string](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("HTML template with special characters", func(t *testing.T) {
		input := []string{"<script>alert('xss')</script>", "& < > \" '"}
		template := "<div>{{.}}</div>"
		expected := []string{
			"<div>&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;</div>",
			"<div>&amp; &lt; &gt; &#34; &#39;</div>",
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				HTMLTemplate[string](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})
}

func TestTemplate_errorHandling(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	t.Run("invalid template syntax", func(t *testing.T) {
		input := []string{"test"}
		template := "{{.Name" // Missing closing brace

		is.Panics(func() {
			_, _ = ro.Collect(
				ro.Pipe1(
					ro.FromSlice(input),
					TextTemplate[string](template),
				),
			)
		})
	})

	t.Run("template with missing field", func(t *testing.T) {
		input := []testStruct{
			{Name: "Alice", Age: 30},
		}
		template := "{{.Name}} is {{.Age}} years old and works at {{.Company}}" // Company field doesn't exist

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[testStruct](template),
			),
		)

		// Should handle missing fields gracefully
		is.Empty(values)
		is.NotNil(err)
	})
}

func TestTemplate_withDifferentTypes(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	t.Run("float template", func(t *testing.T) {
		input := []float64{3.14, 2.718, 1.618}
		template := "Value: {{.}}"
		expected := []string{"Value: 3.14", "Value: 2.718", "Value: 1.618"}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[float64](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("bool template", func(t *testing.T) {
		input := []bool{true, false, true}
		template := "Status: {{.}}"
		expected := []string{"Status: true", "Status: false", "Status: true"}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[bool](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})

	t.Run("map template", func(t *testing.T) {
		input := []map[string]interface{}{
			{"name": "Alice", "age": 30},
			{"name": "Bob", "age": 25, "city": "NYC"},
		}
		template := "{{.name}} is {{.age}} years old{{if .city}} from {{.city}}{{end}}"
		expected := []string{
			"Alice is 30 years old",
			"Bob is 25 years old from NYC",
		}

		values, err := ro.Collect(
			ro.Pipe1(
				ro.FromSlice(input),
				TextTemplate[map[string]interface{}](template),
			),
		)

		is.Equal(expected, values)
		is.Nil(err)
	})
}

# Template Plugin

The template plugin provides operators for processing data with Go's text and HTML templates.

## Installation

```bash
go get github.com/samber/ro/plugins/template
```

## Operators

### TextTemplate

Processes data with text templates using Go's `text/template` package.

```go
import (
    "github.com/samber/ro"
    rotemplate "github.com/samber/ro/plugins/template"
)

type User struct {
    Name string
    Age  int
    City string
}

observable := ro.Pipe1(
    ro.Just(
        User{Name: "Alice", Age: 30, City: "New York"},
        User{Name: "Bob", Age: 25, City: "Los Angeles"},
        User{Name: "Charlie", Age: 35, City: "Chicago"},
    ),
    rotemplate.TextTemplate[User]("Hello {{.Name}}, you are {{.Age}} years old from {{.City}}."),
)

subscription := observable.Subscribe(ro.PrintObserver[string]())
defer subscription.Unsubscribe()

// Output:
// Next: Hello Alice, you are 30 years old from New York.
// Next: Hello Bob, you are 25 years old from Los Angeles.
// Next: Hello Charlie, you are 35 years old from Chicago.
// Completed
```

### HTMLTemplate

Processes data with HTML templates using Go's `html/template` package.

```go
observable := ro.Pipe1(
    ro.Just(
        User{Name: "Alice", Age: 30, City: "New York"},
        User{Name: "Bob", Age: 25, City: "Los Angeles"},
        User{Name: "Charlie", Age: 35, City: "Chicago"},
    ),
    rotemplate.HTMLTemplate[User](`<div class="user">
  <h2>{{.Name}}</h2>
  <p>Age: {{.Age}}</p>
  <p>City: {{.City}}</p>
</div>`),
)

subscription := observable.Subscribe(ro.PrintObserver[string]())
defer subscription.Unsubscribe()

// Output:
// Next: <div class="user">
//   <h2>Alice</h2>
//   <p>Age: 30</p>
//   <p>City: New York</p>
// </div>
// Next: <div class="user">
//   <h2>Bob</h2>
//   <p>Age: 25</p>
//   <p>City: Los Angeles</p>
// </div>
// Next: <div class="user">
//   <h2>Charlie</h2>
//   <p>Age: 35</p>
//   <p>City: Chicago</p>
// </div>
// Completed
```

## Supported Data Types

The template plugin supports various data types:

### Structs

```go
type Person struct {
    Name string
    Age  int
    Email string
}

observable := ro.Pipe1(
    ro.Just(Person{Name: "Alice", Age: 30, Email: "alice@example.com"}),
    rotemplate.TextTemplate[Person]("{{.Name}} ({{.Age}}) - {{.Email}}"),
)
```

### Simple Types

```go
// Strings
observable := ro.Pipe1(
    ro.Just("Alice", "Bob", "Charlie"),
    rotemplate.TextTemplate[string]("Hello {{.}}!"),
)

// Integers
observable := ro.Pipe1(
    ro.Just(1, 2, 3, 4, 5),
    rotemplate.TextTemplate[int]("Number: {{.}}"),
)
```

### Maps

```go
observable := ro.Pipe1(
    ro.Just(
        map[string]interface{}{"name": "Alice", "age": 30, "city": "New York"},
        map[string]interface{}{"name": "Bob", "age": 25, "city": "Los Angeles"},
    ),
    rotemplate.TextTemplate[map[string]interface{}]("User {{.name}} is {{.age}} years old from {{.city}}."),
)
```

## Template Features

### Conditionals

```go
type Person struct {
    Name string
    Age  int
}

observable := ro.Pipe1(
    ro.Just(
        Person{Name: "Alice", Age: 30},
        Person{Name: "Bob", Age: 17},
        Person{Name: "Charlie", Age: 25},
    ),
    rotemplate.TextTemplate[Person](`{{.Name}} is {{if ge .Age 18}}an adult{{else}}a minor{{end}} ({{.Age}} years old).`),
)

// Output:
// Next: Alice is an adult (30 years old).
// Next: Bob is a minor (17 years old).
// Next: Charlie is an adult (25 years old).
```

### Loops

```go
type Team struct {
    Name    string
    Members []string
}

observable := ro.Pipe1(
    ro.Just(
        Team{Name: "Alpha", Members: []string{"Alice", "Bob", "Charlie"}},
        Team{Name: "Beta", Members: []string{"David", "Eve"}},
    ),
    rotemplate.TextTemplate[Team](`Team {{.Name}}:
{{range .Members}}- {{.}}
{{end}}`),
)
```

### Functions

```go
observable := ro.Pipe1(
    ro.Just("hello world", "GOLANG PROGRAMMING"),
    rotemplate.TextTemplate[string](`Original: {{.}}
Upper: {{. | upper}}
Lower: {{. | lower}}`),
)
```

## Error Handling

Both `TextTemplate` and `HTMLTemplate` handle errors gracefully and will emit error notifications for template processing errors:

```go
observable := ro.Pipe1(
    ro.Just(
        User{Name: "Alice", Age: 30, City: "New York"},
        User{Name: "Bob", Age: 25, City: "Los Angeles"},
    ),
    rotemplate.TextTemplate[User]("Hello {{.Name}}, you are {{.Age}} years old from {{.InvalidField}}."),
)

subscription := observable.Subscribe(
    ro.NewObserver(
        func(value string) {
            // Handle successful template processing
        },
        func(err error) {
            // Handle template processing error
            // This could be due to:
            // - Invalid template syntax
            // - Missing template fields
            // - Template execution errors
        },
        func() {
            // Handle completion
        },
    ),
)
defer subscription.Unsubscribe()
```

## Real-world Example

Here's a practical example that generates HTML reports:

```go
import (
    "github.com/samber/ro"
    rotemplate "github.com/samber/ro/plugins/template"
)

type Report struct {
    Title   string
    Date    string
    Items   []string
    Summary string
}

// Generate HTML reports
pipeline := ro.Pipe2(
    // Simulate report data
    ro.Just(
        Report{
            Title:   "Monthly Report",
            Date:    "2024-01-15",
            Items:   []string{"Item 1", "Item 2", "Item 3"},
            Summary: "All items completed successfully",
        },
        Report{
            Title:   "Weekly Summary",
            Date:    "2024-01-12",
            Items:   []string{"Task A", "Task B"},
            Summary: "Tasks in progress",
        },
    ),
    // Generate HTML
    rotemplate.HTMLTemplate[Report](`<!DOCTYPE html>
<html>
<head>
    <title>{{.Title}}</title>
</head>
<body>
    <h1>{{.Title}}</h1>
    <p>Date: {{.Date}}</p>
    <h2>Items:</h2>
    <ul>
    {{range .Items}}
        <li>{{.}}</li>
    {{end}}
    </ul>
    <p><strong>Summary:</strong> {{.Summary}}</p>
</body>
</html>`),
)

subscription := pipeline.Subscribe(ro.PrintObserver[string]())
defer subscription.Unsubscribe()
```

## Performance Considerations

- The plugin uses Go's standard `text/template` and `html/template` packages
- Templates are compiled once and reused for all data items
- Error handling is built into both template operators
- HTML templates provide automatic escaping for security
- Consider template complexity for performance with large datasets
- Use appropriate template syntax for your use case
- Templates support all Go template features including functions, pipelines, and nested templates 
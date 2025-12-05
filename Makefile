MODULES=$(shell go list -m)

build:
	go build -v ${MODULES} ./...

test:
	go test -race ${MODULES} ./...
watch-test:
	reflex -t 50ms -s -- sh -c 'gotest -timeout 300s -race ${MODULES} ./...'

bench:
	go test -v -run=^Benchmark -benchmem -count 3 -bench ${MODULES} ./...
watch-bench:
	reflex -t 50ms -s -- sh -c 'go test -v -run=^Benchmark -benchmem -count 3 -bench ${MODULES} ./...'

coverage:
	go test -v -coverprofile=cover.out -covermode=atomic ${MODULES} ./...
	go tool cover -html=cover.out -o cover.html

tools:
	go install github.com/cespare/reflex@latest
	go install github.com/rakyll/gotest@latest
	go install github.com/psampaz/go-mod-outdated@latest
	go install github.com/jondot/goweight@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go get -t -u golang.org/x/tools/cmd/cover
	go install github.com/sonatype-nexus-community/nancy@latest
	go install golang.org/x/perf/cmd/benchstat@latest
	go install github.com/cespare/prettybench@latest
	go install github.com/samber/headercheck/cmd/headercheck@latest
	go mod tidy

	# brew install hougesen/tap/mdsf

lint:
	golangci-lint run --timeout 60s --max-same-issues 50 ./...
	headercheck --config ./licenses/headercheck.yaml .
	# mdsf verify --debug --log-level warn docs/
lint-fix:
	golangci-lint run --timeout 60s --max-same-issues 50 --fix ./...
	headercheck --config ./licenses/headercheck.yaml --fix .
	# mdsf format --debug --log-level warn docs/

audit:
	go mod tidy
	go list -json -m all | nancy sleuth

outdated:
	go mod tidy
	go list -u -m -json all | go-mod-outdated -update -direct

weight:
	goweight

doc:
	cd docs && npm install && npm start

bench-1m:
	@echo "Generating 1M fixture for examples/billion-rows-benchmark..."
	@mkdir -p examples/billion-rows-benchmark/fixtures
	@./examples/billion-rows-benchmark/scripts/expand_fixture.sh examples/billion-rows-benchmark/fixtures/sample.csv examples/billion-rows-benchmark/fixtures/1m.csv 1000000
	@echo "Running 1M benchmark (may take a while)..."
	FIXTURE_PATH=$(CURDIR)/examples/billion-rows-benchmark/fixtures/1m.csv go test -run=^$ -bench BenchmarkMillionRowChallenge -benchmem ./examples/billion-rows-benchmark

MODULES=$(shell go list -m)

build:
	go build -v ${MODULES} ./...
	@if [ -n "$(GOEXPERIMENT)" ]; then cd ./plugins/exp/simd && GOWORK=off GOEXPERIMENT=simd go build -v ./...; fi

# `go test` also runs every FuzzXxx target on its seeds (see `make fuzz`).
# TODO: once Go < 1.20 is no longer supported (the CI matrix starts at 1.18), skip them here with
# `-skip '^Fuzz'`: the -skip flag does not exist before Go 1.20. `make fuzz` then covers them.
test:
	go test -race ${MODULES} ./...
	@if [ -n "$(GOEXPERIMENT)" ]; then cd ./plugins/exp/simd && GOWORK=off GOEXPERIMENT=simd go test -race ./...; fi

# Runs every fuzz target on RO_FUZZ_ITERATIONS seeds (see internal/xfuzz). Example: `make fuzz RO_FUZZ_ITERATIONS=10000`.
RO_FUZZ_ITERATIONS ?= 1000

# Phony: the ./fuzz directory has the same name, so make would otherwise report "up to date".
.PHONY: fuzz
fuzz:
	RO_FUZZ_ITERATIONS=$(RO_FUZZ_ITERATIONS) go test -race -run=^Fuzz ${MODULES} ./...
	@if [ -n "$(GOEXPERIMENT)" ]; then cd ./plugins/exp/simd && RO_FUZZ_ITERATIONS=$(RO_FUZZ_ITERATIONS) GOWORK=off GOEXPERIMENT=simd go test -race -run=^Fuzz ./...; fi

watch-test:
	reflex -t 50ms -s -- sh -c 'gotest -timeout 300s -race ${MODULES} ./...'

# Phony: the ./bench directory has the same name, so make would otherwise report "up to date".
.PHONY: bench
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
	@if [ -n "$(GOEXPERIMENT)" ]; then cd ./plugins/exp/simd && GOWORK=off GOEXPERIMENT=simd golangci-lint run --timeout 60s --max-same-issues 50 ./...; fi
	headercheck --config ./licenses/headercheck.yaml .
	# mdsf verify --debug --log-level warn docs/
lint-fix:
	golangci-lint run --timeout 60s --max-same-issues 50 --fix ./...
	@if [ -n "$(GOEXPERIMENT)" ]; then cd ./plugins/exp/simd && GOWORK=off GOEXPERIMENT=simd golangci-lint run --timeout 60s --max-same-issues 50 --fix ./...; fi
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

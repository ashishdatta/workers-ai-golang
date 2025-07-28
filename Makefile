.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: lint
lint:
	@golangci-lint run

.PHONY: test
test: unit

.PHONY: unit
unit:
	@echo 'Running unit tests...'
	@mkdir -p .cover
	@GOFLAGS=$(GOFLAGS) go test -v -race -count=10 ./... \
		-coverprofile .cover/cover.out

GO := go
BINARY := sift
MODEL_DIR := models
MODEL_URL_BASE := https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main
ORT_VERSION := 1.24.2
ORT_URL := https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-x64-$(ORT_VERSION).tgz

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE ?= $(shell date -u +'%Y-%m-%d_%H:%M:%S' 2>/dev/null || echo "unknown")

LDFLAGS := -X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)

.PHONY: build clean test bench download-model download-ort

build:
	$(GO) build -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/sift/

test:
	$(GO) test ./internal/chunker/... ./internal/hnsw/... -v -timeout 60s

bench:
	$(GO) test ./internal/hnsw/... -bench=BenchmarkRecall10 -benchtime=1x -v

download-model:
	@echo "Downloading BGE-small-en-v1.5 ONNX model files..."
	@mkdir -p $(MODEL_DIR)
	@curl -L --progress-bar -o $(MODEL_DIR)/model.onnx \
		"$(MODEL_URL_BASE)/onnx/model.onnx"
	@curl -L --progress-bar -o $(MODEL_DIR)/tokenizer.json \
		"$(MODEL_URL_BASE)/tokenizer.json"
	@curl -L --progress-bar -o $(MODEL_DIR)/tokenizer_config.json \
		"$(MODEL_URL_BASE)/tokenizer_config.json"
	@curl -L --progress-bar -o $(MODEL_DIR)/special_tokens_map.json \
		"$(MODEL_URL_BASE)/special_tokens_map.json"
	@curl -L --progress-bar -o $(MODEL_DIR)/vocab.txt \
		"$(MODEL_URL_BASE)/vocab.txt"
	@echo "Model downloaded to $(MODEL_DIR)/"

download-ort:
	@echo "Downloading ONNX Runtime $(ORT_VERSION) shared library for Linux x64…"
	@mkdir -p lib
	@curl -L --progress-bar -o /tmp/ort.tgz "$(ORT_URL)"
	@tar -xzf /tmp/ort.tgz -C /tmp/
	@cp /tmp/onnxruntime-linux-x64-$(ORT_VERSION)/lib/libonnxruntime.so.$(ORT_VERSION) lib/onnxruntime.so
	@rm -f /tmp/ort.tgz
	@echo "onnxruntime.so → lib/onnxruntime.so"

clean:
	rm -f $(BINARY)
	rm -f *.prof

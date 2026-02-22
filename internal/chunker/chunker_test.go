package chunker

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestChunkSmallText(t *testing.T) {
	text := strings.Repeat("hello world ", 50) // ~600 bytes
	chunks, err := chunkBytes([]byte(text), "test.txt", DefaultOptions())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Small text (600 bytes < 1200 window) → exactly one chunk
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestChunkLargeText(t *testing.T) {
	// 3000 bytes → should produce multiple chunks with overlap
	text := strings.Repeat("word ", 600)
	opts := Options{MaxBytes: 1000, OverlapBytes: 200}
	chunks, err := chunkBytes([]byte(text), "test.txt", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks for 3000-byte text, got %d", len(chunks))
	}

	// Verify that chunks are no larger than MaxBytes
	for i, c := range chunks {
		if len(c.Text) > opts.MaxBytes {
			t.Errorf("chunk %d length %d exceeds MaxBytes %d", i, len(c.Text), opts.MaxBytes)
		}
	}
}

func TestIsSupportedFile(t *testing.T) {
	// Create a temp text file.
	dir := t.TempDir()
	tf := filepath.Join(dir, "test.go")
	if err := os.WriteFile(tf, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !IsSupportedFile(tf) {
		t.Error("expected .go file to be supported")
	}

	// Binary: a file with null bytes.
	bf := filepath.Join(dir, "test.bin")
	if err := os.WriteFile(bf, []byte{0x00, 0x01, 0x02}, 0o644); err != nil {
		t.Fatal(err)
	}
	// .bin is not a supported extension.
	if IsSupportedFile(bf) {
		t.Error("expected .bin file to be unsupported")
	}

	// Unsupported extension.
	uf := filepath.Join(dir, "photo.png")
	if err := os.WriteFile(uf, []byte{0x89, 0x50, 0x4E, 0x47}, 0o644); err != nil {
		t.Fatal(err)
	}
	if IsSupportedFile(uf) {
		t.Error("expected .png file to be unsupported")
	}
}

func TestChunkFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "doc.md")
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 60)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	chunks, err := ChunkFile(path, DefaultOptions())
	if err != nil {
		t.Fatalf("ChunkFile error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	for i, c := range chunks {
		if c.Path != path {
			t.Errorf("chunk %d: wrong path", i)
		}
		if strings.TrimSpace(c.Text) == "" {
			t.Errorf("chunk %d: empty text", i)
		}
	}
}

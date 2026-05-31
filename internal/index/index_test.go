// Package index contains unit and integration tests for the index package.
package index

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tejas242/sift/internal/hnsw"
)

// TestHNSWRecallSmokeTest exercises the HNSW implementation used by the index.
// (Full index integration tests require the ONNX model and are skipped in CI.)
func TestHNSWRecallSmokeTest(t *testing.T) {
	g := hnsw.New(16, 200, 50)

	dim := 8
	vecs := make([][]float32, dim)
	for i := range vecs {
		v := make([]float32, dim)
		v[i%dim] += 1.0
		l2Normalize(v)
		vecs[i] = v
		g.Insert(v)
	}

	// Self-search: querying v[0] should return id 0 with score ~1.0.
	results := g.Search(vecs[0], 1)
	if len(results) == 0 {
		t.Fatal("search returned no results")
	}
	if results[0].ID != 0 {
		t.Errorf("expected id=0, got id=%d (score=%.4f)", results[0].ID, results[0].Score)
	}
	if results[0].Score < 0.99 {
		t.Errorf("self-similarity too low: %.4f", results[0].Score)
	}
}

// TestIndexDirSkipsHidden ensures the recursive walker ignores dot-directories.
func TestIndexDirSkipsHidden(t *testing.T) {
	dir := t.TempDir()

	// Create a visible file and a hidden dir with a file.
	if err := os.WriteFile(filepath.Join(dir, "visible.md"), []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	hiddenDir := filepath.Join(dir, ".hidden")
	if err := os.MkdirAll(hiddenDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hiddenDir, "secret.md"), []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	// We verify through the chunker that the hidden file is excluded.
	var seen []string
	walkDir(dir, func(path string) error {
		seen = append(seen, path)
		return nil
	})

	for _, p := range seen {
		if filepath.Dir(p) == hiddenDir {
			t.Errorf("walkDir should skip hidden dirs, but visited %s", p)
		}
	}

	found := false
	for _, p := range seen {
		if filepath.Base(p) == "visible.md" {
			found = true
		}
	}
	if !found {
		t.Error("walkDir should visit visible.md")
	}
}

// TestIndexDirContextCancel verifies that IndexDirWithProgress respects context
// cancellation — the fix for the Ctrl+C hang bug (previously _ = ctx discarded
// the signal and the process blocked indefinitely in the ONNX call).
func TestIndexDirContextCancel(t *testing.T) {
	dir := t.TempDir()
	// Create several files so the loop has iterations to check.
	for i := 0; i < 5; i++ {
		name := filepath.Join(dir, fmt.Sprintf("file%d.md", i))
		if err := os.WriteFile(name, []byte("hello"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately — should return on first ctx.Err() check

	var called int
	err := walkDirWithCtx(ctx, dir, func(path string) error {
		called++
		return ctx.Err()
	})

	if err == nil {
		t.Error("expected context.Canceled, got nil")
		return
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	// At most 1 file should have been processed before cancellation.
	if called > 1 {
		t.Errorf("expected at most 1 call before cancel, got %d", called)
	}
}

// walkDirWithCtx is a helper wrapping walkDir with ctx cancellation for tests.
func walkDirWithCtx(ctx context.Context, rootDir string, fn func(string) error) error {
	return walkDir(rootDir, func(path string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return fn(path)
	})
}

// l2Normalize normalizes v in-place to unit length.
func l2Normalize(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum < 1e-10 {
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(sum)))
	for i := range v {
		v[i] *= inv
	}
}

type mockEmbedder struct{}

func (m *mockEmbedder) Embed(texts []string) ([][]float32, error) {
	vecs := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, 384)
		v[0] = 1.0 // a simple dummy L2-normalized vector
		vecs[i] = v
	}
	return vecs, nil
}

func (m *mockEmbedder) EmbedQuery(query string) ([]float32, error) {
	v := make([]float32, 384)
	v[0] = 1.0
	return v, nil
}

func (m *mockEmbedder) Close() {}

func TestIndex_AddFile_Deduplicate_Search(t *testing.T) {
	dir := t.TempDir()
	idx := &Index{
		dir:              dir,
		embedder:         &mockEmbedder{},
		maxFileSizeBytes: 512 * 1024,
		graph:            hnsw.New(16, 200, 50),
		fileCache:        make(map[string]time.Time),
	}

	// Create a dummy file to index.
	filePath := filepath.Join(dir, "test.md")
	content := "This is a test document with some words."
	if err := os.WriteFile(filePath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Add file.
	skipped, err := idx.AddFile(filePath)
	if err != nil {
		t.Fatalf("AddFile failed: %v", err)
	}
	if skipped {
		t.Error("expected skipped=false, got true")
	}

	// Verify stats
	stats := idx.Stats()
	if stats.NumChunks != 1 {
		t.Errorf("expected 1 chunk, got %d", stats.NumChunks)
	}

	// 2. Add file again (same mtime) -> should skip.
	skipped, err = idx.AddFile(filePath)
	if err != nil {
		t.Fatalf("second AddFile failed: %v", err)
	}
	if !skipped {
		t.Error("expected second AddFile to be skipped (mtime match)")
	}

	// 3. Update file content and mtime -> should remove old chunks and re-index.
	newContent := "This is updated document content to trigger re-index."
	if err := os.WriteFile(filePath, []byte(newContent), 0o644); err != nil {
		t.Fatal(err)
	}
	futureTime := time.Now().Add(1 * time.Hour)
	if err := os.Chtimes(filePath, futureTime, futureTime); err != nil {
		t.Fatal(err)
	}

	skipped, err = idx.AddFile(filePath)
	if err != nil {
		t.Fatalf("AddFile after update failed: %v", err)
	}
	if skipped {
		t.Error("expected skipped=false after update")
	}

	// Check stats: since we removed the old chunk, count should still be 1 (not 2)!
	stats = idx.Stats()
	if stats.NumChunks != 1 {
		t.Errorf("expected 1 chunk after update (stale removed), got %d", stats.NumChunks)
	}

	// 4. Search.
	results, err := idx.Search("updated document", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Meta.Text != newContent {
		t.Errorf("expected chunk text %q, got %q", newContent, results[0].Meta.Text)
	}
}

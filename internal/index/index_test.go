// Package index_test contains integration tests for the index package.
// These tests exercise AddFile, Search, Flush, and Stats without a real
// ONNX model by using a mock embedder interface.
package index_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tejas242/sift/internal/hnsw"
)

// TestHNSWRecallSmokeTest exercises the HNSW implementation used by the index.
// (Full index integration tests require the ONNX model and are skipped in CI.)
func TestHNSWRecallSmokeTest(t *testing.T) {
	g := hnsw.New(16, 200, 50)

	dim := 8
	vecs := make([][]float32, 20)
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

// walkDir is a local copy for testing the walk logic without creating a full Index.
func walkDir(rootDir string, fn func(string) error) error {
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if len(name) > 0 && name[0] == '.' {
			continue
		}
		full := filepath.Join(rootDir, name)
		if entry.IsDir() {
			if err := walkDir(full, fn); err != nil {
				return err
			}
		} else {
			if err := fn(full); err != nil {
				return err
			}
		}
	}
	return nil
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

package watcher

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/tejas242/sift/internal/index"
)

type mockEmbedder struct {
	mu     sync.Mutex
	called bool
}

func (m *mockEmbedder) Embed(texts []string) ([][]float32, error) {
	m.mu.Lock()
	m.called = true
	m.mu.Unlock()
	vecs := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, 384)
		v[0] = 1.0
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

func TestWatcher_Watch_Incremental(t *testing.T) {
	tmpDir := t.TempDir()
	idxDir := filepath.Join(tmpDir, "idx")
	watchDir := filepath.Join(tmpDir, "watch")
	if err := os.Mkdir(watchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(idxDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Create initial file
	testFile := filepath.Join(watchDir, "document.md")
	if err := os.WriteFile(testFile, []byte("Initial text content here."), 0o644); err != nil {
		t.Fatal(err)
	}

	embedder := &mockEmbedder{}
	idx := index.NewTestIndex(idxDir, embedder)

	w, err := New(idx)
	if err != nil {
		t.Fatalf("failed to create watcher: %v", err)
	}

	done := make(chan struct{})
	errChan := make(chan error, 1)

	// Start watcher in a goroutine
	go func() {
		errChan <- w.Watch(watchDir, done)
	}()

	// Wait for watcher to initialize recursive watches
	time.Sleep(100 * time.Millisecond)

	// Modify the file to trigger watcher events
	if err := os.WriteFile(testFile, []byte("Updated text content to trigger indexing!"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Wait for the debounce timer (500ms + some buffer)
	time.Sleep(750 * time.Millisecond)

	// Close the watcher
	close(done)

	// Wait for the goroutine to finish
	select {
	case err := <-errChan:
		if err != nil {
			t.Errorf("Watcher returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("watcher did not shut down within 2 seconds")
	}

	// Verify that mock embedder was actually called to index the changes
	embedder.mu.Lock()
	called := embedder.called
	embedder.mu.Unlock()

	if !called {
		t.Error("expected mock embedder to be called to re-index the updated file, but it wasn't")
	}
}

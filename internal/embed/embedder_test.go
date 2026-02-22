package embed

import (
	"testing"
)

// TestL2Normalize checks that l2Normalize produces a unit vector.
func TestL2Normalize(t *testing.T) {
	v := []float32{3, 4, 0} // norm = 5
	l2Normalize(v)
	want := []float32{0.6, 0.8, 0}
	for i, got := range v {
		if diff := got - want[i]; diff < -1e-5 || diff > 1e-5 {
			t.Errorf("v[%d] = %f, want %f", i, got, want[i])
		}
	}
}

// TestEmbedderNew ensures New returns a useful error if models are missing.
func TestEmbedderNew(t *testing.T) {
	_, err := New("/tmp/nonexistent-model-dir-sift-test", "", 0)
	if err == nil {
		t.Fatal("expected error for missing model dir, got nil")
	}
}

// TestEmbedSemanticSimilarity verifies that the BGE-small embeddings produce
// mathematically meaningful similarities using CLS pooling.
func TestEmbedSemanticSimilarity(t *testing.T) {
	// Skip if model isn't downloaded yet.
	e, err := New("../../models", "../../lib/onnxruntime.so", 0)
	if err != nil {
		t.Skipf("skipping: model not found at ../../models: %v", err)
	}
	defer e.Close()

	// 1. Synonym check (should be highly similar)
	vecs, err := e.Embed([]string{
		"a cute baby feline playing with yarn",
		"a tiny kitten swatting at a string",
	})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}

	simKitten := dotProduct(vecs[0], vecs[1])
	if simKitten < 0.70 { // BGE-small usually scores synonyms > 0.7
		t.Errorf("expected high similarity for synonyms, got %f", simKitten)
	}

	// 2. Unrelated check (should be low similarity)
	vecsUnrelated, err := e.Embed([]string{
		"a cute baby feline playing with yarn",
		"instructions for adjusting the carburetor on a 1998 honda civic",
	})
	if err != nil {
		t.Fatalf("embed unrelated: %v", err)
	}

	simCar := dotProduct(vecsUnrelated[0], vecsUnrelated[1])
	if simCar > 0.5 { // Unrelated text is usually < 0.4
		t.Errorf("expected low similarity for unrelated text, got %f", simCar)
	}
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

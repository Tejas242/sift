package hnsw

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// randomVec generates a random unit vector of dimension d.
func randomVec(rng *rand.Rand, d int) []float32 {
	v := make([]float32, d)
	var norm float64
	for i := range v {
		x := rng.NormFloat64()
		v[i] = float32(x)
		norm += x * x
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] /= float32(norm)
	}
	return v
}

func TestInsertSearch(t *testing.T) {
	const dim = 384
	rng := rand.New(rand.NewSource(1))
	g := New(16, 200, 50)

	const n = 200
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = randomVec(rng, dim)
		g.Insert(vecs[i])
	}

	// Query with the first inserted vector — it should find itself as top result.
	results := g.Search(vecs[0], 5)
	if len(results) == 0 {
		t.Fatal("no results returned")
	}
	if results[0].ID != 0 {
		t.Errorf("expected self (id=0) as top result, got id=%d score=%.4f", results[0].ID, results[0].Score)
	}
	if results[0].Score < 0.99 {
		t.Errorf("self-similarity should be ~1.0, got %.4f", results[0].Score)
	}
}

func TestPersistRoundTrip(t *testing.T) {
	const dim = 64
	rng := rand.New(rand.NewSource(7))
	g := New(16, 200, 50)

	const n = 100
	for i := 0; i < n; i++ {
		g.Insert(randomVec(rng, dim))
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "test.hnsw")

	if err := g.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	g2, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if g2.Len() != n {
		t.Errorf("expected %d nodes after load, got %d", n, g2.Len())
	}

	// Both graphs should return the same top result for a query.
	q := randomVec(rng, dim)
	r1 := g.Search(q, 1)
	r2 := g2.Search(q, 1)
	if len(r1) == 0 || len(r2) == 0 {
		t.Fatal("no results from one of the graphs")
	}
	if r1[0].ID != r2[0].ID {
		t.Errorf("top result mismatch: original=%d loaded=%d", r1[0].ID, r2[0].ID)
	}
}

// BenchmarkRecall10 measures recall@10 of HNSW vs brute force on 1000 vectors.
func BenchmarkRecall10(b *testing.B) {
	const (
		dim    = 384
		nIndex = 1000
		nQuery = 50
		k      = 10
	)
	rng := rand.New(rand.NewSource(42))
	g := New(16, 200, 50)

	vecs := make([][]float32, nIndex)
	for i := range vecs {
		vecs[i] = randomVec(rng, dim)
		g.Insert(vecs[i])
	}

	queries := make([][]float32, nQuery)
	for i := range queries {
		queries[i] = randomVec(rng, dim)
	}

	b.ResetTimer()

	var totalRecall float64
	for _, q := range queries {
		// Brute force top-k.
		type sc struct {
			id  int
			sim float32
		}
		scores := make([]sc, nIndex)
		for i, v := range vecs {
			scores[i] = sc{id: i, sim: sim(q, v)}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].sim > scores[j].sim })
		groundTruth := make(map[int]bool, k)
		for i := 0; i < k && i < len(scores); i++ {
			groundTruth[scores[i].id] = true
		}

		// HNSW search.
		results := g.Search(q, k)
		var hits int
		for _, r := range results {
			if groundTruth[int(r.ID)] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	recall := totalRecall / float64(nQuery)
	b.ReportMetric(recall, "recall@10")

	if recall < 0.80 {
		b.Errorf("recall@10 too low: %.3f (want >= 0.80)", recall)
	}

	// Clean up temp file if any.
	_ = os.Remove("bench.hnsw")
}

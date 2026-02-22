// Package hnsw implements a Hierarchical Navigable Small World graph for
// approximate nearest-neighbour search. Vectors are pre-normalized (L2) so
// similarity is computed as a plain dot product, which equals cosine similarity.
//
// Parameters:
//
//	M             = 16   (max neighbours per node per layer, except layer 0 which uses 2*M)
//	efConstruction = 200  (candidate pool size during insertion)
//	efSearch       = 50   (candidate pool size during query)
package hnsw

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"
)

const (
	// DefaultM is the base number of bi-directional connections per node.
	DefaultM = 16
	// DefaultEfConstruction is the size of the dynamic candidate list during build.
	DefaultEfConstruction = 200
	// DefaultEfSearch is the size of the dynamic candidate list during search.
	DefaultEfSearch = 50
)

// Result is a single search result.
type Result struct {
	ID    uint32
	Score float32 // cosine similarity in [0,1]
}

// node is a vertex in the HNSW graph.
type node struct {
	// neighbors[layer] is the list of neighbour IDs at that layer.
	neighbors [][]uint32
	vec       []float32
}

// Graph is the HNSW index.
type Graph struct {
	mu             sync.RWMutex
	nodes          []node
	entryPoint     uint32
	maxLayer       int
	m              int // max connections per layer (Mmax0 = 2*m at layer 0)
	efConstruction int
	efSearch       int
	ml             float64 // level generation factor = 1/ln(m)
	rng            *rand.Rand
}

// New creates an empty HNSW graph with the given parameters.
func New(m, efConstruction, efSearch int) *Graph {
	if m <= 0 {
		m = DefaultM
	}
	if efConstruction <= 0 {
		efConstruction = DefaultEfConstruction
	}
	if efSearch <= 0 {
		efSearch = DefaultEfSearch
	}
	return &Graph{
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		ml:             1.0 / math.Log(float64(m)),
		rng:            rand.New(rand.NewSource(42)),
	}
}

// Len returns the number of nodes in the graph.
func (g *Graph) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodes)
}

// randomLevel draws a random level for a new node using the HNSW exponential law.
func (g *Graph) randomLevel() int {
	return int(math.Floor(-math.Log(g.rng.Float64()) * g.ml))
}

// sim computes dot-product similarity between two pre-normalized vectors.
func sim(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Insert adds a new vector to the graph. The vector must already be L2-normalized.
// The id must equal the current length of the graph (sequential insert).
func (g *Graph) Insert(vec []float32) {
	g.mu.Lock()
	defer g.mu.Unlock()

	id := uint32(len(g.nodes))
	level := g.randomLevel()

	// Allocate neighbors for each layer.
	neighbors := make([][]uint32, level+1)
	for l := 0; l <= level; l++ {
		maxConn := g.m
		if l == 0 {
			maxConn = 2 * g.m
		}
		neighbors[l] = make([]uint32, 0, maxConn)
	}

	g.nodes = append(g.nodes, node{neighbors: neighbors, vec: vec})

	if id == 0 {
		g.entryPoint = 0
		g.maxLayer = level
		return
	}

	ep := g.entryPoint
	epLevel := g.maxLayer

	// Greedy descent through layers above `level`.
	for lc := epLevel; lc > level; lc-- {
		ep = g.greedySearchLayer(vec, ep, lc)
	}

	// Insert into layers [min(level,epLevel) down to 0].
	for lc := min(level, epLevel); lc >= 0; lc-- {
		candidates := g.searchLayer(vec, ep, g.efConstruction, lc)
		selected := g.selectNeighbours(candidates, g.m, lc)

		// Connect new node to selected neighbours.
		g.nodes[id].neighbors[lc] = selected

		// Connect selected neighbours back to new node (bidirectional).
		for _, nb := range selected {
			g.nodes[nb].neighbors[lc] = append(g.nodes[nb].neighbors[lc], id)
			// Prune if over capacity.
			maxConn := g.m
			if lc == 0 {
				maxConn = 2 * g.m
			}
			if len(g.nodes[nb].neighbors[lc]) > maxConn {
				g.nodes[nb].neighbors[lc] = g.pruneNeighbours(nb, g.nodes[nb].neighbors[lc], maxConn, lc)
			}
		}

		if len(candidates) > 0 {
			ep = candidates[0].id // closest found at this layer
		}
	}

	if level > epLevel {
		g.entryPoint = id
		g.maxLayer = level
	}
}

// Search returns the k nearest neighbours to query (must be L2-normalized).
func (g *Graph) Search(query []float32, k int) []Result {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.nodes) == 0 {
		return nil
	}

	ep := g.entryPoint
	epLevel := g.maxLayer

	// Greedy descent to layer 1.
	for lc := epLevel; lc > 0; lc-- {
		ep = g.greedySearchLayer(query, ep, lc)
	}

	// Full search at layer 0 with ef candidates.
	ef := g.efSearch
	if k > ef {
		ef = k
	}
	candidates := g.searchLayer(query, ep, ef, 0)

	// Take top-k.
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	results := make([]Result, len(candidates))
	for i, c := range candidates {
		results[i] = Result{ID: c.id, Score: c.dist}
	}
	return results
}

// candidate is a (id, similarity) pair used in priority queues.
type candidate struct {
	id   uint32
	dist float32 // higher = more similar
}

// maxHeap is a max-heap of candidates (highest similarity first).
type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// minHeap is a min-heap of candidates (lowest similarity first, for pruning).
type minHeap []candidate

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// greedySearchLayer navigates layer lc from ep to find the single closest node.
func (g *Graph) greedySearchLayer(query []float32, ep uint32, lc int) uint32 {
	best := ep
	bestSim := sim(query, g.nodes[ep].vec)

	changed := true
	for changed {
		changed = false
		if lc < len(g.nodes[best].neighbors) {
			for _, nb := range g.nodes[best].neighbors[lc] {
				s := sim(query, g.nodes[nb].vec)
				if s > bestSim {
					bestSim = s
					best = nb
					changed = true
				}
			}
		}
	}
	return best
}

// searchLayer performs the full ef-based beam search at layer lc.
// Returns candidates sorted descending by similarity (index 0 = best).
//
// Algorithm: maintain C (candidates to explore, max-heap) and W (best results, max-heap).
// Always expand the most promising candidate from C. Stop when the best
// unexplored candidate is worse than the worst element in W and W is full.
func (g *Graph) searchLayer(query []float32, ep uint32, ef, lc int) []candidate {
	visited := make(map[uint32]bool)
	visited[ep] = true

	epSim := sim(query, g.nodes[ep].vec)

	// C = candidates to explore, max-heap (best unexplored first).
	C := &maxHeap{{id: ep, dist: epSim}}
	heap.Init(C)

	// W = result set, max-heap bounded to ef elements.
	// We track the worst (minimum) similarity in W separately for O(1) access.
	W := []candidate{{id: ep, dist: epSim}}
	worstSim := epSim

	minSimInW := func() float32 {
		m := W[0].dist
		for _, c := range W[1:] {
			if c.dist < m {
				m = c.dist
			}
		}
		return m
	}

	for C.Len() > 0 {
		// Pop best unexplored candidate.
		c := heap.Pop(C).(candidate)

		// Early exit: if the best candidate remaining is worse than our worst result
		// and W is full, we cannot improve — stop.
		if len(W) >= ef && c.dist < worstSim {
			break
		}

		if lc < len(g.nodes[c.id].neighbors) {
			for _, nb := range g.nodes[c.id].neighbors[lc] {
				if visited[nb] {
					continue
				}
				visited[nb] = true
				s := sim(query, g.nodes[nb].vec)

				if len(W) < ef || s > worstSim {
					heap.Push(C, candidate{id: nb, dist: s})
					W = append(W, candidate{id: nb, dist: s})
					if len(W) > ef {
						// Remove the worst element from W (linear scan — ef ≤ 200).
						minIdx := 0
						for i := 1; i < len(W); i++ {
							if W[i].dist < W[minIdx].dist {
								minIdx = i
							}
						}
						W[minIdx] = W[len(W)-1]
						W = W[:len(W)-1]
					}
					worstSim = minSimInW()
				}
			}
		}
	}

	// Sort W descending by similarity.
	for i := 0; i < len(W)-1; i++ {
		for j := i + 1; j < len(W); j++ {
			if W[j].dist > W[i].dist {
				W[i], W[j] = W[j], W[i]
			}
		}
	}
	return W
}

// selectNeighbours picks the best `m` candidates from a sorted list.
func (g *Graph) selectNeighbours(candidates []candidate, m, _ int) []uint32 {
	if len(candidates) <= m {
		ids := make([]uint32, len(candidates))
		for i, c := range candidates {
			ids[i] = c.id
		}
		return ids
	}
	ids := make([]uint32, m)
	for i := 0; i < m; i++ {
		ids[i] = candidates[i].id
	}
	return ids
}

// pruneNeighbours reduces the neighbour list of node `id` to at most `maxConn`
// entries, keeping the ones with highest similarity.
func (g *Graph) pruneNeighbours(id uint32, nbs []uint32, maxConn, _ int) []uint32 {
	type nb struct {
		id   uint32
		dist float32
	}
	scored := make([]nb, len(nbs))
	for i, n := range nbs {
		scored[i] = nb{id: n, dist: sim(g.nodes[id].vec, g.nodes[n].vec)}
	}
	// Sort descending.
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].dist > scored[i].dist {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}
	if len(scored) > maxConn {
		scored = scored[:maxConn]
	}
	out := make([]uint32, len(scored))
	for i, s := range scored {
		out[i] = s.id
	}
	return out
}

// Package tui provides the production-grade BubbleTea interactive interface for sift.
//
// Layout:
//
//	┌─────────────────────────────────────┐
//	│  sift  semantic file search         │  ← header
//	│  ❯ <query input>                    │  ← search bar
//	│  ─────────────────────────────────  │  ← divider
//	│  0.94  src/main.go                  │  ← results
//	│        func main() ...              │
//	│  ...                                │
//	│  ─────────────────────────────────  │  ← divider
//	│  [3 results]  ↑↓ enter  ^I  ^Q      │  ← status bar
//	└─────────────────────────────────────┘
package tui

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/screenager/sift/internal/index"
)

// ── Palette ──────────────────────────────────────────────────────────────────

var (
	colorAccent  = lipgloss.Color("#7C6AF7") // purple
	colorDim     = lipgloss.Color("#555555") // dark grey
	colorMuted   = lipgloss.Color("#888888") // mid grey
	colorText    = lipgloss.Color("#DDDDDD") // near-white
	colorSubdued = lipgloss.Color("#444444") // for dividers
	colorScore   = lipgloss.Color("#5ECEF5") // cyan for scores
	colorErr     = lipgloss.Color("#FF6B6B") // red
	colorGreen   = lipgloss.Color("#5AF078") // for "indexed"

	sTitle  = lipgloss.NewStyle().Bold(true).Foreground(colorText)
	sAccent = lipgloss.NewStyle().Foreground(colorAccent)
	sDim    = lipgloss.NewStyle().Foreground(colorDim)
	sMuted  = lipgloss.NewStyle().Foreground(colorMuted)
	sScore  = lipgloss.NewStyle().Foreground(colorScore).Bold(true)
	sPath   = lipgloss.NewStyle().Foreground(colorText)
	sDir    = lipgloss.NewStyle().Foreground(colorMuted)
	sSnip   = lipgloss.NewStyle().Foreground(colorMuted)
	sErr    = lipgloss.NewStyle().Foreground(colorErr)
	sGreen  = lipgloss.NewStyle().Foreground(colorGreen)
	sSel    = lipgloss.NewStyle().
		Background(lipgloss.Color("#1E1A3A")).
		Foreground(colorText)
	sHint = lipgloss.NewStyle().
		Foreground(colorDim).
		Background(lipgloss.Color("#111111"))
	sDivider = lipgloss.NewStyle().Foreground(colorSubdued)
	sBadge   = lipgloss.NewStyle().
			Foreground(colorAccent).
			Bold(true)
)

// ── Extension → icon map ─────────────────────────────────────────────────────

var extIcon = map[string]string{
	".go": "󰟓 ", ".py": "󰌠 ", ".rs": "󱘗 ", ".js": "󰌞 ",
	".ts": "󰛦 ", ".md": "󰍔 ", ".txt": "󰦨 ", ".json": "󰘦 ",
	".yaml": "󰗊 ", ".yml": "󰗊 ", ".toml": " ", ".c": "󰙱 ",
	".cpp": "󰙲 ", ".h": "󰙳 ", ".conf": "󰒓 ", ".sh": " ",
}

func fileIcon(path string) string {
	if icon, ok := extIcon[filepath.Ext(path)]; ok {
		return icon
	}
	return " "
}

// ── Spinner frames ────────────────────────────────────────────────────────────

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

type spinTickMsg struct{}

func spinTick() tea.Cmd {
	return tea.Tick(80*time.Millisecond, func(t time.Time) tea.Msg { return spinTickMsg{} })
}

// ── Messages ─────────────────────────────────────────────────────────────────

type mode int

const (
	modeSearch mode = iota
	modeStats
)

type (
	searchResultMsg []index.SearchResult
	errMsg          struct{ err error }
	debounceMsg     struct {
		query string
		id    int
	}
)

// ── Model ─────────────────────────────────────────────────────────────────────

// Model is the BubbleTea application model.
type Model struct {
	idx        *index.Index
	input      textinput.Model
	results    []index.SearchResult
	cursor     int
	mode       mode
	err        error
	width      int
	height     int
	searching  bool
	spinFrame  int
	stats      *index.Stats
	debounceID int
	lastQuery  string
	rerank     bool
}

// New creates a new TUI model backed by the given index.
func New(idx *index.Index, rerank bool) Model {
	ti := textinput.New()
	ti.Placeholder = "search your codebase…"
	ti.Focus()
	ti.CharLimit = 256
	ti.Width = 60
	ti.PromptStyle = sAccent
	ti.Prompt = "❯ "
	ti.TextStyle = lipgloss.NewStyle().Foreground(colorText)

	return Model{
		idx:    idx,
		input:  ti,
		mode:   modeSearch,
		rerank: rerank,
	}
}

// Init is the BubbleTea init hook.
func (m Model) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, spinTick())
}

// Update processes messages.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.Width = m.width - 8
		return m, nil

	case spinTickMsg:
		m.spinFrame = (m.spinFrame + 1) % len(spinnerFrames)
		return m, spinTick()

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "ctrl+q":
			return m, tea.Quit

		case "ctrl+i":
			if m.mode != modeStats {
				m.mode = modeStats
				s := m.idx.Stats()
				m.stats = &s
				m.input.Blur()
			} else {
				m.mode = modeSearch
				m.input.Focus()
				m.stats = nil
			}
			return m, nil

		case "ctrl+r":
			m.rerank = !m.rerank
			// Re-trigger search if we have a query
			q := strings.TrimSpace(m.input.Value())
			if q != "" {
				m.searching = true
				return m, searchCmd(m.idx, q, m.rerank)
			}
			return m, nil

		case "esc":
			m.mode = modeSearch
			m.input.Focus()
			m.stats = nil
			m.err = nil
			return m, nil

		case "up", "ctrl+p":
			if m.cursor > 0 {
				m.cursor--
			}
			return m, nil

		case "down", "ctrl+n":
			if m.cursor < len(m.results)-1 {
				m.cursor++
			}
			return m, nil

		case "enter":
			if m.mode == modeSearch && len(m.results) > 0 {
				res := m.results[m.cursor].Meta
				return m, openInEditor(res.Path, res.LineNum)
			}
			return m, nil
		}

	case debounceMsg:
		if msg.id == m.debounceID && msg.query == m.input.Value() {
			if strings.TrimSpace(msg.query) == "" {
				m.searching = false
				m.results = nil
				return m, nil
			}
			m.searching = true
			m.lastQuery = msg.query
			return m, searchCmd(m.idx, msg.query, m.rerank)
		}
		return m, nil

	case searchResultMsg:
		m.searching = false
		m.results = []index.SearchResult(msg)
		m.cursor = 0
		m.err = nil
		return m, nil

	case errMsg:
		m.searching = false
		m.err = msg.err
		return m, nil
	}

	// Delegate to text input in search mode.
	if m.mode == modeSearch {
		prevVal := m.input.Value()
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if m.input.Value() != prevVal {
			m.debounceID++
			id := m.debounceID
			q := m.input.Value()
			return m, tea.Batch(cmd, debounceCmd(q, id, 280*time.Millisecond))
		}
		return m, cmd
	}

	return m, nil
}

// ── Views ─────────────────────────────────────────────────────────────────────

func (m Model) View() string {
	if m.width == 0 {
		return ""
	}
	if m.mode == modeStats {
		return m.statsView()
	}
	return m.searchView()
}

func (m Model) searchView() string {
	var b strings.Builder
	w := m.width
	divider := sDivider.Render(strings.Repeat("─", clamp(w-2, 10, 200)))

	// ── Header ───────────────────────────────────────────────────────────────
	left := "  " + sTitle.Render("sift") + "  " + sMuted.Render("semantic file search")
	s := m.idx.Stats()
	right := sDim.Render(fmt.Sprintf("%d chunks · %d files", s.NumChunks, s.NumFiles))
	header := padBetween(left, right, w)
	fmt.Fprintln(&b, header)

	// ── Search bar ───────────────────────────────────────────────────────────
	fmt.Fprintln(&b, "  "+m.input.View())
	fmt.Fprintln(&b, "  "+divider)

	// ── Body ──────────────────────────────────────────────────────────────────
	if m.err != nil {
		fmt.Fprintln(&b, sErr.Render("  error: "+m.err.Error()))
	} else if m.searching {
		frame := spinnerFrames[m.spinFrame]
		fmt.Fprintln(&b, "  "+sAccent.Render(frame)+"  "+sMuted.Render("searching…"))
	} else if len(m.results) == 0 && m.input.Value() == "" {
		// Empty state — show hint
		fmt.Fprintln(&b, "")
		fmt.Fprintln(&b, sMuted.Render("  Start typing to search your index semantically."))
		fmt.Fprintln(&b, sDim.Render("  Natural language works: ")+sMuted.Render("\"how does auth work\""))
	} else if len(m.results) == 0 {
		fmt.Fprintln(&b, "")
		fmt.Fprintln(&b, sMuted.Render("  no results for ")+sAccent.Render("\""+m.lastQuery+"\""))
		fmt.Fprintln(&b, sDim.Render("  try rephrasing or indexing more files"))
	} else {
		// Result list
		bodyHeight := m.height - 7 // header+input+div+statusbar+padding
		m.renderResults(&b, bodyHeight)
	}

	// ── Status bar ───────────────────────────────────────────────────────────
	b.WriteString("\n  " + divider + "\n")
	m.renderStatusBar(&b)

	return b.String()
}

func (m *Model) renderResults(b *strings.Builder, maxRows int) {
	// Each result occupies 2 lines: path + snippet
	maxResults := maxRows / 2
	if maxResults < 1 {
		maxResults = 1
	}

	for i, r := range m.results {
		if i >= maxResults {
			remaining := len(m.results) - i
			fmt.Fprintf(b, "  %s\n", sDim.Render(fmt.Sprintf("  … %d more results", remaining)))
			break
		}

		dir := filepath.Dir(r.Meta.Path)
		base := filepath.Base(r.Meta.Path)
		icon := fileIcon(r.Meta.Path)
		score := fmt.Sprintf("%.2f", r.Score)

		snippet := r.Meta.Text
		maxSnip := clamp(m.width-8, 20, 120)
		if len(snippet) > maxSnip {
			snippet = snippet[:maxSnip-1] + "…"
		}
		// Collapse whitespace in snippet for compact display
		snippet = strings.Join(strings.Fields(snippet), " ")

		filename := fmt.Sprintf("%s:%d", base, r.Meta.LineNum)
		pathStr := sDir.Render(dir+"/") + sPath.Render(filename)
		line1 := fmt.Sprintf("  %s  %s%s", sScore.Render(score), icon, pathStr)
		line2 := fmt.Sprintf("  %s  %s", sDim.Render("    "), sSnip.Render(snippet))

		if i == m.cursor {
			// Pad to width for full-row highlight
			raw1 := stripStyle(score) + "  " + icon + dir + "/" + filename
			raw2 := "       " + snippet
			pad1 := clamp(m.width-len(raw1)-3, 0, m.width)
			pad2 := clamp(m.width-len(raw2)-3, 0, m.width)
			line1 = sSel.Render("  " + sScore.Render(score) + "  " + icon + sDir.Render(dir+"/") + sPath.Render(filename) + strings.Repeat(" ", pad1))
			line2 = sSel.Render("  " + "       " + sSnip.Render(snippet) + strings.Repeat(" ", pad2))
		}

		fmt.Fprintln(b, line1)
		fmt.Fprintln(b, line2)
	}
}

func (m *Model) renderStatusBar(b *strings.Builder) {
	var left string
	if len(m.results) > 0 {
		left = sGreen.Render(fmt.Sprintf("  %d result", len(m.results)))
		if len(m.results) != 1 {
			left += sGreen.Render("s")
		}
	} else if m.err != nil {
		left = "  " + sErr.Render(m.err.Error())
	} else {
		left = sDim.Render("  no results")
	}

	rerankStatus := sDim.Render("rerank:off")
	if m.rerank {
		if m.idx.HasReranker() {
			rerankStatus = sAccent.Render("rerank:on")
		} else {
			rerankStatus = sErr.Render("rerank:missing model")
		}
	}

	right := sHint.Render(rerankStatus + "  ^r toggle  ^i info  esc clear  ↑↓ nav  enter open  ^q quit  ")
	fmt.Fprint(b, padBetween(left, right, m.width))
}

func (m Model) statsView() string {
	var b strings.Builder
	w := clamp(m.width, 10, 200)
	divider := sDivider.Render(strings.Repeat("─", w-2))

	fmt.Fprintln(&b, "  "+sTitle.Render("sift")+" "+sMuted.Render("— index info"))
	fmt.Fprintln(&b, "  "+divider)

	if m.stats != nil {
		s := m.stats
		fmt.Fprintln(&b, "")
		row := func(label, value string) {
			fmt.Fprintf(&b, "  %-22s %s\n", sDim.Render(label), value)
		}
		row("chunks indexed", sAccent.Render(fmt.Sprintf("%d", s.NumChunks)))
		row("files indexed", sAccent.Render(fmt.Sprintf("%d", s.NumFiles)))
		row("index size on disk", sAccent.Render(fmt.Sprintf("%d KB", s.IndexSizeKB)))
		if !s.LastUpdated.IsZero() {
			ago := time.Since(s.LastUpdated).Round(time.Second)
			row("last updated", sMuted.Render(s.LastUpdated.Format("2006-01-02 15:04")+" ("+ago.String()+" ago)"))
		}
		row("embedding model", sMuted.Render("BGE-small-en-v1.5 (384-dim)"))
		row("hnsw parameters", sMuted.Render("M=16  ef_build=200  ef_search=50"))
	}

	fmt.Fprintln(&b, "")
	fmt.Fprintln(&b, "  "+divider)
	fmt.Fprint(&b, sHint.Render("  esc back to search  ctrl+q quit"+strings.Repeat(" ", clamp(w-35, 0, 200))))
	return b.String()
}

// ── Commands ──────────────────────────────────────────────────────────────────

func debounceCmd(query string, id int, delay time.Duration) tea.Cmd {
	return func() tea.Msg {
		time.Sleep(delay)
		return debounceMsg{query: query, id: id}
	}
}

func searchCmd(idx *index.Index, query string, rerank bool) tea.Cmd {
	return func() tea.Msg {
		results, err := idx.Search(query, 10, rerank)
		if err != nil {
			return errMsg{err}
		}
		return searchResultMsg(results)
	}
}

func openInEditor(path string, lineNum int) tea.Cmd {
	editor := os.Getenv("EDITOR")
	if editor == "" {
		// Try common editors in order.
		for _, e := range []string{"nvim", "vim", "nano", "vi"} {
			if _, err := exec.LookPath(e); err == nil {
				editor = e
				break
			}
		}
	}

	args := []string{}
	baseEditor := filepath.Base(editor)
	if baseEditor == "nvim" || baseEditor == "vim" || baseEditor == "vi" || baseEditor == "nano" {
		if lineNum > 0 {
			args = append(args, fmt.Sprintf("+%d", lineNum))
		}
	} else if baseEditor == "code" {
		if lineNum > 0 {
			args = append(args, "--goto", fmt.Sprintf("%s:%d", path, lineNum))
			path = "" // Already included in --goto
		}
	}

	if path != "" {
		args = append(args, path)
	}

	c := exec.Command(editor, args...)
	return tea.ExecProcess(c, func(err error) tea.Msg {
		if err != nil {
			return errMsg{err}
		}
		return nil
	})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// padBetween pads left and right strings to fill width.
func padBetween(left, right string, width int) string {
	// Approximate visible length (ignore ANSI escapes with a rough heuristic).
	lv := visibleLen(left)
	rv := visibleLen(right)
	gap := width - lv - rv - 2
	if gap < 1 {
		gap = 1
	}
	return left + strings.Repeat(" ", gap) + right
}

// visibleLen estimates printable character count (strips common ANSI sequences).
func visibleLen(s string) int {
	// Quick pass: count bytes, subtract escape sequences.
	n := 0
	inEsc := false
	for _, c := range s {
		if c == '\x1b' {
			inEsc = true
		}
		if inEsc {
			if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
				inEsc = false
			}
			continue
		}
		n++
	}
	return n
}

// stripStyle returns the raw string without Lipgloss ANSI styling.
func stripStyle(s string) string { return s }

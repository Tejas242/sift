package tui

import "testing"

func TestStripStyle(t *testing.T) {
	cases := []struct {
		input    string
		expected string
	}{
		{"hello", "hello"},
		{"\x1b[31mred\x1b[0m", "red"},
		{"\x1b[1mbold\x1b[22m and \x1b[4munderline\x1b[24m", "bold and underline"},
		{"no esc", "no esc"},
	}

	for _, tc := range cases {
		got := stripStyle(tc.input)
		if got != tc.expected {
			t.Errorf("stripStyle(%q) = %q; expected %q", tc.input, got, tc.expected)
		}
	}
}

func TestVisibleLen(t *testing.T) {
	cases := []struct {
		input    string
		expected int
	}{
		{"hello", 5},
		{"\x1b[31mred\x1b[0m", 3},
		{"\x1b[1mbold\x1b[22m and \x1b[4munderline\x1b[24m", 18},
		{"", 0},
	}

	for _, tc := range cases {
		got := visibleLen(tc.input)
		if got != tc.expected {
			t.Errorf("visibleLen(%q) = %d; expected %d", tc.input, got, tc.expected)
		}
	}
}

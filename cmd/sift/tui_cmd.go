package main

import (
	tea "github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"
	"github.com/tejas242/sift/internal/tui"
)

func init() {
	rootCmd.AddCommand(&cobra.Command{
		Use:   "tui",
		Short: "Launch interactive BubbleTea search interface",
		RunE: func(cmd *cobra.Command, args []string) error {
			idx, err := openIndex(ortLib)
			if err != nil {
				return err
			}
			defer idx.Close()

			m := tui.New(idx)
			p := tea.NewProgram(m, tea.WithAltScreen())
			_, err = p.Run()
			return err
		},
	})
}

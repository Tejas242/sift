package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/tejas242/sift/internal/config"
)

var forceFlag bool

func init() {
	clearCmd := &cobra.Command{
		Use:   "clear",
		Short: "Remove the sift index (.sift/ directory)",
		RunE: func(cmd *cobra.Command, args []string) error {
			if _, err := os.Stat(config.DefaultSiftDir); os.IsNotExist(err) {
				fmt.Println("No index found — nothing to clear.")
				return nil
			}
			if !forceFlag {
				fmt.Printf("Remove %s? This cannot be undone. [y/N] ", config.DefaultSiftDir)
				var ans string
				fmt.Scanln(&ans)
				if ans != "y" && ans != "Y" {
					fmt.Println("Aborted.")
					return nil
				}
			}
			if err := os.RemoveAll(config.DefaultSiftDir); err != nil {
				return fmt.Errorf("clear: %w", err)
			}
			fmt.Println("Index cleared.")
			return nil
		},
	}
	clearCmd.Flags().BoolVar(&forceFlag, "force", false, "skip confirmation prompt")
	rootCmd.AddCommand(clearCmd)
}

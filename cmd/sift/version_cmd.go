package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func init() {
	rootCmd.AddCommand(&cobra.Command{
		Use:   "version",
		Short: "Print the version number of sift",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("sift version %s (%s) built on %s\n", version, commit, date)
		},
	})
}

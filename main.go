package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/ssh"
	"github.com/charmbracelet/wish"
	"github.com/charmbracelet/wish/activeterm"
	"github.com/charmbracelet/wish/bubbletea"
	"github.com/charmbracelet/wish/logging"

	"wriggle/game"
	"wriggle/ui"
)

func main() {
	policyPath := flag.String("bot-policy", "", "JSON neural policy for bots (default: heuristic AI)")
	port := flag.Int("port", 23234, "SSH listening port")
	flag.Parse()
	if *port < 1 || *port > 65535 {
		log.Fatal("port must be between 1 and 65535")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	world := game.NewWorld()
	if *policyPath != "" {
		p, err := game.LoadPolicy(*policyPath)
		if err != nil {
			log.Fatalf("load bot policy: %v", err)
		}
		if err = world.SetPolicy(p); err != nil {
			log.Fatal(err)
		}
		log.Printf("Neural bot policy: %s", *policyPath)
	}
	go world.Run(ctx)
	cleanup := func(next ssh.Handler) ssh.Handler {
		return func(s ssh.Session) {
			defer func() {
				if m, ok := s.Context().Value(modelKey{}).(*ui.Model); ok {
					m.Close()
				}
			}()
			next(s)
		}
	}
	server, err := wish.NewServer(wish.WithAddress(fmt.Sprintf(":%d", *port)), wish.WithHostKeyPath(".ssh/wriggle_ed25519"), wish.WithMiddleware(bubbletea.Middleware(func(s ssh.Session) (tea.Model, []tea.ProgramOption) {
		m := ui.NewModel(s.Context(), world, s.User())
		s.Context().SetValue(modelKey{}, m)
		return m, []tea.ProgramOption{tea.WithAltScreen(), tea.WithMouseAllMotion()}
	}), cleanup, activeterm.Middleware(), logging.Middleware()))
	if err != nil {
		log.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- server.ListenAndServe() }()
	log.Printf("Wriggle listening — ssh -p %d localhost", *port)
	select {
	case <-ctx.Done():
	case err := <-done:
		if err != nil && !errors.Is(err, net.ErrClosed) && !errors.Is(err, ssh.ErrServerClosed) {
			log.Print(err)
		}
	}
	cancel()
	shutdown, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	if err := server.Shutdown(shutdown); err != nil {
		log.Print(err)
		_ = server.Close()
	}
}

type modelKey struct{}

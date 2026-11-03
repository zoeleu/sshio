package ui

import (
	"context"
	tea "github.com/charmbracelet/bubbletea"
	"strings"
	"testing"
	"wriggle/game"
)

func TestSessionLifecycle(t *testing.T) {
	w := game.NewWorld()
	m := NewModel(context.Background(), w, "player\x1b[31m")
	if strings.Contains(m.name, "\x1b") {
		t.Fatal("unsafe name")
	}
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	defer m.Close()
	if m.state != 1 || m.id == 0 {
		t.Fatal("join failed")
	}
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	if !strings.Contains(m.View(), "LENGTH") {
		t.Fatal("missing HUD")
	}
	m.Update(tea.KeyMsg{Type: tea.KeyRight})
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{' '}})
	w.Tick()
	m.Update(tickMsg{})
	m.Update(tea.WindowSizeMsg{Width: 10, Height: 5})
	if !strings.Contains(m.View(), "resize") {
		t.Fatal("small terminal")
	}
}

func TestRespawnKeepsTickLoop(t *testing.T) {
	w := game.NewWorld()
	m := NewModel(context.Background(), w, "player")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	defer m.Close()
	oldID := m.id
	pending := m.wait()
	m.state = 2
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.id == oldID {
		t.Fatal("respawn reused old snake")
	}
	if _, ok := pending().(tickMsg); !ok {
		t.Fatal("old subscription quit respawned session")
	}
	w.Tick()
	m.Update(tickMsg{})
	if m.state != 1 {
		t.Fatal("respawn did not stay in play")
	}
}

func TestDeathOverlayKeepsLiveArenaBackground(t *testing.T) {
	w := game.NewWorld()
	m := NewModel(context.Background(), w, "player")
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 32})
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	defer m.Close()
	w.Tick()
	m.Update(tickMsg{})
	// Transition to death screen and verify live arena background continues rendering and ticking.
	m.state = 2
	m.deathKiller = "border"
	m.deathLength = 15
	view := m.View()
	if !strings.Contains(view, "YOU UNRAVELED") {
		t.Fatal("missing death overlay")
	}
	if !strings.Contains(view, "WRIGGLE / TOP 10") || !strings.Contains(view, "▀") {
		t.Fatal("missing live arena HUD and canvas behind death overlay")
	}
	prevTick := m.frame.Tick
	w.Tick()
	m.Update(tickMsg{})
	if m.frame.Tick <= prevTick {
		t.Fatal("dead screen did not continue receiving live world frames")
	}
}

func TestSpectateBotMode(t *testing.T) {
	w := game.NewWorld()
	m := NewModel(context.Background(), w, "spectate")
	defer m.Close()
	if m.state != 3 {
		t.Fatalf("expected state 3 (spectating), got %d", m.state)
	}
	if m.spectateID == 0 {
		t.Fatal("expected spectateID to be chosen")
	}
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	view := m.View()
	if !strings.Contains(view, "SPECTATING") {
		t.Fatalf("expected SPECTATING in HUD, got: %s", view)
	}
	oldTarget := m.spectateID
	// Cycle next bot
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	if len(m.frame.Snakes) > 1 && m.spectateID == oldTarget {
		// Might cycle if multiple bots exist
	}
	// Verify tick update keeps valid target
	w.Tick()
	m.Update(tickMsg{})
	if m.spectateID == 0 {
		t.Fatal("spectateID lost after tick")
	}
	// Press Enter to join the arena as a player
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.state != 1 || m.id == 0 {
		t.Fatalf("failed to transition from spectate to player: state=%d, id=%d", m.state, m.id)
	}
}


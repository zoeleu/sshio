package ui

import (
	"context"
	"math"
	"math/rand"
	"strings"
	"time"
	"unicode"
	"wriggle/game"

	tea "github.com/charmbracelet/bubbletea"
)

type tickMsg struct{}
type Model struct {
	world             *game.World
	ctx               context.Context
	name              string
	id, width, height int
	ticks             <-chan struct{}
	frame             game.Snapshot
	state             int // 0: welcome, 1: playing, 2: unraveled, 3: spectating
	heading           float64
	lastKey           string
	lastAt            time.Time
	deathPos          game.Vec2
	deathLength       int
	deathKiller       string
	spectateID        int
}

func NewModel(ctx context.Context, w *game.World, name string) *Model {
	m := &Model{world: w, ctx: ctx, name: cleanName(name), width: 80, height: 24}
	if strings.EqualFold(m.name, "spectate") || strings.EqualFold(m.name, "watch") || strings.EqualFold(m.name, "bot") {
		m.spectate()
	}
	return m
}
func cleanName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == ' ' || r == '_' || r == '-' {
			b.WriteRune(r)
		}
		if b.Len() >= 20 {
			break
		}
	}
	return strings.TrimSpace(b.String())
}
func (m *Model) Init() tea.Cmd {
	if m.state == 3 {
		return m.wait()
	}
	return nil
}
func (m *Model) wait() tea.Cmd {
	ticks, ctx := m.ticks, m.ctx
	return func() tea.Msg {
		select {
		case <-ctx.Done():
			return tea.Quit()
		case <-ticks:
			return tickMsg{}
		}
	}
}
func (m *Model) join() {
	if m.id != 0 {
		m.world.Leave(m.id)
		m.id = 0
	}
	m.id, m.ticks = m.world.Join(m.name)
	m.frame = m.world.Snapshot()
	for _, s := range m.frame.Snakes {
		if s.ID == m.id {
			m.heading = s.Heading
			if len(s.Segments) > 0 {
				m.deathPos = s.Segments[0]
			}
			m.deathLength = s.Length()
			break
		}
	}
	m.lastKey = ""
	m.state = 1
}
func (m *Model) spectate() {
	if m.id != 0 {
		m.world.Leave(m.id)
		m.id = 0
	}
	m.id, m.ticks = m.world.Spectate()
	m.frame = m.world.Snapshot()
	m.state = 3
	m.pickBot(0)
}
func (m *Model) pickBot(dir int) {
	var bots []int
	for _, s := range m.frame.Snakes {
		if s.Alive && len(s.Segments) > 0 && s.ID != m.id {
			bots = append(bots, s.ID)
		}
	}
	if len(bots) == 0 {
		m.spectateID = 0
		return
	}
	if dir == 0 {
		m.spectateID = bots[rand.Intn(len(bots))]
		return
	}
	currIdx := -1
	for i, id := range bots {
		if id == m.spectateID {
			currIdx = i
			break
		}
	}
	if currIdx == -1 {
		m.spectateID = bots[0]
		return
	}
	nextIdx := (currIdx + dir) % len(bots)
	if nextIdx < 0 {
		nextIdx += len(bots)
	}
	m.spectateID = bots[nextIdx]
}
func (m *Model) Close() {
	if m.id != 0 {
		m.world.Leave(m.id)
		m.id = 0
	}
}
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch v := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = v.Width
		m.height = v.Height
	case tickMsg:
		m.frame = m.world.Snapshot()
		if m.state == 3 {
			valid := false
			for _, s := range m.frame.Snakes {
				if s.ID == m.spectateID && s.Alive && len(s.Segments) > 0 {
					valid = true
					break
				}
			}
			if !valid {
				m.pickBot(0)
			}
			return m, m.wait()
		}
		for _, s := range m.frame.Snakes {
			if s.ID == m.id {
				if len(s.Segments) > 0 {
					m.deathPos = s.Segments[0]
				}
				m.deathLength = s.Length()
				m.deathKiller = s.Killer
				if !s.Alive {
					m.state = 2
				}
			}
		}
		return m, m.wait()
	case tea.KeyMsg:
		k := v.String()
		if k == "ctrl+c" || (m.state != 0 && (k == "q" || k == "esc")) {
			m.Close()
			return m, tea.Quit
		}
		if m.state == 0 {
			switch k {
			case "enter":
				if m.name == "" {
					m.name = "Wriggler"
				}
				m.join()
				return m, m.wait()
			case "tab":
				m.spectate()
				return m, m.wait()
			case "backspace", "ctrl+h":
				r := []rune(m.name)
				if len(r) > 0 {
					m.name = string(r[:len(r)-1])
				}
			default:
				if v.Type == tea.KeyRunes {
					m.name = cleanName(m.name + string(v.Runes))
				}
			}
			return m, nil
		}
		if m.state == 2 {
			if k == "enter" {
				m.join()
				return m, m.wait()
			}
			if k == "tab" || k == "s" || k == "S" {
				m.spectate()
				return m, m.wait()
			}
			return m, nil
		}
		if m.state == 3 {
			switch k {
			case "enter":
				if m.name == "" {
					m.name = "Wriggler"
				}
				m.join()
				return m, m.wait()
			case "tab", "n", "N", "right", "down", " ":
				m.pickBot(1)
			case "p", "P", "left", "up":
				m.pickBot(-1)
			case "r", "R":
				m.pickBot(0)
			}
			return m, nil
		}
		if k == "tab" {
			m.spectate()
			return m, m.wait()
		}
		dirs := map[string]game.Vec2{"up": {X: 0, Y: -1}, "w": {X: 0, Y: -1}, "down": {X: 0, Y: 1}, "s": {X: 0, Y: 1}, "left": {X: -1, Y: 0}, "a": {X: -1, Y: 0}, "right": {X: 1, Y: 0}, "d": {X: 1, Y: 0}}
		if d, ok := dirs[k]; ok {
			now := time.Now()
			if p, ok := dirs[m.lastKey]; ok && now.Sub(m.lastAt) < 130*time.Millisecond && p.X*d.X+p.Y*d.Y == 0 {
				d = d.Add(p)
			}
			m.heading = math.Atan2(d.Y, d.X)
			m.lastKey = k
			m.lastAt = now
			m.world.SetInput(m.id, m.heading, false)
		}
		if k == " " {
			m.world.SetInput(m.id, m.heading, true)
		}
	case tea.MouseMsg:
		if m.state == 1 {
			x := float64(v.X - m.width/2)
			y := float64(v.Y-(m.height-2)/2) * 2
			if x != 0 || y != 0 {
				m.heading = math.Atan2(y, x)
				m.world.SetInput(m.id, m.heading, false)
			}
			if v.Action == tea.MouseActionRelease {
				m.world.SetMouseBoost(m.id, false)
			} else if v.Button == tea.MouseButtonLeft {
				m.world.SetMouseBoost(m.id, true)
			}
		}
	}
	return m, nil
}

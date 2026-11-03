package game

import (
	"context"
	"sync"
	"testing"
)

func fixture() *World { return newWorld(100, 0, 0) }
func put(w *World, id int, p Vec2, h float64) *Snake {
	s := &Snake{ID: id, Name: "test", Alive: true, Score: 30, Heading: h, TargetHeading: h, Segments: []Vec2{p, p.Sub(direction(h).Scale(Spacing))}}
	w.snakes[id] = s
	return s
}
func TestHeadBodyCollision(t *testing.T) {
	w := fixture()
	a := put(w, 1, Vec2{}, 0)
	put(w, 2, Vec2{1, 0}, 0)
	w.Tick()
	if a.Alive {
		t.Fatal("head touching another snake survived")
	}
	if len(w.pellets) == 0 {
		t.Fatal("death did not drop pellets")
	}
}
func TestHeadOn(t *testing.T) {
	w := fixture()
	a := put(w, 1, Vec2{-0.8, 0}, 0)
	b := put(w, 2, Vec2{0.8, 0}, 3.141592653589793)
	w.Tick()
	if a.Alive || b.Alive {
		t.Fatal("both heads must die")
	}
}
func TestGrowth(t *testing.T) {
	w := fixture()
	s := put(w, 1, Vec2{}, 0)
	w.pellets = []Pellet{{Vec2{0.8, 0}, 5}}
	w.Tick()
	if s.Score != 35 || len(s.Segments) != s.Length() {
		t.Fatalf("growth: %+v", s)
	}
}
func TestBoost(t *testing.T) {
	w := fixture()
	s := put(w, 1, Vec2{}, 0)
	s.BoostUntil = 10
	w.Tick()
	if s.Score >= 30 || s.Segments[0].X < 1.5 {
		t.Fatal("boost did not spend score and increase speed")
	}
	s.Score = MinScore
	w.Tick()
	if s.Score < MinScore {
		t.Fatal("boost spent minimum score")
	}
}
func TestBorder(t *testing.T) {
	w := fixture()
	s := put(w, 1, Vec2{99.9, 0}, 0)
	w.Tick()
	if s.Alive || s.Killer != "the border" {
		t.Fatal("border did not kill")
	}
}
func TestOwnBodySafe(t *testing.T) {
	w := fixture()
	s := put(w, 1, Vec2{}, 0)
	s.Segments = append(s.Segments, Vec2{0.8, 0})
	w.Tick()
	if !s.Alive {
		t.Fatal("own body killed snake")
	}
}
func TestSnapshotIsolationAndLeave(t *testing.T) {
	w := fixture()
	id, ch := w.Join("human")
	v := w.Snapshot()
	v.Snakes[0].Segments[0] = Vec2{999, 999}
	if w.Snapshot().Snakes[0].Segments[0].X == 999 {
		t.Fatal("snapshot aliases world")
	}
	w.Leave(id)
	if _, ok := <-ch; ok {
		t.Fatal("leave did not close subscription")
	}
	w.Leave(id)
}
func TestConcurrentSessions(t *testing.T) {
	w := newWorld(100, 4, 100)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id, _ := w.Join("human")
			defer w.Leave(id)
			for n := 0; n < 100; n++ {
				w.SetInput(id, float64(n), true)
				w.SetMouseBoost(id, n%2 == 0)
				w.Snapshot()
			}
		}()
	}
	for i := 0; i < 100; i++ {
		w.Tick()
	}
	wg.Wait()
}
func TestRunCancelled(t *testing.T) {
	w := fixture()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w.Run(ctx)
}

func TestBotPopulationAndNotifications(t *testing.T) {
	w := newWorld(100, 3, 0)
	id, ch := w.Join("human")
	defer w.Leave(id)
	humans, bots := 0, 0
	for _, s := range w.Snapshot().Snakes {
		if s.Bot {
			bots++
		} else {
			humans++
		}
	}
	if humans != 1 || bots != 2 {
		t.Fatalf("got %d humans and %d bots", humans, bots)
	}
	for i := 0; i < 10; i++ {
		w.Tick()
	}
	// A session that does not consume frames must never block the world.
	if len(ch) != 1 {
		t.Fatalf("expected one coalesced tick, got %d", len(ch))
	}
	w.Leave(id)
	for _, s := range w.Snapshot().Snakes {
		if !s.Bot {
			t.Fatal("human retained after leave")
		}
	}
	if len(w.Snapshot().Snakes) != 3 {
		t.Fatal("bots did not refill arena")
	}
}

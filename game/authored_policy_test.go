package game

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func authoredPolicy(t *testing.T) *Policy {
	t.Helper()
	path := os.Getenv("AUTHORED_POLICY")
	if path == "" {
		path = "../models/authored.json"
	}
	p, err := LoadPolicy(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestAuthoredArtifact(t *testing.T) {
	p := authoredPolicy(t)
	if p.Version != 3 || p.InputSize() != 692 || len(p.W1) != 64 || len(p.WH) != 0 || len(p.W2) != 14 {
		t.Fatal("architecture changed")
	}
	count := len(p.B1) + len(p.B2)
	for _, matrix := range [][][]float64{p.W1, p.W2} {
		for _, row := range matrix {
			count += len(row)
		}
	}
	if count != 45262 {
		t.Fatalf("parameter count %d", count)
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "policy.json")
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	q, err := LoadPolicy(path)
	if err != nil || !reflect.DeepEqual(p, q) {
		t.Fatal("round trip", err)
	}
	// Prove identical paired weights and a strict boost bias disadvantage.
	for a := 0; a < 7; a++ {
		if !reflect.DeepEqual(p.W2[a], p.W2[a+7]) || p.B2[a+7] != p.B2[a]-1 {
			t.Fatal("boost invariant")
		}
	}
	w, s := authoredFixture("food_left")
	obs := w.observe(s)
	for personality := 0; personality <= 4; personality++ {
		if p.PersonalityAction(obs, personality) != p.Action(obs) {
			t.Fatal("unexpected personality dependence")
		}
	}
	// Exercise publication and the actual live bot's staggered decision path.
	if err = w.SetPolicy(p); err != nil {
		t.Fatal(err)
	}
	s.Bot = true
	w.target = 1
	w.tick = 2 // ID 1 chooses on tick 3.
	want := p.PersonalityAction(w.observe(s), s.Personality)
	before := s.Heading
	w.Tick()
	if math.Abs(s.TargetHeading-before-actionTurns[want%7]) > 1e-12 {
		t.Fatal("live bot dispatch differs")
	}
}

func authoredFixture(name string) (*World, *Snake) {
	w := newSeededWorld(42, 80, 0, 0)
	add := func(pos Vec2, h, score float64) *Snake {
		id := w.spawn("diagnostic", false)
		s := w.snakes[id]
		placeTrainingSnake(s, pos, h, score)
		return s
	}
	s := add(Vec2{}, 0, 12)
	switch name {
	case "food_left":
		w.pellets = []Pellet{{Vec2{12, -8}, 2}}
	case "food_right":
		w.pellets = []Pellet{{Vec2{12, 8}, 2}}
	case "food_straight":
		w.pellets = []Pellet{{Vec2{12, 0}, 2}}
	case "hazard_food":
		add(Vec2{7, -13}, -math.Pi/2, 32)
		w.pellets = []Pellet{{Vec2{12, 0}, 8}}
	case "head_on":
		add(Vec2{9, 0}, math.Pi, 32)
	case "corner":
		add(Vec2{7, -13}, -math.Pi/2, 32)
		add(Vec2{-12, -6}, math.Pi, 32)
		w.pellets = []Pellet{{Vec2{12, -3}, 8}}
	case "corridor":
		add(Vec2{-12, -3}, math.Pi, 32)
		add(Vec2{-12, 3}, math.Pi, 32)
		w.pellets = []Pellet{{Vec2{15, 0}, 2}}
	case "border":
		placeTrainingSnake(s, Vec2{70, 0}, 0, 12)
	case "unavoidable_border":
		placeTrainingSnake(s, Vec2{79.95, 0}, 0, 12)
	case "own_tail":
		s.Segments[len(s.Segments)-1] = Vec2{0.8, 0}
	}
	return w, s
}

func TestAuthoredDiagnostics(t *testing.T) {
	p := authoredPolicy(t)
	cases := []struct {
		name    string
		allowed []int
		survive bool
	}{
		{"food_left", []int{0, 1, 2}, true}, {"food_right", []int{4, 5, 6}, true},
		{"food_straight", []int{3}, true}, {"hazard_food", []int{0, 6}, true},
		{"head_on", []int{0, 6}, true}, {"corner", []int{6}, true},
		{"corridor", []int{3}, true}, {"border", []int{0, 6}, true},
		{"unavoidable_border", []int{0, 6}, false}, {"own_tail", []int{3}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w, s := authoredFixture(tc.name)
			action := p.Action(w.observe(s))
			allowed := false
			for _, a := range tc.allowed {
				allowed = allowed || action == a
			}
			if !allowed {
				t.Errorf("action %d, want one of %v", action, tc.allowed)
			}
			w.applyAction(s, action, 4)
			for i := 0; i < 4 && s.Alive; i++ {
				w.Tick()
			}
			t.Logf("first action=%d alive after 4 ticks=%v score=%.2f", action, s.Alive, s.Score)
			if s.Alive != tc.survive {
				t.Errorf("survival=%v want %v", s.Alive, tc.survive)
			}
		})
	}
}

func TestAuthoredRotation(t *testing.T) {
	p := authoredPolicy(t)
	for _, name := range []string{"food_left", "food_right", "corner"} {
		w, s := authoredFixture(name)
		obs := w.observe(s)
		action := p.Action(obs)
		for _, angle := range []float64{math.Pi / 2, math.Pi, 3 * math.Pi / 2, 0.37} {
			v, me := authoredFixture(name)
			rot := func(p Vec2) Vec2 {
				return Vec2{p.X*math.Cos(angle) - p.Y*math.Sin(angle), p.X*math.Sin(angle) + p.Y*math.Cos(angle)}
			}
			for _, snake := range v.snakes {
				for j, pos := range snake.Segments {
					snake.Segments[j] = rot(pos)
				}
				snake.Heading += angle
				snake.TargetHeading += angle
			}
			for j, pellet := range v.pellets {
				v.pellets[j].Pos = rot(pellet.Pos)
			}
			got := v.observe(me)
			for j, x := range got {
				if math.Abs(x-obs[j]) > 1e-10 {
					t.Fatalf("%s angle %f feature %d differs: %g vs %g", name, angle, j, x, obs[j])
				}
			}
			if p.Action(got) != action {
				t.Fatalf("%s rotation changed action", name)
			}
		}
	}
}

// Growth is applied after all movement/collision substeps. It duplicates the
// already-moved last segment, rather than leaving an occupied grid cell behind.
func TestAuthoredTailMovementAndGrowth(t *testing.T) {
	var tails []Vec2
	for _, growth := range []bool{false, true} {
		w := newSeededWorld(42, 80, 0, 0)
		id := w.spawn("learner", false)
		s := w.snakes[id]
		placeTrainingSnake(s, Vec2{}, 0, 12)
		oid := w.spawn("tail", false)
		other := w.snakes[oid]
		placeTrainingSnake(other, Vec2{1.3, 12.45}, math.Pi/2, 12)
		oldTail := other.Segments[len(other.Segments)-1]
		if growth {
			w.pellets = []Pellet{{Vec2{1.3, 13.25}, 5}}
		}
		p := authoredPolicy(t)
		action := p.Action(w.observe(s))
		w.applyAction(s, action, 4)
		w.Tick()
		tail := other.Segments[len(other.Segments)-1]
		tails = append(tails, tail)
		if tail.Y <= oldTail.Y || !s.Alive {
			t.Fatalf("growth %v: tail did not move safely", growth)
		}
		if growth && len(other.Segments) != 25 {
			t.Fatal("growth not applied")
		}
		t.Logf("growth=%v action=%d tailY=%.4f->%.4f alive=%v", growth, action, oldTail.Y, tail.Y, s.Alive)
	}
	if tails[0].Sub(tails[1]).Len() > 1e-12 {
		t.Fatal("growth unexpectedly held tail before collision")
	}
}

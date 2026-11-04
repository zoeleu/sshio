package game

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func deepPolicy(t *testing.T) *Policy {
	t.Helper()
	p, err := LoadPolicy("../models/authored-deep.json")
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestAuthoredDeepArtifact(t *testing.T) {
	p := deepPolicy(t)
	if p.Version != 3 || len(p.W1) != 64 || p.InputSize() != 692 || len(p.WH) != 64 || len(p.W2) != 14 {
		t.Fatal("wrong deep architecture")
	}
	count := len(p.B1) + len(p.BH) + len(p.B2)
	for _, matrix := range [][][]float64{p.W1, p.WH, p.W2} {
		for _, row := range matrix {
			count += len(row)
		}
	}
	if count != 49422 {
		t.Fatalf("parameter count=%d", count)
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "deep.json")
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	q, err := LoadPolicy(path)
	if err != nil || !reflect.DeepEqual(p, q) {
		t.Fatal("round trip failed", err)
	}
	w, s := authoredFixture("food_left")
	if err = w.SetPolicy(p); err != nil {
		t.Fatal(err)
	}
	want := p.PersonalityAction(w.observe(s), s.Personality)
	s.Bot = true
	w.target = 1
	w.tick = 2
	before := s.Heading
	w.Tick()
	if math.Abs(s.TargetHeading-before-actionTurns[want%7]) > 1e-12 {
		t.Fatal("deep live dispatch mismatch")
	}
}

func TestAuthoredDeepBoostConditions(t *testing.T) {
	cases := []struct {
		name   string
		score  float64
		food   Vec2
		value  float64
		hazard bool
		boost  bool
	}{
		{"rich_aligned_clear", 80, Vec2{12, 0}, 8, false, true},
		{"insufficient_score", 12, Vec2{12, 0}, 8, false, false},
		{"food_too_far", 80, Vec2{40, 0}, 8, false, false},
		{"food_behind", 80, Vec2{-12, 0}, 8, false, false},
		{"food_left", 80, Vec2{12, -4}, 8, false, false},
		{"food_right", 80, Vec2{12, 4}, 8, false, false},
		{"poor_food", 80, Vec2{12, 0}, 1, false, false},
		{"rival_ahead", 80, Vec2{12, 0}, 8, true, false},
	}
	p := deepPolicy(t)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := newSeededWorld(42, 80, 0, 0)
			id := w.spawn("learner", false)
			s := w.snakes[id]
			placeTrainingSnake(s, Vec2{}, 0, tc.score)
			w.pellets = []Pellet{{tc.food, tc.value}}
			if tc.hazard {
				oid := w.spawn("rival", false)
				placeTrainingSnake(w.snakes[oid], Vec2{9, 0}, math.Pi, 32)
			}
			action := p.Action(w.observe(s))
			if (action >= 7) != tc.boost {
				t.Fatalf("action=%d expected boost=%v", action, tc.boost)
			}
			if action >= 7 && action != 10 {
				t.Fatal("turning boost enabled")
			}
			for personality := 0; personality <= 4; personality++ {
				if p.PersonalityAction(w.observe(s), personality) != action {
					t.Fatal("personality dependence")
				}
			}
			w.applyAction(s, action, 4)
			for i := 0; i < 4 && s.Alive; i++ {
				w.Tick()
			}
			if !s.Alive {
				t.Fatal("died during boost diagnostic")
			}
			if tc.boost && math.Abs(s.BoostSpent-.8) > 1e-10 {
				t.Fatalf("boost did not execute: %g", s.BoostSpent)
			}
			t.Logf("action=%d alive=%v boost_spent=%.2f", action, s.Alive, s.BoostSpent)
		})
	}
}

func TestAuthoredDeepFoodSuppression(t *testing.T) {
	p := deepPolicy(t)
	w, s := authoredFixture("food_straight")
	obs := w.observe(s)
	extended := append(obs, 0, 0, 0, 0)
	foodScore := func(o []float64) float64 {
		h := tanhLayer(p.W1, p.B1, o)
		hh := tanhLayer(p.WH, p.BH, h)
		return 16 * (hh[13] + hh[14]) // action 3's paired food gate
	}
	clear := foodScore(extended)
	// Identical target; represented forward body hazard at every visible ray.
	for ray := 0; ray < 24; ray++ {
		extended[516+76+4*ray] = 0
	}
	blocked := foodScore(extended)
	if clear < 1 || math.Abs(blocked) > .01 {
		t.Fatalf("food gate clear=%g blocked=%g", clear, blocked)
	}
	t.Logf("food contribution clear=%.6f blocked=%.6f", clear, blocked)
}

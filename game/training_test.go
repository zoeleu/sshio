package game

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func testPolicy() *Policy {
	p := &Policy{Version: PolicyVersion, W1: [][]float64{make([]float64, ObservationSize)}, B1: []float64{0}, W2: make([][]float64, ActionCount), B2: make([]float64, ActionCount)}
	for i := range p.W2 {
		p.W2[i] = []float64{0}
	}
	p.W1[0][0] = 2
	p.W2[4][0] = 1
	p.B2[3] = 0.1
	return p
}
func TestPolicyInferenceAndValidation(t *testing.T) {
	p := testPolicy()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	o := make([]float64, ObservationSize)
	if p.Action(o) != 3 {
		t.Fatal("wrong bias action")
	}
	o[0] = 1
	if p.Action(o) != 4 {
		t.Fatal("wrong tanh action")
	}
	p.W1[0][0] = math.NaN()
	if p.Validate() == nil {
		t.Fatal("accepted NaN")
	}
	p = testPolicy()
	p.W2[0] = nil
	if p.Validate() == nil {
		t.Fatal("accepted wrong dimensions")
	}
}
func TestPolicyRoundTripAndCopy(t *testing.T) {
	p := testPolicy()
	data, _ := json.Marshal(p)
	path := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadPolicy(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p, loaded) {
		t.Fatal("policy round trip changed weights")
	}
	w := fixture()
	if err = w.SetPolicy(p); err != nil {
		t.Fatal(err)
	}
	p.B2[3] = 999
	if w.policy.B2[3] == 999 {
		t.Fatal("published mutable policy")
	}
	if err = os.WriteFile(path, append(data, []byte(" {}")...), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = LoadPolicy(path); err == nil {
		t.Fatal("accepted trailing JSON")
	}
}
func TestLegacyPolicyStillValidates(t *testing.T) {
	p := &Policy{Version: LegacyPolicyVersion, W1: [][]float64{make([]float64, LegacyObservationSize)}, B1: []float64{0}, W2: make([][]float64, ActionCount), B2: make([]float64, ActionCount)}
	for i := range p.W2 {
		p.W2[i] = []float64{0}
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if p.InputSize() != LegacyObservationSize {
		t.Fatal("bad legacy input size")
	}
	if len(fixture().observeLegacy(put(fixture(), 1, Vec2{}, 0))) != LegacyObservationSize {
		t.Fatal("bad legacy observation")
	}
	// A v2 policy must reject a v1-width first layer.
	bad := testPolicy()
	bad.W1[0] = make([]float64, LegacyObservationSize)
	if bad.Validate() == nil {
		t.Fatal("v2 accepted v1 width")
	}
}

func TestV2ObservationLayout(t *testing.T) {
	w := fixture()
	me := put(w, 1, Vec2{}, 0)
	near := put(w, 2, Vec2{10, 0}, math.Pi)
	_ = put(w, 3, Vec2{30, 0}, 0)
	near.speed, near.boosting = 1.6, true
	w.pellets = []Pellet{{Vec2{5, 0}, 3}, {Vec2{8, 2}, 1}}
	o := w.observeSingle(me)
	if len(o) != SingleObservationSize {
		t.Fatal("wrong single size")
	}
	for _, v := range o {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < -1 || v > 1 {
			t.Fatalf("out of range %v", o)
		}
	}
	// First rival slot is the nearest rival, present, ahead, boosting.
	if o[rivalBlock+10] != 1 || o[rivalBlock] <= 0 || o[rivalBlock+8] != 1 {
		t.Fatalf("rival slot wrong: %v", o[rivalBlock:rivalBlock+11])
	}
	if o[rivalBlock+rivalStride+10] != 1 {
		t.Fatal("second rival missing")
	}
	if o[rivalBlock+2*rivalStride+10] != 0 {
		t.Fatal("empty rival slot marked present")
	}
	// First food target is the high-value pellet ahead.
	if o[foodBlock] <= 0 || o[foodBlock+3] <= 0 {
		t.Fatalf("food target wrong: %v", o[foodBlock:foodBlock+4])
	}
	// Forward sector: body clearance is finite, head clearance separate.
	if o[sectorBlock] >= 1 && o[sectorBlock+1] >= 1 {
		t.Fatal("forward sector should sense the rival ahead")
	}
	stacked := w.observe(me)
	if len(stacked) != ObservationSize {
		t.Fatal("wrong stacked size")
	}
	for frame := 0; frame < FrameCount; frame++ {
		if !reflect.DeepEqual(stacked[frame*SingleObservationSize:(frame+1)*SingleObservationSize], o) {
			t.Fatal("fresh stack must repeat the first frame")
		}
	}
}

func TestFrameStackOrderAndReset(t *testing.T) {
	var stack frameStack
	a := make([]float64, SingleObservationSize)
	b := make([]float64, SingleObservationSize)
	a[0], b[0] = 1, 2
	first := stack.push(a)
	if first[0] != 1 || first[(FrameCount-1)*SingleObservationSize] != 1 {
		t.Fatal("first push must fill")
	}
	out := stack.push(b)
	if out[0] != 1 || out[(FrameCount-1)*SingleObservationSize] != 2 {
		t.Fatal("stack order wrong")
	}
	stack.reset()
	if stack.peek()[0] != 0 {
		t.Fatal("reset must clear")
	}
	out = stack.push(b)
	if out[0] != 2 {
		t.Fatal("push after reset must refill")
	}
}

func TestSeededTrainingReproducible(t *testing.T) {
	a, _ := NewTrainingEnv(123, 3, 200)
	b, _ := NewTrainingEnv(123, 3, 200)
	if !reflect.DeepEqual(a.State(), b.State()) {
		t.Fatal("reset not reproducible")
	}
	for i := 0; i < 40; i++ {
		x, err := a.Step(-1)
		if err != nil {
			t.Fatal(err)
		}
		y, err := b.Step(-1)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(x, y) {
			t.Fatalf("seed diverged at step %d", i)
		}
		for _, v := range x.Observation {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < -1 || v > 1 {
				t.Fatalf("invalid observation %v", v)
			}
		}
		if x.Done {
			break
		}
	}
}
func TestTrainingBoundsAndTerminal(t *testing.T) {
	if _, err := NewTrainingEnv(1, 100, 100); err == nil {
		t.Fatal("accepted excessive opponents")
	}
	e, _ := NewTrainingEnv(1, 0, 4)
	if _, err := e.Step(-2); err == nil {
		t.Fatal("allowed missing policy")
	}
	if _, err := e.Step(99); err == nil {
		t.Fatal("accepted invalid action")
	}
	x, err := e.Step(3)
	if err != nil || !x.Done {
		t.Fatal("horizon not terminal", err)
	}
	if _, err = e.Step(3); err == nil {
		t.Fatal("stepped terminal episode")
	}
}
func TestKillCredits(t *testing.T) {
	w := fixture()
	victim := put(w, 1, Vec2{}, 0)
	killer := put(w, 2, Vec2{1, 0}, 0)
	w.Tick()
	if victim.Alive || killer.Kills != 1 {
		t.Fatalf("missing kill credit: alive=%v kills=%d", victim.Alive, killer.Kills)
	}
}
func TestBatchReproducibleAndReset(t *testing.T) {
	a, err := NewTrainingBatch(22, 3, 4, 8)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := NewTrainingBatch(22, 3, 4, 8)
	if !reflect.DeepEqual(a.State(true), b.State(true)) {
		t.Fatal("batch reset not deterministic")
	}
	actions := make([]int, 12)
	for i := range actions {
		actions[i] = 3
	}
	for step := 0; step < 4; step++ {
		x, err := a.Step(actions, false)
		if err != nil {
			t.Fatal(err)
		}
		y, _ := b.Step(actions, false)
		if !reflect.DeepEqual(x, y) {
			t.Fatalf("parallel arenas diverged on step %d", step)
		}
		if len(x.Observations) != 12 || len(x.Observations[0]) != ObservationSize {
			t.Fatal("wrong observation dimensions")
		}
		if step%2 == 1 {
			for _, done := range x.Dones {
				if !done {
					t.Fatal("arena horizon did not terminate every trajectory")
				}
			}
		}
	}
	if _, err = a.Step(actions[:1], false); err == nil {
		t.Fatal("accepted short action batch")
	}
	actions[0] = -2
	if _, err = a.Step(actions, false); err == nil {
		t.Fatal("accepted invalid batch action")
	}
}
func TestBatchRespawnsDeadWorm(t *testing.T) {
	b, _ := NewTrainingBatch(1, 1, 2, 100)
	a := b.arenas[0]
	old := a.ids[0]
	s := a.world.snakes[old]
	s.Segments[0] = Vec2{a.world.radius - 0.01, 0}
	s.Heading = 0
	x, err := b.Step([]int{3, 3}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !x.Dones[0] || x.Deaths < 1 || a.ids[0] == old {
		t.Fatal("dead worm did not respawn with terminal signal")
	}
	if _, ok := a.world.snakes[old]; ok {
		t.Fatal("retained dead learner")
	}
}

func TestCurriculumReproducibleAndVaried(t *testing.T) {
	a, _ := NewTrainingBatch(73, 12, 4, 12)
	b, _ := NewTrainingBatch(73, 12, 4, 12)
	if err := a.SetCurriculum(2, true); err != nil {
		t.Fatal(err)
	}
	_ = b.SetCurriculum(2, true)
	x, y := a.State(false), b.State(false)
	if !reflect.DeepEqual(x, y) {
		t.Fatal("curriculum reset is not reproducible")
	}
	seen := map[string]bool{}
	radii := map[float64]bool{}
	for i, name := range x.Scenarios {
		seen[name] = true
		radii[x.Radii[i]] = true
	}
	for _, name := range []string{"forage", "border", "cutoff", "escape", "chase", "arena", "sparse", "dense", "production"} {
		if !seen[name] {
			t.Fatalf("missing scenario %s", name)
		}
	}
	if len(radii) < 4 {
		t.Fatal("arena geometry did not vary")
	}
	actions := make([]int, 48)
	for i := range actions {
		actions[i] = 3
	}
	for i := 0; i < 4; i++ {
		x, _ = a.Step(actions, false)
		y, _ = b.Step(actions, false)
		if !reflect.DeepEqual(x, y) {
			t.Fatalf("curriculum diverged at step %d", i)
		}
	}
	if a.SetCurriculum(3, false) == nil {
		t.Fatal("accepted invalid stage")
	}
}
func TestBatchHeuristicOpponentsAndMetrics(t *testing.T) {
	b, _ := NewTrainingBatch(82, 1, 4, 100)
	state, err := b.Step([]int{-1, 3, -1, 3}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(state.AgentFood) != 4 || len(state.AgentKills) != 4 || len(state.AgentDeaths) != 4 {
		t.Fatal("missing per-agent statistics")
	}
	for _, i := range []int{0, 2} {
		s := b.arenas[0].world.snakes[b.arenas[0].ids[i]]
		if s.brain.until == 0 {
			t.Fatal("heuristic opponent did not steer")
		}
	}
	sum := 0.0
	for _, v := range state.AgentFood {
		sum += v
	}
	if math.Abs(sum-state.Food) > 1e-9 {
		t.Fatal("aggregate food differs from per-agent food")
	}
}
func TestMutualKillAndExactCounters(t *testing.T) {
	w := fixture()
	a := put(w, 1, Vec2{-0.8, 0}, 0)
	b := put(w, 2, Vec2{0.8, 0}, math.Pi)
	w.Tick()
	if a.MutualKills != 1 || b.MutualKills != 1 {
		t.Fatal("mutual collision not credited separately")
	}
	// Boosting snake wins a head-on collision even against a larger non-boosting snake.
	w = fixture()
	a = put(w, 1, Vec2{-0.8, 0}, 0)
	b = put(w, 2, Vec2{0.8, 0}, math.Pi)
	a.Score = 15
	a.BoostUntil = 5
	b.Score = 40
	w.Tick()
	if !a.Alive || b.Alive || a.Kills != 1 {
		t.Fatalf("expected boosting snake a to win head-on: a.Alive=%v b.Alive=%v a.Kills=%d", a.Alive, b.Alive, a.Kills)
	}
	// When both snakes are boosting, the larger snake wins.
	w = fixture()
	a = put(w, 1, Vec2{-0.8, 0}, 0)
	b = put(w, 2, Vec2{0.8, 0}, math.Pi)
	a.Score = 20
	a.BoostUntil = 5
	b.Score = 45
	b.BoostUntil = 5
	w.Tick()
	if a.Alive || !b.Alive || b.Kills != 1 {
		t.Fatalf("expected larger boosting snake b to win head-on: a.Alive=%v b.Alive=%v b.Kills=%d", a.Alive, b.Alive, b.Kills)
	}
	w = fixture()
	s := put(w, 1, Vec2{}, 0)
	s.BoostUntil = 5
	w.pellets = []Pellet{{Vec2{1.6, 0}, 3}}
	w.Tick()
	if s.FoodEaten != 3 || math.Abs(s.BoostSpent-0.2) > 1e-9 {
		t.Fatalf("incorrect counters food=%f boost=%f", s.FoodEaten, s.BoostSpent)
	}
}
func TestMotionTrackerDistinguishesCircles(t *testing.T) {
	var circle, straight motionTracker
	count := 0
	for i := 0; i <= 40; i++ {
		a := float64(i) * 2 * math.Pi / 36
		if circle.sample(direction(a).Scale(4), a+math.Pi/2) {
			count++
		}
		if straight.sample(Vec2{float64(i), 0}, 0) {
			t.Fatal("straight movement flagged as circling")
		}
	}
	if count == 0 {
		t.Fatal("circle was not detected")
	}
}
func TestProductionEvaluationMatchesLiveGeometry(t *testing.T) {
	e, _ := NewTrainingEnv(2, 3, 20)
	if err := e.Scenario("production"); err != nil {
		t.Fatal(err)
	}
	if e.world.radius != 300 || len(e.world.pellets) != 3500 || len(e.world.snakes) != 15 {
		t.Fatal("production evaluation differs from live population/geometry")
	}
}

func TestPersonalityPolicyAndModulation(t *testing.T) {
	p3 := &Policy{
		Version: PersonalityPolicyVersion,
		W1:      [][]float64{make([]float64, PersonalityObservationSize)},
		B1:      []float64{0},
		W2:      make([][]float64, ActionCount),
		B2:      make([]float64, ActionCount),
	}
	for i := range p3.W2 {
		p3.W2[i] = []float64{0}
	}
	// Map personality slot 0 (Aggressive) to action 10, slot 3 (Afraid) to action 3.
	p3.W1[0][ObservationSize+0] = 2.0
	p3.W1[0][ObservationSize+3] = -2.0
	p3.W2[10][0] = 1.0
	p3.W2[3][0] = -1.0
	if err := p3.Validate(); err != nil {
		t.Fatal(err)
	}
	obs := make([]float64, ObservationSize)
	if got := p3.PersonalityAction(obs, PersonalityAggressive); got != 10 {
		t.Fatalf("expected action 10 for aggressive v3, got %d", got)
	}
	if got := p3.PersonalityAction(obs, PersonalityAfraid); got != 3 {
		t.Fatalf("expected action 3 for afraid v3, got %d", got)
	}
	// Verify v2 policy modulation produces distinct boost preference between Aggressive and Conservative.
	newest := (FrameCount - 1) * SingleObservationSize
	obs[newest+2] = 1.0  // canBoost
	obs[newest+12] = 0.4 // rival in attack range
	p2 := testPolicy()
	p2.B2[3] = 0.25
	p2.B2[10] = 0.20
	if got := p2.PersonalityAction(obs, PersonalityAggressive); got != 10 {
		t.Fatalf("expected aggressive v2 to prefer boost action 10, got %d", got)
	}
	if got := p2.PersonalityAction(obs, PersonalityConservative); got != 3 {
		t.Fatalf("expected conservative v2 to avoid boost and pick action 3, got %d", got)
	}
}


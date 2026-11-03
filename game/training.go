package game

import (
	"fmt"
	"math"
)

// TrainingEnv advances the real authoritative simulation with no wall-clock waits.
// It is owned by one caller; each reset starts a new seeded episode.
type TrainingEnv struct {
	world            *World
	policy           *Policy
	id, steps, limit int
	motion           motionTracker
	circles          int
	stack            frameStack
	TeacherEnabled   bool
}
type Transition struct {
	Observation []float64 `json:"observation"`
	Reward      float64   `json:"reward"`
	Done        bool      `json:"done"`
	Teacher     int       `json:"teacher"`
	Score       float64   `json:"score"`
	Food        float64   `json:"food"`
	Kills       int       `json:"kills"`
	Ticks       int       `json:"ticks"`
	Alive       bool      `json:"alive"`
	MutualKills int       `json:"mutual_kills"`
	Circles     int       `json:"circles"`
	BoostSpent  float64   `json:"boost_spent"`
}

func NewTrainingEnv(seed int64, opponents, limit int) (*TrainingEnv, error) {
	if opponents < 0 || opponents > 63 || limit < 4 || limit > 20000 {
		return nil, fmt.Errorf("opponents must be 0..63 and horizon 4..20000 ticks")
	}
	// Compact arena preserves food density and creates regular encounters.
	scale := math.Max(1, float64(opponents+1)/4)
	w := newSeededWorld(seed, 90*math.Sqrt(scale), opponents, int(320*scale))
	id := w.spawn("learner", false)
	w.target = opponents + 1
	return &TrainingEnv{world: w, id: id, limit: limit, TeacherEnabled: true}, nil
}
func (e *TrainingEnv) teacher() int {
	s := e.world.snakes[e.id]
	copy := *s
	g := grid{}
	for _, id := range e.world.snakeIDs() {
		other := e.world.snakes[id]
		if other.Alive {
			for _, p := range other.Segments {
				g.add(bodyPoint{p, id})
			}
		}
	}
	e.world.steerBot(&copy, g)
	s.brain = copy.brain
	d := math.Atan2(math.Sin(copy.TargetHeading-s.Heading), math.Cos(copy.TargetHeading-s.Heading))
	action := 0
	best := math.Inf(1)
	for i, a := range actionTurns {
		if dist := math.Abs(a - d); dist < best {
			best = dist
			action = i
		}
	}
	if copy.BoostUntil > e.world.tick {
		action += 7
	}
	return action
}
func (e *TrainingEnv) State() Transition {
	s := e.world.snakes[e.id]
	t := Transition{Observation: e.stack.push(e.world.observeSingle(s)), Done: !s.Alive || e.steps >= e.limit, Score: s.Score, Food: s.FoodEaten, Kills: s.Kills, Ticks: e.steps, Alive: s.Alive, MutualKills: s.MutualKills, Circles: e.circles, BoostSpent: s.BoostSpent}
	if !t.Done && e.TeacherEnabled {
		t.Teacher = e.teacher()
	}
	return t
}

// -1 runs the original heuristic every tick; -2 runs the loaded network.
func (e *TrainingEnv) Step(action int) (Transition, error) {
	if action < -2 || action >= ActionCount {
		return Transition{}, fmt.Errorf("action must be -2..%d", ActionCount-1)
	}
	s := e.world.snakes[e.id]
	if !s.Alive || e.steps >= e.limit {
		return Transition{}, fmt.Errorf("episode is done; reset before stepping")
	}
	if action == -2 {
		if e.policy == nil {
			return Transition{}, fmt.Errorf("no policy loaded")
		}
		obs := e.stack.peek()
		if e.policy.Version == LegacyPolicyVersion {
			obs = e.world.observeLegacy(s)
		} else if len(e.stack.frames) == 0 {
			obs = e.stack.push(e.world.observeSingle(s))
		}
		if e.policy.Version == PersonalityPolicyVersion {
			p := s.Personality
			if p == PersonalityBalanced {
				p = (e.id % PersonalityCount) + 1
			}
			action = e.policy.PersonalityAction(obs, p)
		} else {
			action = e.policy.Action(obs)
		}
	}
	previous, kills := s.Score, s.Kills
	if action >= 0 {
		e.world.applyAction(s, action, 4)
	}
	for i := 0; i < 4 && s.Alive && e.steps < e.limit; i++ {
		if action == -1 {
			g := grid{}
			for _, id := range e.world.snakeIDs() {
				other := e.world.snakes[id]
				if other.Alive {
					for _, p := range other.Segments {
						g.add(bodyPoint{p, id})
					}
				}
			}
			e.world.steerBot(s, g)
		}
		e.world.Tick()
		e.steps++
	}
	if e.motion.sample(s.Segments[0], s.Heading) {
		e.circles++
	}
	t := e.State()
	t.Reward = 0.3*(s.Score-previous) + 6*float64(s.Kills-kills) - 0.01
	if !s.Alive {
		t.Reward -= 10
	}
	return t, nil
}
func (e *TrainingEnv) SetPolicy(p *Policy) error {
	// Evaluation opponents remain the heuristic baseline, not the network.
	if p != nil {
		if err := p.Validate(); err != nil {
			return err
		}
	}
	e.policy = p
	return nil
}

// SetOpponentPolicy freezes the opposing population for a training rollout.
// nil keeps the hand-written opponents for baseline and mixed-opponent training.
func (e *TrainingEnv) SetOpponentPolicy(p *Policy) error { return e.world.SetPolicy(p) }

func (e *TrainingEnv) Scenario(name string) error {
	if name == "production" {
		w := newSeededWorld(e.world.rng.Int63(), 300, 14, 3500)
		e.id = w.spawn("learner", false)
		w.target = 15
		e.world = w
		e.stack.reset()
		return nil
	}
	if name == "" || name == "arena" {
		return nil
	}
	if name != "cutoff" && name != "escape" && name != "chase" && name != "border" && name != "forage" {
		return fmt.Errorf("unknown scenario %q", name)
	}
	if name == "border" {
		s := e.world.snakes[e.id]
		a := e.world.rng.Float64() * 2 * math.Pi
		placeTrainingSnake(s, direction(a).Scale(e.world.radius-10), a, 20)
		return nil
	}
	if name == "forage" {
		s := e.world.snakes[e.id]
		for i := 0; i < 12; i++ {
			e.world.pellets = append(e.world.pellets, Pellet{s.Segments[0].Add(direction(s.Heading).Scale(8 + float64(i))), 1})
		}
		return nil
	}
	ids := e.world.snakeIDs()
	var victim *Snake
	for _, id := range ids {
		if id != e.id {
			victim = e.world.snakes[id]
			break
		}
	}
	if victim == nil {
		return fmt.Errorf("cutoff requires an opponent")
	}
	a := e.world.rng.Float64() * 2 * math.Pi
	rotate := func(p Vec2) Vec2 { return Vec2{p.X*math.Cos(a) - p.Y*math.Sin(a), p.X*math.Sin(a) + p.Y*math.Cos(a)} }
	p0, p1, h0, h1, score := Vec2{0, -10}, Vec2{-10, 0}, math.Pi/2, 0.0, 30.0
	if name == "escape" {
		p0, p1 = Vec2{0, -7}, Vec2{-7, 0}
		score = 45
	}
	if name == "chase" {
		p0, p1, h1 = Vec2{0, -12}, Vec2{0, 10}, math.Pi/2
		score = 45
	}
	score0, score1 := score, score
	if name == "escape" {
		score0, score1 = 20, 70
	}
	if name == "chase" {
		score1 = 20
	}
	placeTrainingSnake(e.world.snakes[e.id], rotate(p0), h0+a, score0)
	placeTrainingSnake(victim, rotate(p1), h1+a, score1)
	return nil
}

func placeTrainingSnake(s *Snake, p Vec2, heading, score float64) {
	s.Score = score
	s.Heading = heading
	s.TargetHeading = heading
	s.Segments = nil
	s.brain = botBrain{}
	s.speed, s.turnRate, s.prevTurn, s.prevBoost = 0.8, 0, 0, 0
	s.boosting, s.headingInit = false, true
	s.prevHeading = heading
	s.BoostUntil, s.MouseBoost = 0, false
	for i := 0; i < s.Length(); i++ {
		s.Segments = append(s.Segments, p.Sub(direction(heading).Scale(float64(i)*Spacing)))
	}
}

type motionTracker struct {
	origin          Vec2
	heading, turned float64
	samples         int
	initialized     bool
}

func (m *motionTracker) sample(p Vec2, heading float64) bool {
	if !m.initialized {
		*m = motionTracker{origin: p, heading: heading, initialized: true}
		return false
	}
	m.turned += math.Atan2(math.Sin(heading-m.heading), math.Cos(heading-m.heading))
	m.heading = heading
	m.samples++
	circle := math.Abs(m.turned) > 5 && p.Sub(m.origin).Len() < 5
	if circle || m.samples >= 60 {
		*m = motionTracker{origin: p, heading: heading, initialized: true}
	}
	return circle
}

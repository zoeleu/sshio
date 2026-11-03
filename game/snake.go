package game

const MinScore = 12.0
const Spacing = 0.65

const (
	PersonalityCount        = 4
	PersonalityBalanced     = 0
	PersonalityAggressive   = 1
	PersonalityGreedy       = 2
	PersonalityConservative = 3
	PersonalityAfraid       = 4
)

var personalityTags = [...]string{"", "AGG", "GRD", "CNS", "AFR"}

type Snake struct {
	brain                         botBrain
	speed                         float64
	turnRate                      float64
	prevHeading                   float64
	prevTurn, prevBoost           float64
	boosting, headingInit         bool
	ID                            int
	Personality                   int
	Kills                         int
	MutualKills                   int
	FoodEaten, BoostSpent         float64
	Name                          string
	Segments                      []Vec2
	Heading, TargetHeading, Score float64
	Color                         int
	Bot, Alive                    bool
	Killer                        string
	BoostUntil                    uint64
	MouseBoost                    bool
}

func (s Snake) Length() int { return int(s.Score) + 8 }

// WidthScale returns the snake's width multiplier in [1.0, 4.0] as it grows.
func (s Snake) WidthScale() float64 {
	if s.Score <= MinScore {
		return 1.0
	}
	t := (s.Score - MinScore) / 180.0
	if t > 1.0 {
		t = 1.0
	}
	return 1.0 + 3.0*t
}

// Boosting reports whether the snake is actively boosting at the given world tick.
func (s Snake) Boosting(tick uint64) bool {
	return (s.boosting || s.MouseBoost || s.BoostUntil > tick) && s.Score > MinScore
}

// PersonalityTag returns a short 3-letter label for non-default bot personalities.
func (s Snake) PersonalityTag() string {
	if s.Personality >= 0 && s.Personality < len(personalityTags) {
		return personalityTags[s.Personality]
	}
	return ""
}

type Pellet struct {
	Pos   Vec2
	Value float64
}
type Snapshot struct {
	Snakes  []Snake
	Pellets []Pellet
	Radius  float64
	Tick    uint64
}

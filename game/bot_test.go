package game

import (
	"math"
	"math/rand"
	"testing"
)

func TestBotEatsOffAxisFood(t *testing.T) {
	w := fixture()
	w.target = 1
	s := put(w, 1, Vec2{}, 0)
	s.Bot = true
	// This angle falls between the old fixed steering choices.
	w.pellets = []Pellet{{Vec2{12, 5}, 5}}
	for i := 0; i < 100 && s.Score == 30; i++ {
		w.Tick()
	}
	if !s.Alive || s.Score != 35 {
		t.Fatalf("bot circled instead of eating: alive=%v score=%v head=%v", s.Alive, s.Score, s.Segments[0])
	}
}
func TestBotIgnoresUnreachableFoodBehindHead(t *testing.T) {
	w := fixture()
	w.target = 1
	s := put(w, 1, Vec2{}, 0)
	w.pellets = []Pellet{{Vec2{-3, 0}, 1}, {Vec2{15, 4}, 1}}
	if goal := w.foodGoal(s); goal != w.pellets[1].Pos {
		t.Fatalf("targeted food inside turning circle: %v", goal)
	}
}
func TestBotPrefersRichFoodPatch(t *testing.T) {
	w := fixture()
	w.target = 1
	s := put(w, 1, Vec2{}, 0)
	w.pellets = []Pellet{{Vec2{10, -2}, 1}}
	for i := 0; i < 10; i++ {
		w.pellets = append(w.pellets, Pellet{Vec2{16 + float64(i%3), 3 + float64(i/3)}, 1})
	}
	if goal := w.foodGoal(s); goal.Y < 0 {
		t.Fatalf("ignored richer food patch: %v", goal)
	}
}
func TestBotCutsOffRival(t *testing.T) {
	w := fixture()
	w.target = 2
	bot := put(w, 1, Vec2{0, -10}, math.Pi/2)
	bot.Bot = true
	victim := put(w, 2, Vec2{-10, 0}, 0)
	if _, prey := w.attackGoal(bot); prey != victim.ID {
		t.Fatal("bot missed a winning interception")
	}
	for i := 0; i < 35 && victim.Alive; i++ {
		w.Tick()
	}
	if victim.Alive {
		t.Fatalf("bot failed to cut off rival; bot=%v rival=%v", bot.Segments[0], victim.Segments[0])
	}
	if !bot.Alive {
		t.Fatal("bot rammed rival instead of laying a body across its route")
	}
}
func TestBotRejectsLosingHeadChase(t *testing.T) {
	w := fixture()
	w.target = 1
	bot := put(w, 1, Vec2{}, 0)
	put(w, 2, Vec2{20, 0}, 0)
	if _, prey := w.attackGoal(bot); prey != 0 {
		t.Fatal("bot chose an unwinnable chase")
	}
}
func TestBotsGrowOverTime(t *testing.T) {
	w := fixture()
	w.target = 1
	w.rng = rand.New(rand.NewSource(42))
	w.pelletTarget = 400
	w.replenish()
	bot := put(w, 1, Vec2{}, 0)
	bot.Bot = true
	for i := 0; i < 600; i++ {
		w.Tick()
	}
	if !bot.Alive || bot.Score < 45 {
		t.Fatalf("bot failed to forage: alive=%v score=%.1f", bot.Alive, bot.Score)
	}
}

func TestBotTurnsAwayFromBorder(t *testing.T) {
	w := fixture()
	w.target = 1
	bot := put(w, 1, Vec2{88, 0}, 0)
	bot.Bot = true
	w.pellets = []Pellet{{Vec2{99, 0}, 5}, {Vec2{65, 15}, 1}}
	for i := 0; i < 100; i++ {
		w.Tick()
	}
	if !bot.Alive {
		t.Fatal("bot pursued food through the arena border")
	}
}

func BenchmarkBotArenaTick(b *testing.B) {
	w := NewWorld()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.Tick()
	}
}

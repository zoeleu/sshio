package game

import "math"

type Vec2 struct{ X, Y float64 }

func (v Vec2) Add(w Vec2) Vec2      { return Vec2{v.X + w.X, v.Y + w.Y} }
func (v Vec2) Scale(s float64) Vec2 { return Vec2{v.X * s, v.Y * s} }
func (v Vec2) Sub(w Vec2) Vec2      { return v.Add(w.Scale(-1)) }
func (v Vec2) Len() float64         { return math.Hypot(v.X, v.Y) }
func direction(a float64) Vec2      { return Vec2{math.Cos(a), math.Sin(a)} }
func turn(a, b, limit float64) float64 {
	d := math.Atan2(math.Sin(b-a), math.Cos(b-a))
	return a + math.Max(-limit, math.Min(limit, d))
}

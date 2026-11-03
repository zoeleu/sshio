package game

import "math"

type cell struct{ x, y int }
type bodyPoint struct {
	pos Vec2
	id  int
}
type grid map[cell][]bodyPoint

func key(v Vec2) cell          { return cell{int(math.Floor(v.X / 4)), int(math.Floor(v.Y / 4))} }
func (g grid) add(p bodyPoint) { k := key(p.pos); g[k] = append(g[k], p) }
func (g grid) near(p Vec2, radius float64, visit func(bodyPoint)) {
	k := key(p)
	n := int(math.Ceil(radius / 4))
	for x := -n; x <= n; x++ {
		for y := -n; y <= n; y++ {
			for _, b := range g[cell{k.x + x, k.y + y}] {
				visit(b)
			}
		}
	}
}

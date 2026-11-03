package ui

import (
	"fmt"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/termenv"
	"math"
	"strconv"
	"strings"
	"wriggle/game"
)

var renderer = func() *lipgloss.Renderer {
	r := lipgloss.NewRenderer(nil)
	r.SetColorProfile(termenv.ANSI256)
	return r
}()

func style(c int) lipgloss.Style {
	return renderer.NewStyle().Foreground(lipgloss.Color(strconv.Itoa(c)))
}

type shadeStops struct {
	rim, dark, mid, base, spec int
}

// shadePalette maps each ANSI-256 base snake color to a 5-stop 3D cylindrical gradient
// (outer rim shadow -> shaded flank -> midtone -> lit body -> top-left specular highlight).
func shadePalette(base int) shadeStops {
	switch base {
	case 45: // Cyan
		return shadeStops{24, 31, 38, 45, 159}
	case 81: // Sky Blue
		return shadeStops{25, 39, 75, 81, 195}
	case 213: // Pink / Magenta
		return shadeStops{90, 133, 176, 213, 225}
	case 207: // Orchid / Purple
		return shadeStops{89, 127, 169, 207, 225}
	case 118: // Neon Green
		return shadeStops{22, 34, 76, 118, 193}
	case 220: // Gold / Yellow
		return shadeStops{94, 136, 178, 220, 229}
	case 209: // Coral / Orange
		return shadeStops{88, 130, 166, 209, 223}
	case 226: // Electric Yellow (Boost)
		return shadeStops{136, 178, 220, 226, 231}
	case 231: // Bright White (Boost)
		return shadeStops{244, 249, 253, 231, 231}
	default:
		return shadeStops{238, base, base, base, 231}
	}
}

func (m *Model) View() string {
	if m.width < 30 || m.height < 10 {
		return "Wriggle — resize to at least 30 × 10"
	}
	if m.state == 0 {
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, style(118).Bold(true).Render("W R I G G L E")+"\n\nA shared snake arena over SSH\n\nName: "+m.name+"▏\n\nEnter to join • Tab to spectate bots • Ctrl+C to quit\nArrows / WASD • mouse to steer • Space to boost")
	}
	var me game.Snake
	foundMe := false
	for _, s := range m.frame.Snakes {
		if s.ID == m.id {
			me = s
			foundMe = true
		}
	}
	var target game.Snake
	foundTarget := false
	if m.state == 3 {
		for _, s := range m.frame.Snakes {
			if s.ID == m.spectateID && s.Alive && len(s.Segments) > 0 {
				target = s
				foundTarget = true
				break
			}
		}
		if !foundTarget {
			for _, s := range m.frame.Snakes {
				if s.Alive && len(s.Segments) > 0 {
					target = s
					foundTarget = true
					break
				}
			}
		}
	}
	if m.state == 1 && (!foundMe || len(me.Segments) == 0) {
		return "Entering the arena…"
	}
	if m.state == 3 && (!foundTarget || len(target.Segments) == 0) {
		return "Waiting for snakes in arena…"
	}
	w, h := m.width, m.height-2
	pixels := make([]int, w*h*2)
	center := m.deathPos
	if foundMe && len(me.Segments) > 0 {
		center = me.Segments[0]
	} else if m.state == 3 && foundTarget && len(target.Segments) > 0 {
		center = target.Segments[0]
	}
	origin := center.Sub(game.Vec2{X: float64(w) / 2, Y: float64(h)})
	for y := 0; y < h*2; y++ {
		for x := 0; x < w; x++ {
			p := origin.Add(game.Vec2{X: float64(x), Y: float64(y)})
			c := 16
			if int(math.Floor(p.X))%8 == 0 && int(math.Floor(p.Y))%8 == 0 {
				c = 234
			}
			if math.Abs(p.Len()-m.frame.Radius) < 1 {
				c = 240
			}
			if p.Len() > m.frame.Radius+1 {
				c = 233
			}
			pixels[y*w+x] = c
		}
	}
	shadedSeg := func(p game.Vec2, baseColor int, r float64, bandDark bool, boostGlow bool) {
		cx, cy := p.X-origin.X, p.Y-origin.Y
		x0, y0 := int(math.Floor(cx-r-1)), int(math.Floor(cy-r-1))
		x1, y1 := int(math.Ceil(cx+r+1)), int(math.Ceil(cy+r+1))
		pal := shadePalette(baseColor)
		effR := math.Max(0.95, r)
		// Offset light source slightly toward top-left for 2.5D spherical/tubular volume
		lx, ly := cx-0.28*effR, cy-0.32*effR
		for yy := y0; yy <= y1; yy++ {
			if yy < 0 || yy >= h*2 {
				continue
			}
			for xx := x0; xx <= x1; xx++ {
				if xx < 0 || xx >= w {
					continue
				}
				dx, dy := float64(xx)-cx, float64(yy)-cy
				dist := math.Hypot(dx, dy)
				if dist > effR {
					// Subtle outer aura when boosting
					if boostGlow && dist <= effR+0.85 && (pixels[yy*w+xx] == 16 || pixels[yy*w+xx] == 234 || pixels[yy*w+xx] == 232) {
						pixels[yy*w+xx] = pal.dark
					}
					continue
				}
				ldist := math.Hypot(float64(xx)-lx, float64(yy)-ly) / effR
				edge := dist / effR
				var c int
				if edge > 0.82 {
					c = pal.rim // Dark outer rim creates Slither.io overlapping segment creases
				} else if ldist < 0.36 && effR >= 1.15 {
					c = pal.spec // Specular ridge highlight
				} else if ldist < 0.72 {
					if bandDark {
						c = pal.mid
					} else {
						c = pal.base
					}
				} else if ldist < 1.05 {
					if bandDark {
						c = pal.dark
					} else {
						c = pal.mid
					}
				} else {
					c = pal.dark
				}
				pixels[yy*w+xx] = c
			}
		}
	}
	pelletColors := [...]int{45, 81, 213, 207, 118, 220, 209}
	for _, p := range m.frame.Pellets {
		// Deterministic spatial hash so each pellet keeps a stable neon color and twinkle phase
		hx := int(math.Round(p.Pos.X*3.7)) * 73856093
		hy := int(math.Round(p.Pos.Y*3.7)) * 19349663
		hash := hx ^ hy
		if hash < 0 {
			hash = -hash
		}
		baseColor := pelletColors[hash%len(pelletColors)]
		pal := shadePalette(baseColor)
		phase := (int(m.frame.Tick) + hash%16) % 16

		// Gentle sub-pixel floating wobble so the field of orbs feels alive
		wobbleAng := float64(m.frame.Tick)*0.18 + float64(hash%31)*0.2
		wx := p.Pos.X + math.Cos(wobbleAng)*0.28
		wy := p.Pos.Y + math.Sin(wobbleAng)*0.28
		cx := int(math.Round(wx - origin.X))
		cy := int(math.Round(wy - origin.Y))
		if cx < -2 || cx >= w+2 || cy < -2 || cy >= h*2+2 {
			continue
		}

		if p.Value < 0.8 {
			// Boost trail sparks: compact flickering embers
			c := pal.mid
			if phase < 4 {
				c = pal.base
			} else if phase >= 12 {
				c = pal.dark
			}
			if cx >= 0 && cx < w && cy >= 0 && cy < h*2 {
				pixels[cy*w+cx] = c
			}
		} else if p.Value < 1.8 {
			// Ambient food orbs: neon core with out-of-phase specular twinkle and soft cross halo
			if phase < 6 {
				for _, off := range [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
					xx, yy := cx+off[0], cy+off[1]
					if xx >= 0 && xx < w && yy >= 0 && yy < h*2 && (pixels[yy*w+xx] == 16 || pixels[yy*w+xx] == 234) {
						pixels[yy*w+xx] = pal.rim
					}
				}
			}
			if cx >= 0 && cx < w && cy >= 0 && cy < h*2 {
				core := pal.base
				if phase < 3 {
					core = pal.spec
				} else if phase >= 13 {
					core = pal.mid
				}
				pixels[cy*w+cx] = core
			}
		} else {
			// Rich corpse orbs (Value >= 2): large multi-layered glowing energy orbs
			outerR := 1.45
			if phase < 5 {
				outerR = 1.8
			}
			for dy := -2; dy <= 2; dy++ {
				for dx := -2; dx <= 2; dx++ {
					dist := math.Hypot(float64(dx), float64(dy))
					if dist > outerR {
						continue
					}
					xx, yy := cx+dx, cy+dy
					if xx < 0 || xx >= w || yy < 0 || yy >= h*2 {
						continue
					}
					if dist == 0 {
						if phase%4 == 0 {
							pixels[yy*w+xx] = 231 // Bright white sparkle
						} else {
							pixels[yy*w+xx] = pal.spec
						}
					} else if dist <= 1.05 {
						if phase < 4 {
							pixels[yy*w+xx] = pal.spec
						} else {
							pixels[yy*w+xx] = pal.base
						}
					} else if pixels[yy*w+xx] == 16 || pixels[yy*w+xx] == 234 {
						pixels[yy*w+xx] = pal.dark
					}
				}
			}
		}
	}
	alive, rank := 0, 0
	for _, s := range m.frame.Snakes {
		if !s.Alive || len(s.Segments) == 0 {
			continue
		}
		alive++
		if s.ID == m.id {
			rank = alive
		}
		isBoosting := s.Boosting(m.frame.Tick)
		// Base radius starts at 1.05 so even starter worms show 3D shading and rim separation, scaling up to 4x
		segRadius := math.Max(1.05, 0.65*s.WidthScale())
		// Pass 1: Soft drop shadow underneath the worm body to ground it against the grid
		for i := len(s.Segments) - 1; i >= 0; i-- {
			r := segRadius
			if i >= len(s.Segments)-2 && r > 0.8 {
				r *= 0.75
			}
			sx := int(math.Round(s.Segments[i].X - origin.X + 0.6))
			sy := int(math.Round(s.Segments[i].Y - origin.Y + 0.8))
			ir := max(1, int(math.Ceil(r)))
			for dy := -ir; dy <= ir; dy++ {
				for dx := -ir; dx <= ir; dx++ {
					if float64(dx*dx+dy*dy) > r*r {
						continue
					}
					xx, yy := sx+dx, sy+dy
					if xx >= 0 && xx < w && yy >= 0 && yy < h*2 && (pixels[yy*w+xx] == 16 || pixels[yy*w+xx] == 234) {
						pixels[yy*w+xx] = 232
					}
				}
			}
		}
		// Pass 2: Overlapping 2.5D shaded segments from tail to head (Painter's algorithm)
		for i := len(s.Segments) - 1; i >= 0; i-- {
			c := s.Color
			if isBoosting {
				// Travelling alternating high-contrast flash along the body while boosting
				phase := (i + int(m.frame.Tick*2)) % 4
				if phase == 0 {
					c = 231 // bright white
				} else if phase == 2 {
					c = 226 // electric yellow
				}
			}
			r := segRadius
			if i >= len(s.Segments)-2 && r > 0.8 {
				r *= 0.75
			}
			bandDark := i%2 == 1 && !isBoosting
			shadedSeg(s.Segments[i], c, r, bandDark, isBoosting)
		}
		// Pass 3: Slither.io style eyes on the head segment oriented along s.Heading
		head := s.Segments[0]
		hx, hy := head.X-origin.X, head.Y-origin.Y
		eyeDist := math.Max(0.75, segRadius*0.58)
		eyeAngle := 0.68
		lookX, lookY := math.Cos(s.TargetHeading)*0.45, math.Sin(s.TargetHeading)*0.45
		for _, side := range []float64{-1, 1} {
			ang := s.Heading + side*eyeAngle
			ex := int(math.Round(hx + math.Cos(ang)*eyeDist))
			ey := int(math.Round(hy + math.Sin(ang)*eyeDist))
			if ex >= 0 && ex < w && ey >= 0 && ey < h*2 {
				pixels[ey*w+ex] = 231 // White sclera
			}
			if segRadius >= 1.5 {
				px := int(math.Round(hx + math.Cos(ang)*eyeDist + lookX))
				py := int(math.Round(hy + math.Sin(ang)*eyeDist + lookY))
				if px >= 0 && px < w && py >= 0 && py < h*2 {
					pixels[py*w+px] = 16 // Dark pupil looking toward TargetHeading
				}
			}
		}
	}
	// Corner minimap (bottom-right) showing the arena boundary, rich mass clusters, and all worms
	if w >= 50 && h >= 16 && m.frame.Radius > 0 {
		mapR := math.Min(13, math.Min(float64(w)/6, float64(h)/2-2))
		if mapR >= 6 {
			cx := float64(w) - mapR - 3
			cy := float64(h*2) - mapR - 2
			ir := int(math.Ceil(mapR))
			for dy := -ir; dy <= ir; dy++ {
				for dx := -ir; dx <= ir; dx++ {
					dist := math.Hypot(float64(dx), float64(dy))
					if dist > mapR+0.5 {
						continue
					}
					px, py := int(math.Round(cx))+dx, int(math.Round(cy))+dy
					if px < 0 || px >= w || py < 0 || py >= h*2 {
						continue
					}
					if math.Abs(dist-mapR) <= 0.75 {
						pixels[py*w+px] = 242 // minimap border ring
					} else if (dx == 0 && math.Abs(float64(dy)) <= 1) || (dy == 0 && math.Abs(float64(dx)) <= 1) {
						pixels[py*w+px] = 238 // subtle center crosshair
					} else {
						pixels[py*w+px] = 234 // dark radar interior
					}
				}
			}
			scale := (mapR - 1.5) / m.frame.Radius
			// Draw high-value corpse mass on the minimap
			for _, p := range m.frame.Pellets {
				if p.Value < 2 {
					continue
				}
				px := int(math.Round(cx + p.Pos.X*scale))
				py := int(math.Round(cy + p.Pos.Y*scale))
				if px >= 0 && px < w && py >= 0 && py < h*2 {
					pixels[py*w+px] = 178
				}
			}
			// Draw all alive snakes on the minimap
			for _, s := range m.frame.Snakes {
				if !s.Alive || len(s.Segments) == 0 {
					continue
				}
				// Draw a couple of body segments for large snakes so their territory is visible
				step := max(1, len(s.Segments)/4)
				for i := len(s.Segments) - 1; i >= 0; i -= step {
					seg := s.Segments[i]
					px := int(math.Round(cx + seg.X*scale))
					py := int(math.Round(cy + seg.Y*scale))
					if px >= 0 && px < w && py >= 0 && py < h*2 {
						pixels[py*w+px] = s.Color
					}
				}
				head := s.Segments[0]
				hx := int(math.Round(cx + head.X*scale))
				hy := int(math.Round(cy + head.Y*scale))
				if s.ID == m.id || (m.state == 3 && s.ID == target.ID) {
					// Highlight the player or spectated bot with a bright marker
					for dy := -1; dy <= 1; dy++ {
						for dx := -1; dx <= 1; dx++ {
							if dx*dx+dy*dy > 1 {
								continue
							}
							px, py := hx+dx, hy+dy
							if px >= 0 && px < w && py >= 0 && py < h*2 {
								if dx == 0 && dy == 0 {
									pixels[py*w+px] = 231
								} else {
									pixels[py*w+px] = 118
								}
							}
						}
					}
				} else if hx >= 0 && hx < w && hy >= 0 && hy < h*2 {
					c := s.Color
					if s.Boosting(m.frame.Tick) && m.frame.Tick%2 == 0 {
						c = 231
					}
					pixels[hy*w+hx] = c
				}
			}
		}
	}
	board := []string{"  WRIGGLE / TOP 10"}
	for _, s := range m.frame.Snakes {
		if s.Alive && len(board) <= 10 {
			mark := " "
			if s.ID == m.id || (m.state == 3 && s.ID == target.ID) {
				mark = ">"
			}
			label := s.Name
			if tag := s.PersonalityTag(); tag != "" {
				label = fmt.Sprintf("%s·%s", ansi.Truncate(s.Name, 8, ""), tag)
			}
			trunc := ansi.Truncate(label, 12, "")
			board = append(board, fmt.Sprintf("%s %s%s %4d", mark, trunc, strings.Repeat(" ", max(0, 12-lipgloss.Width(trunc))), s.Length()))
		}
	}
	// Optional centered death modal overlaid on top of the live background arena
	modalW, modalH := 36, 7
	modalX0, modalY0 := (w-modalW)/2, (h-modalH)/2
	var modalLines []string
	if m.state == 2 {
		killer := me.Killer
		if killer == "" {
			killer = m.deathKiller
		}
		finalLen := m.deathLength
		if foundMe {
			finalLen = me.Length()
		}
		modalLines = []string{
			"",
			"YOU UNRAVELED",
			"",
			ansi.Truncate(fmt.Sprintf("You were eaten by %s", killer), modalW-2, ""),
			fmt.Sprintf("Final length %d", finalLen),
			"Enter to respawn • q to quit",
			"",
		}
	}
	var b strings.Builder
	for y := 0; y < h; y++ {
		for x := 0; x < w; {
			if m.state == 2 && y >= modalY0 && y < modalY0+modalH && x == modalX0 && modalX0 >= 0 && modalX0+modalW <= w {
				line := modalLines[y-modalY0]
				padLeft := max(0, (modalW-lipgloss.Width(line))/2)
				padRight := max(0, modalW-lipgloss.Width(line)-padLeft)
				padded := strings.Repeat(" ", padLeft) + line + strings.Repeat(" ", padRight)
				rowStyle := style(252).Background(lipgloss.Color("235"))
				if y-modalY0 == 1 {
					rowStyle = style(209).Background(lipgloss.Color("235")).Bold(true)
				}
				b.WriteString(rowStyle.Render(padded))
				x += modalW
				continue
			}
			if w >= 65 && y < len(board) && x == w-22 {
				text := ansi.Truncate(board[y], 22, "")
				text += strings.Repeat(" ", max(0, 22-lipgloss.Width(text)))
				b.WriteString(style(252).Background(lipgloss.Color("235")).Render(text))
				break
			}
			top, bot := pixels[2*y*w+x], pixels[(2*y+1)*w+x]
			end := x + 1
			for end < w && pixels[2*y*w+end] == top && pixels[(2*y+1)*w+end] == bot &&
				!(w >= 65 && y < len(board) && end == w-22) &&
				!(m.state == 2 && y >= modalY0 && y < modalY0+modalH && end == modalX0 && modalX0 >= 0 && modalX0+modalW <= w) {
				end++
			}
			b.WriteString(style(top).Background(lipgloss.Color(strconv.Itoa(bot))).Render(strings.Repeat("▀", end-x)))
			x = end
		}
		b.WriteByte('\n')
	}
	myLen := m.deathLength
	if foundMe {
		myLen = me.Length()
	}
	hud := fmt.Sprintf(" %s  LENGTH %d  RANK %d  SNAKES %d", m.name, myLen, rank, alive)
	if m.state == 2 {
		hud = fmt.Sprintf(" %s  LENGTH %d (UNRAVELED)  SNAKES %d", m.name, myLen, alive)
	} else if m.state == 3 {
		targetName := target.Name
		if tag := target.PersonalityTag(); tag != "" {
			targetName = fmt.Sprintf("%s [%s]", target.Name, tag)
		}
		targetRank := 0
		for idx, s := range m.frame.Snakes {
			if s.Alive && s.ID == target.ID {
				targetRank = idx + 1
				break
			}
		}
		hud = fmt.Sprintf(" 👁 SPECTATING %s  LENGTH %d  RANK %d  SNAKES %d", targetName, target.Length(), targetRank, alive)
	}
	b.WriteString(style(118).Render(lipgloss.NewStyle().MaxWidth(w).Render(hud)))
	b.WriteByte('\n')
	controls := " Arrows/WASD or mouse • Space/click boost • Tab spectate • q quit"
	if m.state == 3 {
		controls = " Tab/n next bot • p prev • r random • Enter join & play • q quit"
	}
	b.WriteString(style(244).Render(lipgloss.NewStyle().MaxWidth(w).Render(controls)))
	return b.String()
}

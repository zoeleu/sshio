use super::*;
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Clear, Paragraph},
};
use sshio_core::{Point, RADIUS};
use std::io::{self, Write};
const COLORS: [(u8, u8, u8); 6] = [
    (63, 225, 205),
    (255, 128, 165),
    (171, 137, 255),
    (255, 202, 83),
    (95, 180, 255),
    (160, 230, 92),
];
#[derive(Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn rgb(c: (u8, u8, u8)) -> Color {
    Color::Rgb(c.0, c.1, c.2)
}
fn shade(c: (u8, u8, u8), f: f32) -> (u8, u8, u8) {
    (
        (c.0 as f32 * f).min(255.0) as u8,
        (c.1 as f32 * f).min(255.0) as u8,
        (c.2 as f32 * f).min(255.0) as u8,
    )
}
pub(super) async fn run(
    arena: Shared,
    ui: Arc<Mutex<Ui>>,
    handle: server::Handle,
    ch: ChannelId,
) -> anyhow::Result<()> {
    let mut size = (80, 24);
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = Terminal::with_options(
        CrosstermBackend::new(Sink(output.clone())),
        TerminalOptions {
            viewport: Viewport::Fixed(Rect::new(0, 0, size.0, size.1)),
        },
    )?;
    let mut frames = tokio::time::interval(Duration::from_millis(100));
    frames.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        frames.tick().await;
        {
            let a = arena.lock().unwrap();
            let mut u = ui.lock().unwrap();
            if size != (u.width, u.height) {
                size = (u.width, u.height);
                terminal.resize(Rect::new(0, 0, size.0, size.1))?;
                terminal.clear()?;
            }
            terminal.draw(|f| {
                let w = size.0 as usize;
                let h = (size.1 - 2) as usize * 2;
                let mut pixels = vec![(7, 11, 19); w * h];
                let player = u.player.and_then(|id| a.world.snakes.get(&id));
                if let Some(s) = player {
                    u.last_position = s.head();
                }
                let center = if u.player.is_some() {
                    u.last_position
                } else {
                    a.world
                        .snakes
                        .values()
                        .next()
                        .map(|s| s.head())
                        .unwrap_or_default()
                };
                // Fixed 320x220 world viewport independent of terminal dimensions.
                let project = |p: Point| -> (f32, f32) {
                    (
                        (p.x - center.x) / 320.0 * w as f32 + w as f32 / 2.0,
                        (p.y - center.y) / 220.0 * h as f32 + h as f32 / 2.0,
                    )
                };
                for y in 0..h {
                    for x in 0..w {
                        let p = Point {
                            x: center.x + (x as f32 / w as f32 - 0.5) * 320.0,
                            y: center.y + (y as f32 / h as f32 - 0.5) * 220.0,
                        };
                        if p.norm() > RADIUS {
                            pixels[y * w + x] = (27, 15, 31);
                        } else if (p.norm() - RADIUS).abs() < 4.0 {
                            pixels[y * w + x] = (165, 62, 95);
                        } else if p.x.rem_euclid(24.0) < 320.0 / w as f32
                            && p.y.rem_euclid(24.0) < 220.0 / h as f32
                        {
                            pixels[y * w + x] = (26, 38, 52);
                        }
                    }
                }
                let mut disk = |p: Point, r: f32, c: (u8, u8, u8)| {
                    let (cx, cy) = project(p);
                    let rx = (r / 320.0 * w as f32).max(0.6);
                    let ry = (r / 220.0 * h as f32).max(0.6);
                    for y in ((cy - ry).floor() as i32).max(0)
                        ..=((cy + ry).ceil() as i32).min(h as i32 - 1)
                    {
                        for x in ((cx - rx).floor() as i32).max(0)
                            ..=((cx + rx).ceil() as i32).min(w as i32 - 1)
                        {
                            if ((x as f32 - cx) / rx).powi(2) + ((y as f32 - cy) / ry).powi(2)
                                <= 1.0
                            {
                                // Wriggle reference: a top-left highlight and dark rim
                                // make overlapping half-block segments read as tubes.
                                let dx = (x as f32 - cx) / rx;
                                let dy = (y as f32 - cy) / ry;
                                let edge = dx * dx + dy * dy;
                                let light = if edge > 0.82 {
                                    0.55
                                } else {
                                    (0.85 - 0.2 * dx - 0.25 * dy).clamp(0.5, 1.25)
                                };
                                pixels[y as usize * w + x as usize] = shade(c, light);
                            }
                        }
                    }
                };
                for food in &a.world.food {
                    disk(
                        food.pos,
                        if food.mass >= 12 {
                            3.0
                        } else if food.mass >= 4 {
                            2.0
                        } else {
                            1.0
                        },
                        COLORS[food.color],
                    );
                }
                for s in a.world.snakes.values() {
                    let mut arc = 0.0;
                    for i in (0..s.body.len()).rev() {
                        if i + 1 < s.body.len() {
                            arc += s.body[i].distance(s.body[i + 1]);
                        }
                        let stripe = (arc / 7.0) as u64;
                        let phase = if s.boosting { a.world.tick / 6 } else { 0 };
                        let c = shade(
                            COLORS[s.color],
                            if (stripe + phase).is_multiple_of(2) {
                                1.0
                            } else {
                                0.52
                            },
                        );
                        disk(s.body[i], s.width(), c);
                    }
                    disk(s.head(), s.width(), shade(COLORS[s.color], 1.25));
                    // Heading-oriented eyes, inspired by Wriggle's terminal renderer.
                    for side in [-1.0, 1.0] {
                        let angle = s.heading + side * 0.68;
                        let eye = Point {
                            x: s.head().x + angle.cos() * s.width() * 0.65,
                            y: s.head().y + angle.sin() * s.width() * 0.65,
                        };
                        disk(eye, 1.05, (245, 250, 255));
                        if s.width() > 4.0 {
                            disk(
                                Point {
                                    x: eye.x + s.heading.cos() * 0.65,
                                    y: eye.y + s.heading.sin() * 0.65,
                                },
                                0.5,
                                (7, 11, 19),
                            );
                        }
                    }
                }
                let buf = f.buffer_mut();
                for y in 0..h / 2 {
                    for x in 0..w {
                        buf[(x as u16, y as u16)]
                            .set_symbol("▀")
                            .set_fg(rgb(pixels[(y * 2) * w + x]))
                            .set_bg(rgb(pixels[(y * 2 + 1) * w + x]));
                    }
                }
                let status = if let Some(s) = player {
                    format!(
                        " sshio | {} | mass {} | boost {} | {}",
                        s.name,
                        s.mass,
                        if s.action.boost { "ON" } else { "off" },
                        if s.crown_at.is_some() {
                            "CROWN: finite reign"
                        } else {
                            "grow, hunt, survive"
                        }
                    )
                } else {
                    format!(
                        " sshio | spectating | {} snakes | Enter to play",
                        a.world.snakes.len()
                    )
                };
                f.render_widget(
                    Paragraph::new(status)
                        .style(Style::default().fg(Color::White).bg(rgb((12, 18, 29)))),
                    Rect::new(0, size.1 - 2, size.0, 1),
                );
                f.render_widget(
                    Paragraph::new(" WASD/arrows steer | Space boost | v spectate | q quit")
                        .style(Style::default().fg(Color::Gray)),
                    Rect::new(0, size.1 - 1, size.0, 1),
                );
                if u.joined && u.player.is_some() && player.is_none() {
                    let width = 38.min(size.0);
                    let rect = Rect::new((size.0 - width) / 2, (size.1 - 5) / 2, width, 5);
                    f.render_widget(Clear, rect);
                    f.render_widget(
                        Paragraph::new("Your trail ends here.\nEnter to respawn | v to spectate")
                            .block(Block::default().title(" UNRAVELED ").borders(Borders::ALL))
                            .style(Style::default().fg(Color::White).bg(rgb((12, 18, 29)))),
                        rect,
                    );
                }
                if !u.joined {
                    let width = 38.min(size.0);
                    let rect = Rect::new((size.0 - width) / 2, (size.1 - 9) / 2, width, 9);
                    f.render_widget(Clear, rect);
                    let lines = vec![
                        Line::from("Enter the arena"),
                        Line::from(format!("Name: {}_", u.name)),
                        Line::from(format!("Color: {} (Tab to change)", u.color + 1)),
                        Line::from("Enter: play   Ctrl-S: spectate"),
                        Line::from("WASD/arrows + Space to boost"),
                    ];
                    f.render_widget(
                        Paragraph::new(lines)
                            .block(Block::default().title(" sshio ").borders(Borders::ALL))
                            .style(
                                Style::default()
                                    .fg(rgb(COLORS[u.color]))
                                    .bg(rgb((12, 18, 29))),
                            ),
                        rect,
                    );
                }
            })?;
        }
        let data = std::mem::take(&mut *output.lock().unwrap());
        if !data.is_empty() {
            tokio::time::timeout(Duration::from_secs(2), handle.data(ch, data))
                .await?
                .map_err(|_| anyhow::anyhow!("SSH channel closed"))?;
        }
    }
}

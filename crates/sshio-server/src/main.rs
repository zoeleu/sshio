mod render;
use russh::{
    Channel, ChannelId, Pty,
    server::{self, Auth, Handler, Msg, Server as _, Session},
};
use sshio_core::World;
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::task::JoinHandle;

type Shared = Arc<Mutex<Arena>>;
struct Arena {
    world: World,
    bots: BTreeSet<u64>,
    clients: usize,
}
#[derive(Default)]
struct Ui {
    name: String,
    color: usize,
    player: Option<u64>,
    joined: bool,
    width: u16,
    height: u16,
    escape: u8,
    last_position: sshio_core::Point,
}
struct Gateway {
    arena: Shared,
}
struct Client {
    arena: Shared,
    ui: Arc<Mutex<Ui>>,
    channel: Option<ChannelId>,
    render: Option<JoinHandle<()>>,
    admitted: bool,
}
impl server::Server for Gateway {
    type Handler = Client;
    fn new_client(&mut self, _: Option<std::net::SocketAddr>) -> Client {
        let mut a = self.arena.lock().unwrap();
        let admitted = a.clients < 16;
        if admitted {
            a.clients += 1;
        }
        Client {
            arena: self.arena.clone(),
            ui: Arc::new(Mutex::new(Ui {
                width: 80,
                height: 24,
                ..Ui::default()
            })),
            channel: None,
            render: None,
            admitted,
        }
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        if let Some(t) = self.render.take() {
            t.abort();
        }
        let mut a = self.arena.lock().unwrap();
        if self.admitted {
            a.clients -= 1;
        }
        if let Some(id) = self.ui.lock().unwrap().player {
            a.world.remove(id);
        }
    }
}
impl Handler for Client {
    type Error = anyhow::Error;
    async fn auth_none(&mut self, _: &str) -> Result<Auth, Self::Error> {
        Ok(if self.admitted {
            Auth::Accept
        } else {
            Auth::Reject {
                proceed_with_methods: None,
                partial_success: false,
            }
        })
    }
    async fn auth_publickey(
        &mut self,
        _: &str,
        _: server::VerifiedPublicKey<'_>,
    ) -> Result<Auth, Self::Error> {
        self.auth_none("").await
    }
    async fn channel_open_session(
        &mut self,
        channel: Channel<Msg>,
        reply: server::ChannelOpenHandle,
        _: &mut Session,
    ) -> Result<(), Self::Error> {
        if self.channel.is_some() || !self.admitted {
            reply
                .reject(russh::ChannelOpenFailure::AdministrativelyProhibited)
                .await;
        } else {
            self.channel = Some(channel.id());
            reply.accept().await;
        }
        Ok(())
    }
    async fn pty_request(
        &mut self,
        ch: ChannelId,
        _: &str,
        w: u32,
        h: u32,
        _: u32,
        _: u32,
        _: &[(Pty, u32)],
        s: &mut Session,
    ) -> Result<(), Self::Error> {
        self.resize(w, h);
        s.channel_success(ch)?;
        Ok(())
    }
    async fn window_change_request(
        &mut self,
        _: ChannelId,
        w: u32,
        h: u32,
        _: u32,
        _: u32,
        _: &mut Session,
    ) -> Result<(), Self::Error> {
        self.resize(w, h);
        Ok(())
    }
    async fn shell_request(&mut self, ch: ChannelId, s: &mut Session) -> Result<(), Self::Error> {
        if self.render.is_some() {
            s.channel_failure(ch)?;
            return Ok(());
        }
        s.channel_success(ch)?;
        s.data(ch, b"\x1b[?1049h\x1b[?25l".to_vec())?;
        let arena = self.arena.clone();
        let ui = self.ui.clone();
        let handle = s.handle();
        self.render = Some(tokio::spawn(async move {
            let result = render::run(arena, ui, handle.clone(), ch).await;
            if let Err(e) = result {
                eprintln!("render session ended: {e}");
            }
            let _ = handle.close(ch).await;
        }));
        Ok(())
    }
    async fn exec_request(
        &mut self,
        ch: ChannelId,
        _: &[u8],
        s: &mut Session,
    ) -> Result<(), Self::Error> {
        s.channel_failure(ch)?;
        s.close(ch)?;
        Ok(())
    }
    async fn subsystem_request(
        &mut self,
        ch: ChannelId,
        _: &str,
        s: &mut Session,
    ) -> Result<(), Self::Error> {
        s.channel_failure(ch)?;
        s.close(ch)?;
        Ok(())
    }
    async fn data(
        &mut self,
        ch: ChannelId,
        data: &[u8],
        s: &mut Session,
    ) -> Result<(), Self::Error> {
        let mut a = self.arena.lock().unwrap();
        let mut ui = self.ui.lock().unwrap();
        // Bounded ASCII/CSI streaming parser: handles fragmented arrows and pasted input.
        for &b in data.iter().take(4096) {
            if b == 3 || b == 4 {
                s.data(ch, b"\x1b[0m\x1b[?25h\x1b[?1049l".to_vec())?;
                s.close(ch)?;
                break;
            }
            let key = match (ui.escape, b) {
                (_, 27) => {
                    ui.escape = 1;
                    continue;
                }
                (1, b'[' | b'O') => {
                    ui.escape = 2;
                    continue;
                }
                (2, b'A') => b'w',
                (2, b'B') => b's',
                (2, b'C') => b'd',
                (2, b'D') => b'a',
                (2, 0x30..=0x3f) => continue,
                (1 | 2, _) => {
                    ui.escape = 0;
                    continue;
                }
                _ => b,
            };
            ui.escape = 0;
            if !ui.joined {
                match key {
                    b'\t' => ui.color = (ui.color + 1) % 6,
                    b'\r' | b'\n' => {
                        let name = if ui.name.is_empty() {
                            "guest"
                        } else {
                            &ui.name
                        };
                        ui.player = a.world.spawn(name, ui.color);
                        ui.joined = ui.player.is_some();
                    }
                    b'\x13' => ui.joined = true,
                    127 | 8 => {
                        ui.name.pop();
                    }
                    32..=126 if ui.name.len() < 16 => ui.name.push(key as char),
                    _ => {}
                }
            } else if key == b'q' {
                s.data(ch, b"\x1b[0m\x1b[?25h\x1b[?1049l".to_vec())?;
                s.close(ch)?;
                break;
            } else if key == b'\r' && ui.player.is_none_or(|id| !a.world.snakes.contains_key(&id)) {
                ui.player = a.world.spawn(
                    if ui.name.is_empty() {
                        "guest"
                    } else {
                        &ui.name
                    },
                    ui.color,
                );
            } else if key == b'v' {
                if let Some(id) = ui.player.take() {
                    a.world.remove(id);
                }
            } else if let Some(id) = ui.player
                && let Some(snake) = a.world.snakes.get(&id)
            {
                let mut act = snake.action;
                match key {
                    b'w' => act.heading = -std::f32::consts::FRAC_PI_2,
                    b's' => act.heading = std::f32::consts::FRAC_PI_2,
                    b'a' => act.heading = std::f32::consts::PI,
                    b'd' => act.heading = 0.0,
                    b' ' => act.boost = !act.boost,
                    _ => {}
                }
                a.world.action(id, act);
            }
        }
        Ok(())
    }
}
impl Client {
    fn resize(&self, w: u32, h: u32) {
        let mut ui = self.ui.lock().unwrap();
        ui.width = w.clamp(40, 160) as u16;
        ui.height = h.clamp(16, 60) as u16;
    }
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let address = std::env::var("SSHIO_BIND").unwrap_or("127.0.0.1:2222".into());
    let key = std::env::var("SSHIO_HOST_KEY").unwrap_or(".sshio/host_key".into());
    if !std::path::Path::new(&key).exists() {
        anyhow::bail!(
            "Missing host key: run mkdir -p .sshio && ssh-keygen -t ed25519 -N '' -f .sshio/host_key"
        );
    }
    let config = server::Config {
        keys: vec![russh::keys::load_secret_key(key, None)?],
        inactivity_timeout: Some(Duration::from_secs(900)),
        auth_rejection_time: Duration::from_secs(1),
        auth_rejection_time_initial: Some(Duration::ZERO),
        event_buffer_size: 2,
        window_size: 65536,
        maximum_packet_size: 32768,
        nodelay: true,
        ..Default::default()
    };
    let arena = Arc::new(Mutex::new(Arena {
        world: World::new(42),
        bots: BTreeSet::new(),
        clients: 0,
    }));
    let sim = arena.clone();
    let ticks = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_nanos(1_000_000_000 / 30));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let mut a = sim.lock().unwrap();
            let alive: BTreeSet<_> = a.world.snakes.keys().copied().collect();
            a.bots.retain(|id| alive.contains(id));
            let bot_color = (a.world.tick / 30) as usize % 6;
            if a.world.tick.is_multiple_of(30)
                && a.bots.len() < 8
                && let Some(id) = a.world.spawn("bot", bot_color)
            {
                a.bots.insert(id);
            }
            if a.world.tick.is_multiple_of(3) {
                let actions: Vec<_> = a
                    .bots
                    .iter()
                    .map(|&id| (id, a.world.heuristic(id)))
                    .collect();
                for (id, action) in actions {
                    a.world.action(id, action);
                }
            }
            a.world.step();
        }
    });
    eprintln!("sshio listening on {address}; ssh -p 2222 localhost (Ctrl-C to stop)");
    let mut gateway = Gateway { arena };
    tokio::select! {result=gateway.run_on_address(Arc::new(config),&address)=>{result?;},_=tokio::signal::ctrl_c()=>{}}
    ticks.abort();
    Ok(())
}

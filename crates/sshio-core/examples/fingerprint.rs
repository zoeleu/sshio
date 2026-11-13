use sshio_core::{Action, World};
fn main() {
    let mut w = World::new(1);
    let ids: Vec<_> = (0..8).map(|i| w.spawn("synthetic", i).unwrap()).collect();
    for id in ids {
        let heading = w.snakes[&id].heading;
        w.action(
            id,
            Action {
                heading,
                boost: false,
            },
        );
    }
    for _ in 0..3 {
        w.step();
    }
    println!("{}", w.state_hash());
}

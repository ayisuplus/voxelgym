//! Gate-level digital circuit layer (M3 triple + logic gates).
//!
//! The world IS a synchronous digital simulator with unit gate delay —
//! isomorphic to textbook gate-level event simulation:
//!
//! - `wire`: a net. power = max(adjacent source 15, adjacent max wire
//!   power - 1), recomputed same-tick by max-Dijkstra over circuit cells.
//!   Joining wires is an implicit wired-OR.
//! - `lever` / `pressure_plate`: input pins (sources of 15).
//! - `redstone_torch`: NOT gate. Input = the cell below it; output = its 4
//!   horizontal neighbors + the cell above. lit_next = !powered(below).
//!   NOT + wired-OR = NOR -> functionally complete.
//! - `repeater`: buffer + diode + unit delay. Reads the cell BEHIND
//!   (opposite its facing), drives the wire in FRONT to 15 next tick.
//! - `door` / `lamp`: outputs. Door opens / lamp lights while any
//!   6-neighbor is powered.
//!
//! Tick semantics (phase 5): all gates read the network solved from
//! CURRENT gate states, then all gate states update simultaneously —
//! so a gate's output changes exactly one tick after its input. Feedback
//! loops are therefore well-defined: cross-coupled NORs store a bit, an
//! odd ring of torches oscillates with period = 2 * ring size.

use std::collections::{BinaryHeap, HashMap};

use crate::block::*;
use crate::world::World;

const DIRS6: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Repeater facing -> output direction vector. bit encoding:
/// 0:+z, 1:-x, 2:-z, 3:+x (matches placement from agent yaw).
pub fn repeater_dir(state: u16) -> (i32, i32, i32) {
    match state & 3 {
        0 => (0, 0, 1),
        1 => (-1, 0, 0),
        2 => (0, 0, -1),
        _ => (1, 0, 0),
    }
}

pub fn is_circuit(id: u16) -> bool {
    matches!(id, WIRE | LEVER | DOOR | PRESSURE_PLATE | RTORCH | REPEATER | LAMP)
}

/// A cell that actively drives power right now (source semantics), given
/// raw block states. Wire power is NOT consulted here — wires are solved.
fn is_source_on(cell: u16) -> bool {
    let id = cell_id(cell);
    let st = cell_state(cell);
    match id {
        LEVER => st & 1 == 1,
        RTORCH => st & 1 == 1,
        REPEATER => st & 4 == 4,
        _ => false,
    }
}

/// Phase 5. Recompute when this tick's changes touch the circuit — or
/// every tick while dynamic cells exist: pressure plates (their input is
/// the agent's position) and torches/repeaters (feedback loops evolve
/// autonomously: oscillators change state with no block change).
pub fn tick_circuits(world: &mut World, dirty: &[(i32, i32, i32)]) {
    if world.circuit_cells.is_empty() {
        return;
    }
    let has_dynamic = world.circuit_cells.iter().any(|&(x, y, z)| {
        matches!(
            cell_id(world.peek_block(x, y, z)),
            PRESSURE_PLATE | RTORCH | REPEATER
        )
    });
    let relevant = has_dynamic
        || dirty.iter().any(|&(x, y, z)| {
            if world.circuit_cells.contains(&(x, y, z)) {
                return true;
            }
            DIRS6
                .iter()
                .any(|(dx, dy, dz)| world.circuit_cells.contains(&(x + dx, y + dy, z + dz)))
        });
    if !relevant {
        return;
    }
    // circuit cells may sit in never-generated chunks (peek_block would
    // report air and truncate propagation at chunk borders)
    let cells_to_load: Vec<(i32, i32, i32)> = world.circuit_cells.iter().copied().collect();
    for (x, _, z) in cells_to_load {
        world.ensure_chunk(x.div_euclid(16), z.div_euclid(16));
    }

    let amn = world.agent.aabb_min();
    let amx = world.agent.aabb_max();
    let occupied = |x: i32, y: i32, z: i32| {
        (x as f64 + 1.0) > amn[0]
            && (x as f64) < amx[0]
            && (y as f64 + 1.0) > amn[1]
            && (y as f64) < amx[1]
            && (z as f64 + 1.0) > amn[2]
            && (z as f64) < amx[2]
    };

    // ---- Phase A: seed the wire network from current gate states ----
    let mut power: HashMap<(i32, i32, i32), u8> = HashMap::new();
    let mut heap: BinaryHeap<(u8, (i32, i32, i32))> = BinaryHeap::new();
    let mut cells: Vec<(i32, i32, i32)> = world.circuit_cells.iter().copied().collect();
    cells.sort_unstable();
    let mut seed = |power: &mut HashMap<(i32, i32, i32), u8>,
                    heap: &mut BinaryHeap<(u8, (i32, i32, i32))>,
                    world: &World,
                    n: (i32, i32, i32)| {
        let nc = world.peek_block(n.0, n.1, n.2);
        if cell_id(nc) == WIRE && power.get(&n).copied().unwrap_or(0) < 15 {
            power.insert(n, 15);
            heap.push((15, n));
        }
    };
    for &(x, y, z) in &cells {
        let cell = world.peek_block(x, y, z);
        let id = cell_id(cell);
        if id == PRESSURE_PLATE {
            // write back the plate state so voxel obs shows it
            let on = occupied(x, y, z);
            let cur = cell_state(cell) & 1;
            if cur != on as u16 {
                world.set_block(x, y, z, make_cell(PRESSURE_PLATE, on as u16));
            }
            if on {
                for d in DIRS6 {
                    seed(&mut power, &mut heap, world, (x + d.0, y + d.1, z + d.2));
                }
            }
            continue;
        }
        if !is_source_on(cell) {
            continue;
        }
        match id {
            // lever: powers every adjacent wire
            LEVER => {
                for d in DIRS6 {
                    seed(&mut power, &mut heap, world, (x + d.0, y + d.1, z + d.2));
                }
            }
            // redstone torch: outputs to horizontal neighbors + above —
            // never to the cell below (that is its input)
            RTORCH => {
                for d in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0)] {
                    seed(&mut power, &mut heap, world, (x + d.0, y + d.1, z + d.2));
                }
            }
            // repeater: drives only the wire in front (diode)
            REPEATER => {
                let d = repeater_dir(cell_state(cell));
                seed(&mut power, &mut heap, world, (x + d.0, y + d.1, z + d.2));
            }
            _ => {}
        }
    }

    // max-Dijkstra: propagate power-1 through wires
    while let Some((p, c)) = heap.pop() {
        if power.get(&c).copied().unwrap_or(0) > p {
            continue; // stale entry
        }
        if p <= 1 {
            continue;
        }
        for (dx, dy, dz) in DIRS6 {
            let n = (c.0 + dx, c.1 + dy, c.2 + dz);
            let nc = world.peek_block(n.0, n.1, n.2);
            if cell_id(nc) != WIRE {
                continue;
            }
            let np = p - 1;
            if power.get(&n).copied().unwrap_or(0) < np {
                power.insert(n, np);
                heap.push((np, n));
            }
        }
    }

    // apply wire power states
    for &(x, y, z) in &cells {
        let cell = world.peek_block(x, y, z);
        if cell_id(cell) != WIRE {
            continue;
        }
        let p = power.get(&(x, y, z)).copied().unwrap_or(0) as u16;
        if cell_state(cell) != p {
            world.set_block(x, y, z, make_cell(WIRE, p));
        }
    }

    // ---- Phase B: all gates read the solved network, update at once ----
    // powered(p): a wire's solved power, or a source cell's current state.
    let powered = |world: &World, p: (i32, i32, i32)| -> bool {
        let c = world.peek_block(p.0, p.1, p.2);
        if cell_id(c) == WIRE {
            return cell_state(c) > 0;
        }
        if cell_id(c) == PRESSURE_PLATE {
            return occupied(p.0, p.1, p.2);
        }
        is_source_on(c)
    };
    let mut updates: Vec<((i32, i32, i32), u16)> = Vec::new();
    for &(x, y, z) in &cells {
        let cell = world.peek_block(x, y, z);
        let id = cell_id(cell);
        let st = cell_state(cell);
        match id {
            RTORCH => {
                let lit = !powered(world, (x, y - 1, z));
                if (st & 1 == 1) != lit {
                    updates.push(((x, y, z), make_cell(RTORCH, lit as u16)));
                }
            }
            REPEATER => {
                let d = repeater_dir(st);
                let behind = (x - d.0, y - d.1, z - d.2);
                let out = powered(world, behind);
                if (st & 4 == 4) != out {
                    updates.push(((x, y, z), make_cell(REPEATER, (st & 3) | ((out as u16) << 2))));
                }
            }
            LAMP => {
                let lit = DIRS6.iter().any(|d| powered(world, (x + d.0, y + d.1, z + d.2)));
                if (st & 1 == 1) != lit {
                    updates.push(((x, y, z), make_cell(LAMP, lit as u16)));
                }
            }
            DOOR => {
                // driven iff any adjacent circuit cell that can carry power
                // (wire-free doors keep their manual `use` toggle)
                let mut driven = false;
                let mut open = false;
                for d in DIRS6 {
                    let n = (x + d.0, y + d.1, z + d.2);
                    let nc = world.peek_block(n.0, n.1, n.2);
                    let nid = cell_id(nc);
                    if matches!(nid, WIRE | LEVER | PRESSURE_PLATE | RTORCH | REPEATER) {
                        driven = true;
                        if powered(world, n) {
                            open = true;
                        }
                    }
                }
                if !driven {
                    continue;
                }
                let cur = st & 1;
                if cur != open as u16 {
                    // never close on the agent (elevator-door rule): a
                    // circuit trying to shut a doorway the agent occupies
                    // is deferred
                    if cur == 1 && !open && occupied(x, y, z) {
                        continue;
                    }
                    updates.push(((x, y, z), make_cell(DOOR, open as u16)));
                }
            }
            _ => {}
        }
    }
    for ((x, y, z), cell) in updates {
        world.set_block(x, y, z, cell);
    }
}

/// Register/unregister circuit cells on block changes (called from set_block).
pub fn on_cell_changed(world: &mut World, x: i32, y: i32, z: i32, old: u16, new: u16) {
    let old_id = cell_id(old);
    let new_id = cell_id(new);
    if is_circuit(old_id) && !is_circuit(new_id) {
        world.circuit_cells.remove(&(x, y, z));
    }
    if is_circuit(new_id) {
        world.circuit_cells.insert((x, y, z));
    }
    if block_def(new_id).fluid.is_some() {
        world.active_fluids.insert((x, y, z));
        for (dx, dy, dz) in DIRS6 {
            world.active_fluids.insert((x + dx, y + dy, z + dz));
        }
    }
    if old_id == FIRE && new_id != FIRE {
        world.active_fire.remove(&(x, y, z));
    }
    if new_id == FIRE {
        world.active_fire.insert((x, y, z));
    }
    if old_id == TNT && new_id != TNT {
        world.tnt_cells.remove(&(x, y, z));
    }
    if new_id == TNT {
        world.tnt_cells.insert((x, y, z));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::{step, Action};
    use crate::worldgen::Preset;

    fn idle() -> Action {
        Action::default()
    }

    fn flat() -> World {
        World::new(1, Preset::Flat, Vec::new())
    }

    fn power_of(w: &World, x: i32, y: i32, z: i32) -> u16 {
        cell_state(w.peek_block(x, y, z))
    }

    #[test]
    fn wire_power_decays_one_per_cell() {
        let mut w = flat();
        // lever(on) + n wires in a row: power at wire i (1-based) = 16-i
        w.set_block(2, 6, 2, make_cell(LEVER, 1)); // ON
        let n = 16;
        for i in 0..n {
            w.set_block(3 + i as i32, 6, 2, WIRE);
        }
        step(&mut w, &idle());
        for i in 0..14 {
            assert_eq!(power_of(&w, 3 + i, 6, 2), 15 - i as u16, "wire {}", i);
        }
        assert_eq!(power_of(&w, 3 + 15, 6, 2), 0, "16th wire unpowered");
    }

    #[test]
    fn lever_drives_door() {
        let mut w = flat();
        // lever at (5,6,5) OFF, wire at (6,6,5), door at (7,6,5)
        w.set_block(5, 6, 5, LEVER);
        w.set_block(6, 6, 5, WIRE);
        w.set_block(7, 6, 5, DOOR);
        step(&mut w, &idle());
        assert_eq!(cell_state(w.peek_block(7, 6, 5)) & 1, 0, "door closed");
        // flip the lever via `use`: stand at (6.5,6,5.5), face -x, look
        // 30 deg down — the ray lands on the lever cell
        w.agent.pos = [6.5, 6.0, 5.5];
        let flip = Action {
            yaw: 6, // 90 deg = -x (MC convention)
            pitch: 6,
            use_: true,
            ..Default::default()
        };
        step(&mut w, &flip);
        assert_eq!(cell_state(w.peek_block(5, 6, 5)) & 1, 1, "lever on");
        step(&mut w, &idle());
        assert!(power_of(&w, 6, 6, 5) > 0, "wire powered");
        assert_eq!(cell_state(w.peek_block(7, 6, 5)) & 1, 1, "door open");
    }

    #[test]
    fn torch_is_a_not_gate_with_unit_delay() {
        let mut w = flat();
        // input: lever -> wire (4,6,2) -> (5,6,2); torch at (5,7,2) reads
        // the cell below; output wire at (6,7,2)
        w.set_block(3, 6, 2, LEVER);
        w.set_block(4, 6, 2, WIRE);
        w.set_block(5, 6, 2, WIRE);
        w.set_block(5, 7, 2, make_cell(RTORCH, 1)); // placed lit
        w.set_block(6, 7, 2, WIRE);
        step(&mut w, &idle());
        // lever off -> torch stays lit -> output powered
        assert_eq!(cell_state(w.peek_block(5, 7, 2)) & 1, 1, "torch lit");
        assert!(power_of(&w, 6, 7, 2) > 0, "output powered when input low");
        // turn the lever ON (state write = dirty event)
        w.set_block(3, 6, 2, make_cell(LEVER, 1));
        step(&mut w, &idle());
        // unit delay: this tick's solve still saw the LIT torch, so the
        // output is still powered; the torch itself has switched off
        assert_eq!(cell_state(w.peek_block(5, 7, 2)) & 1, 0, "torch off");
        assert!(power_of(&w, 6, 7, 2) > 0, "output lags one tick");
        step(&mut w, &idle());
        assert_eq!(power_of(&w, 6, 7, 2), 0, "output low one tick later");
        // and back: lever OFF -> torch relights
        w.set_block(3, 6, 2, LEVER);
        step(&mut w, &idle());
        step(&mut w, &idle());
        assert_eq!(cell_state(w.peek_block(5, 7, 2)) & 1, 1, "torch relit");
        assert!(power_of(&w, 6, 7, 2) > 0, "output high again");
    }

    #[test]
    fn nor_gate_from_torch_and_wire_join() {
        // NOR(A,B): both inputs feed the cell under one torch.
        let mut w = flat();
        w.set_block(2, 6, 2, LEVER); // A
        w.set_block(2, 6, 4, LEVER); // B
        w.set_block(3, 6, 2, WIRE);
        w.set_block(3, 6, 4, WIRE);
        w.set_block(4, 6, 2, WIRE);
        w.set_block(4, 6, 3, WIRE); // join under the torch
        w.set_block(4, 6, 4, WIRE);
        w.set_block(4, 7, 3, make_cell(RTORCH, 1));
        w.set_block(5, 7, 3, WIRE); // output
        w.set_block(6, 7, 3, LAMP);
        let settle = |w: &mut World| {
            for _ in 0..4 {
                step(w, &idle());
            }
        };
        settle(&mut w);
        assert!(power_of(&w, 5, 7, 3) > 0, "NOR(0,0)=1");
        assert_eq!(cell_state(w.peek_block(6, 7, 3)) & 1, 1, "lamp lit");
        for (a, b, want) in [(1u16, 0u16, false), (0, 1, false), (1, 1, false), (0, 0, true)] {
            w.set_block(2, 6, 2, make_cell(LEVER, a));
            w.set_block(2, 6, 4, make_cell(LEVER, b));
            settle(&mut w);
            assert_eq!(power_of(&w, 5, 7, 3) > 0, want, "NOR({},{})", a, b);
        }
    }

    #[test]
    fn rs_latch_stores_a_bit() {
        // Cross-coupled NORs. Geometry respects the routing rules: a torch
        // powers every horizontal wire at its level, so output nets leave
        // at y=7 and only drop to y=6 at least 2 cells from any torch.
        //
        // U1=(0,6,0) under T1=(0,7,0); Q1 net: (1,7,0),(2,7,0), drop
        // (2,6,0), route (3,6,0) -> U2=(4,6,0) under T2=(4,7,0);
        // Q2 net: (5,7,0),(6,7,0), drop (6,6,0), route
        // (6,6,1),(6,6,2),(5,6,2)..(0,6,2),(0,6,1) -> U1.
        // S lever (-1,6,0) feeds U1; R lever (4,6,-1) feeds U2;
        // Q lamp at (1,7,1) reads the Q1 net.
        let mut w = flat();
        w.set_block(0, 6, 0, WIRE); // U1
        w.set_block(0, 7, 0, RTORCH); // T1 (dark; power-up is indeterminate)
        w.set_block(1, 7, 0, WIRE); // Q1
        w.set_block(2, 7, 0, WIRE);
        w.set_block(2, 6, 0, WIRE);
        w.set_block(3, 6, 0, WIRE);
        w.set_block(4, 6, 0, WIRE); // U2
        w.set_block(4, 7, 0, RTORCH); // T2
        w.set_block(5, 7, 0, WIRE); // Q2
        w.set_block(6, 7, 0, WIRE);
        w.set_block(6, 6, 0, WIRE);
        w.set_block(6, 6, 1, WIRE);
        w.set_block(6, 6, 2, WIRE);
        for x in 0..=5 {
            w.set_block(x, 6, 2, WIRE);
        }
        w.set_block(0, 6, 1, WIRE);
        w.set_block(-1, 6, 0, make_cell(LEVER, 1)); // S, asserted at power-up
        w.set_block(4, 6, -1, LEVER); // R
        w.set_block(1, 7, 1, LAMP); // Q
        let settle = |w: &mut World, n: usize| {
            for _ in 0..n {
                step(w, &idle());
            }
        };
        let q = |w: &World| cell_state(w.peek_block(1, 7, 1)) & 1;
        // A perfectly symmetric NOR latch powers up metastable (it
        // oscillates in a synchronous sim exactly like a real latch is
        // indeterminate) — so S is asserted from tick 0 to force a state.
        settle(&mut w, 8);
        assert_eq!(q(&w), 0, "S=1 forces Q=0 (and stabilizes the latch)");
        w.set_block(-1, 6, 0, LEVER); // S released
        settle(&mut w, 8);
        assert_eq!(q(&w), 0, "Q holds 0 after S released");
        w.set_block(4, 6, -1, make_cell(LEVER, 1)); // R asserted
        settle(&mut w, 8);
        assert_eq!(q(&w), 1, "R=1 sets Q=1");
        w.set_block(4, 6, -1, LEVER); // R released
        settle(&mut w, 8);
        assert_eq!(q(&w), 1, "Q holds 1 after R released — memory");
        w.set_block(-1, 6, 0, make_cell(LEVER, 1)); // S again
        settle(&mut w, 8);
        assert_eq!(q(&w), 0, "bistable: S resets Q again");
    }

    #[test]
    fn ring_of_three_torches_oscillates_with_period_6() {
        // 3-inverter ring. T_i reads the wire below; output nets route at
        // y=7 before dropping to the next input, respecting the 2-cell
        // clearance rule. One torch is preset LIT to break the power-up
        // symmetry (a perfectly symmetric ring is metastable: period 2).
        let mut w = flat();
        // stage 1: In1 (0,6,0), T1 (0,7,0) preset lit, out (1,7,0),(2,7,0),
        // drop (2,6,0), feed (3,6,0) -> In2 (4,6,0)
        w.set_block(0, 6, 0, WIRE);
        w.set_block(0, 7, 0, make_cell(RTORCH, 1));
        w.set_block(1, 7, 0, WIRE);
        w.set_block(2, 7, 0, WIRE);
        w.set_block(2, 6, 0, WIRE);
        w.set_block(3, 6, 0, WIRE);
        // stage 2: T2 (4,7,0), out (5,7,0),(6,7,0), drop (6,6,0),
        // route (6,6,1),(6,6,2) -> In3 (6,6,3)
        w.set_block(4, 6, 0, WIRE);
        w.set_block(4, 7, 0, RTORCH);
        w.set_block(5, 7, 0, WIRE);
        w.set_block(6, 7, 0, WIRE);
        w.set_block(6, 6, 0, WIRE);
        w.set_block(6, 6, 1, WIRE);
        w.set_block(6, 6, 2, WIRE);
        // stage 3: T3 (6,7,3), out (6,7,4),(6,7,5), drop (6,6,5), route
        // (5..=0,6,5),(0,6,4),(0,6,3),(0,6,2),(0,6,1) -> In1 (0,6,0)
        w.set_block(6, 6, 3, WIRE);
        w.set_block(6, 7, 3, RTORCH);
        w.set_block(6, 7, 4, WIRE);
        w.set_block(6, 7, 5, WIRE);
        w.set_block(6, 6, 5, WIRE);
        for x in 0..=5 {
            w.set_block(x, 6, 5, WIRE);
        }
        w.set_block(0, 6, 4, WIRE);
        w.set_block(0, 6, 3, WIRE);
        w.set_block(0, 6, 2, WIRE);
        w.set_block(0, 6, 1, WIRE);
        for _ in 0..6 {
            step(&mut w, &idle());
        }
        let probe = |w: &World| power_of(w, 0, 6, 0) > 0;
        let mut seq = Vec::new();
        for _ in 0..12 {
            step(&mut w, &idle());
            seq.push(probe(&w));
        }
        assert!(
            seq.iter().any(|&v| v) && seq.iter().any(|&v| !v),
            "ring oscillates: {:?}",
            seq
        );
        for t in 0..6 {
            assert_eq!(seq[t], seq[t + 6], "period 6 at offset {}", t);
        }
    }

    #[test]
    fn repeater_is_buffer_delay_and_diode() {
        let mut w = flat();
        // lever -> 3 wires -> repeater(facing +x) -> 3 wires; a second
        // lever past the repeater must not backfeed.
        w.set_block(2, 6, 2, LEVER);
        for x in 3..=5 {
            w.set_block(x, 6, 2, WIRE);
        }
        w.set_block(6, 6, 2, make_cell(REPEATER, 3)); // dir +x
        for x in 7..=9 {
            w.set_block(x, 6, 2, WIRE);
        }
        // downstream lever ON: powers wires 9..7 but must NOT cross back
        w.set_block(10, 6, 2, make_cell(LEVER, 1));
        step(&mut w, &idle());
        step(&mut w, &idle());
        assert!(power_of(&w, 7, 6, 2) > 0, "downstream powered by its lever");
        assert_eq!(power_of(&w, 5, 6, 2), 0, "no backfeed through repeater");
        assert_eq!(cell_state(w.peek_block(6, 6, 2)) & 4, 0, "repeater out low");
        // upstream lever ON instead
        w.set_block(10, 6, 2, LEVER);
        w.set_block(2, 6, 2, make_cell(LEVER, 1));
        step(&mut w, &idle());
        // unit delay: the repeater's out-bit updates after this tick's solve
        assert_eq!(power_of(&w, 7, 6, 2), 0, "output not yet driven (delay)");
        step(&mut w, &idle());
        assert_eq!(cell_state(w.peek_block(6, 6, 2)) & 4, 4, "repeater out high");
        assert_eq!(power_of(&w, 7, 6, 2), 15, "signal refreshed to 15");
        assert!(power_of(&w, 9, 6, 2) > 0);
    }

    #[test]
    fn repeater_refresh_extends_past_15_cells() {
        let mut w = flat();
        w.set_block(2, 6, 2, make_cell(LEVER, 1));
        for x in 3..=12 {
            w.set_block(x, 6, 2, WIRE); // 10 wires: far end at power 5
        }
        w.set_block(13, 6, 2, make_cell(REPEATER, 3)); // re-drive
        for x in 14..=25 {
            w.set_block(x, 6, 2, WIRE); // 12 more, reachable only via refresh
        }
        for _ in 0..3 {
            step(&mut w, &idle());
        }
        assert_eq!(power_of(&w, 25, 6, 2), 4, "refreshed: 15-11=4 at the end");
    }

    #[test]
    fn self_loop_torch_blinks_and_snapshots_deterministically() {
        let build = || {
            let mut w = flat();
            w.set_block(5, 6, 2, WIRE);
            w.set_block(5, 7, 2, RTORCH);
            w.set_block(6, 7, 2, WIRE);
            w.set_block(6, 6, 2, WIRE);
            // self-loop: the drop cell (6,6,2) is adjacent to the input
            // (5,6,2) -> 1-inverter ring, period 2
            w.set_block(7, 7, 2, LAMP);
            w
        };
        let mut a = build();
        for _ in 0..50 {
            step(&mut a, &idle());
        }
        let snap = a.snapshot();
        for _ in 0..20 {
            step(&mut a, &idle());
        }
        let ha = a.hash();
        let mut b = World::restore(&snap).unwrap();
        for _ in 0..20 {
            step(&mut b, &idle());
        }
        assert_eq!(ha, b.hash(), "oscillator replay from snapshot");
        // lamp blinks: over any 4-tick window it takes both states
        let mut seen = [false, false];
        for _ in 0..4 {
            step(&mut a, &idle());
            seen[(cell_state(a.peek_block(7, 7, 2)) & 1) as usize] = true;
        }
        assert!(seen[0] && seen[1], "lamp follows the blinker");
    }

    #[test]
    fn placed_torch_defaults_lit() {
        // scenario/placement default: an RTORCH cell with state 0
        // self-corrects to lit (input unpowered) after one tick
        let mut w = flat();
        w.set_block(5, 7, 2, RTORCH); // state 0
        w.set_block(6, 7, 2, WIRE);
        step(&mut w, &idle());
        step(&mut w, &idle());
        assert_eq!(cell_state(w.peek_block(5, 7, 2)) & 1, 1);
        assert!(power_of(&w, 6, 7, 2) > 0);
    }
}

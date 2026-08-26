//! 36-slot inventory (slots 0..8 = hotbar), stack limit 64.

use crate::block::MAX_STACK;

pub const SLOTS: usize = 36;
pub const HOTBAR: usize = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stack {
    pub item: u16,
    pub count: u16,
}

#[derive(Clone, Debug)]
pub struct Inventory {
    pub slots: [Stack; SLOTS],
}

impl Default for Inventory {
    fn default() -> Self {
        Inventory::new()
    }
}

impl Inventory {
    pub fn new() -> Self {
        Inventory {
            slots: [Stack::default(); SLOTS],
        }
    }

    /// Add up to `count` of `item`; returns the leftover that did not fit.
    pub fn add(&mut self, item: u16, count: u16) -> u16 {
        let mut left = count;
        for s in self.slots.iter_mut() {
            if left == 0 {
                break;
            }
            if s.item == item && s.count < MAX_STACK {
                let take = (MAX_STACK - s.count).min(left);
                s.count += take;
                left -= take;
            }
        }
        for s in self.slots.iter_mut() {
            if left == 0 {
                break;
            }
            if s.count == 0 {
                let take = MAX_STACK.min(left);
                s.item = item;
                s.count = take;
                left -= take;
            }
        }
        left
    }

    /// Consume `count` of `item` across slots; true if fully satisfied.
    pub fn consume(&mut self, item: u16, count: u16) -> bool {
        if self.count(item) < count {
            return false;
        }
        let mut left = count;
        for s in self.slots.iter_mut() {
            if left == 0 {
                break;
            }
            if s.item == item && s.count > 0 {
                let take = s.count.min(left);
                s.count -= take;
                left -= take;
                if s.count == 0 {
                    s.item = 0;
                }
            }
        }
        true
    }

    pub fn count(&self, item: u16) -> u16 {
        self.slots
            .iter()
            .filter(|s| s.item == item)
            .map(|s| s.count)
            .sum()
    }

    /// Currently selected hotbar stack.
    pub fn held(&self, selected: usize) -> Stack {
        self.slots[selected.min(HOTBAR - 1)]
    }

    /// Remove one unit from the selected hotbar stack (e.g. on place).
    pub fn consume_held(&mut self, selected: usize) -> Option<u16> {
        let s = &mut self.slots[selected.min(HOTBAR - 1)];
        if s.count == 0 {
            return None;
        }
        let item = s.item;
        s.count -= 1;
        if s.count == 0 {
            s.item = 0;
        }
        Some(item)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn add_stacks_then_spills() {
        let mut inv = Inventory::new();
        assert_eq!(inv.add(5, 70), 0);
        assert_eq!(inv.count(5), 70);
        assert_eq!(inv.slots[0], Stack { item: 5, count: 64 });
        assert_eq!(inv.slots[1], Stack { item: 5, count: 6 });
    }

    #[test]
    fn add_overflow_returns_leftover() {
        let mut inv = Inventory::new();
        let left = inv.add(5, 36 * 64 + 3);
        assert_eq!(left, 3);
        assert_eq!(inv.count(5), 36 * 64);
    }

    #[test]
    fn consume_semantics() {
        let mut inv = Inventory::new();
        inv.add(9, 10);
        assert!(!inv.consume(9, 11));
        assert_eq!(inv.count(9), 10); // failed consume is atomic
        assert!(inv.consume(9, 4));
        assert_eq!(inv.count(9), 6);
        assert!(inv.consume(9, 6));
        assert_eq!(inv.count(9), 0);
        // slot cleared
        assert!(inv.slots.iter().all(|s| s.item != 9));
    }
}

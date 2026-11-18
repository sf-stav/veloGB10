    use super::*;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    pub const RING: usize = 4;
    pub const SLOT_WORDS: usize = 8192;

    pub struct Inner {
        slots: Vec<Vec<u32>>,
        pending: Vec<bool>,
        pub dead: Vec<bool>,
        pub abort_code: Vec<u64>,
        pub clobbers: Vec<String>,
    }

    pub struct Net { pub world: usize, clear_frame: bool, pub inner: Mutex<Inner>, cv: Condvar }

    impl Net {
        pub fn new(world: usize, clear_frame: bool) -> Arc<Net> {
            Arc::new(Net {
                world, clear_frame, cv: Condvar::new(),
                inner: Mutex::new(Inner {
                    slots: vec![vec![0u32; SLOT_WORDS]; world * world * RING], pending: vec![false; world * world * RING],
                    dead: vec![false; world], abort_code: vec![0; world], clobbers: Vec::new(),
                }),
            })
        }
        pub fn hub(self: &Arc<Net>, rank: usize, deadline: Option<Duration>) -> MockHub {
            MockHub { net: self.clone(), rank, gen: vec![0; self.world], send: vec![0; SLOT_WORDS],
                      last: vec![vec![0; SLOT_WORDS]; self.world], deadline, probe_after: Duration::from_millis(30) }
        }
        pub fn mark_dead(&self, r: usize) {
            self.inner.lock().unwrap().dead[r] = true;
            self.cv.notify_all();
        }
        pub fn abort_code(&self, r: usize) -> u64 { self.inner.lock().unwrap().abort_code[r] }
        pub fn clobbers(&self) -> Vec<String> { self.inner.lock().unwrap().clobbers.clone() }
    }

    pub struct MockHub {
        net: Arc<Net>, rank: usize, gen: Vec<u64>, send: Vec<u32>, last: Vec<Vec<u32>>,
        deadline: Option<Duration>, probe_after: Duration,
    }

    impl HubXport for MockHub {
        fn world(&self) -> usize { self.net.world }
        fn rank(&self) -> usize { self.rank }
        fn send_buf(&mut self, words: usize) -> &mut [u32] { &mut self.send[..words] }
        fn recv_copy(&self, src: usize, words: usize) -> Vec<u32> { self.last[src][..words].to_vec() }
        fn exchange_one(&mut self, peer: usize, nbytes: usize) -> i32 {
            assert!(peer != self.rank && peer < self.net.world && nbytes >= 16 && nbytes % 4 == 0);
            let (world, rank, words) = (self.net.world, self.rank, nbytes / 4);
            assert!(words <= SLOT_WORDS, "mock slot is {SLOT_WORDS} words");
            self.gen[peer] += 1;
            let g = self.gen[peer];
            let tag = (g << 8) | rank as u64;
            self.send[words - 2] = tag as u32;
            self.send[words - 1] = (tag >> 32) as u32;
            let mut inn = self.net.inner.lock().unwrap();
            let widx = (peer * world + rank) * RING + g as usize % RING;
            if inn.pending[widx] {
                inn.clobbers.push(format!("rank {rank} gen {g} overwrote an unconsumed frame in rank {peer}'s ring"));
            }
            inn.slots[widx][..words].copy_from_slice(&self.send[..words]);
            inn.pending[widx] = true;
            self.net.cv.notify_all();
            let ridx = (rank * world + peer) * RING + g as usize % RING;
            let expect = (g << 8) | peer as u64;
            let t0 = Instant::now();
            loop {
                let w = &inn.slots[ridx];
                if w[words - 2] as u64 | ((w[words - 1] as u64) << 32) == expect { break; }
                if inn.abort_code[rank] != 0 { return -2; }
                if let Some(d) = self.deadline {
                    if t0.elapsed() >= d {
                        if inn.abort_code[rank] == 0 { inn.abort_code[rank] = 12; }
                        return -4;
                    }
                }
                if t0.elapsed() >= self.probe_after && inn.dead[peer] {
                    if inn.abort_code[rank] == 0 { inn.abort_code[rank] = 10; }
                    return -3;
                }
                inn = self.net.cv.wait_timeout(inn, Duration::from_millis(5)).unwrap().0;
            }
            let frame: Vec<u32> = inn.slots[ridx][..words].to_vec();
            self.last[peer][..words].copy_from_slice(&frame);
            inn.pending[ridx] = false;
            if self.net.clear_frame { inn.slots[ridx][..words].iter_mut().for_each(|w| *w = 0); }
            0
        }
    }

    /// Run `f(rank, hub)` on one thread per rank; a rank whose closure returns is marked dead (its process ended).
    pub fn run_ranks<T: Send>(net: &Arc<Net>, deadline: Option<Duration>,
                              f: impl Fn(usize, &mut MockHub) -> T + Sync) -> Vec<T> {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..net.world).map(|r| {
                let (f, mut hub) = (&f, net.hub(r, deadline));
                s.spawn(move || { let out = f(r, &mut hub); hub.net.mark_dead(r); out })
            }).collect();
            hs.into_iter().map(|h| h.join().expect("rank thread panicked")).collect()
        })
    }

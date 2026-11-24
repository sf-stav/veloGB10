//! Minimal raw io_uring (setup / mmap / enter) for random reads. No crate dependency: the kernel
//! ABI is stable and the engine only needs IORING_OP_READ. EIO builds its reader pool on this.

use std::io;
use std::sync::atomic::{AtomicU32, Ordering};

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SqOff { head: u32, tail: u32, ring_mask: u32, ring_entries: u32, flags: u32, dropped: u32, array: u32, resv1: u32, user_addr: u64 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct CqOff { head: u32, tail: u32, ring_mask: u32, ring_entries: u32, overflow: u32, cqes: u32, flags: u32, resv1: u32, user_addr: u64 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Params { sq_entries: u32, cq_entries: u32, flags: u32, sq_thread_cpu: u32, sq_thread_idle: u32, features: u32, wq_fd: u32, resv: [u32; 3], sq_off: SqOff, cq_off: CqOff }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Sqe { opcode: u8, flags: u8, ioprio: u16, fd: i32, off: u64, addr: u64, len: u32, rw_flags: u32, user_data: u64, buf_index: u16, personality: u16, splice_fd_in: i32, addr3: u64, pad: u64 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Cqe { user_data: u64, res: i32, flags: u32 }

const OFF_SQ_RING: i64 = 0;
const OFF_CQ_RING: i64 = 0x8000000;
const OFF_SQES: i64 = 0x10000000;
const FEAT_SINGLE_MMAP: u32 = 1;
const OP_READ: u8 = 22;
const ENTER_GETEVENTS: u32 = 1;

pub struct Ring {
    fd: i32,
    sq_ptr: *mut u8,
    sq_len: usize,
    cq_ptr: *mut u8,
    cq_len: usize,
    sqes: *mut Sqe,
    sqes_len: usize,
    p: Params,
    sq_tail_local: u32,
    to_submit: u32,
}
unsafe impl Send for Ring {}

impl Ring {
    pub fn new(entries: u32) -> io::Result<Self> {
        let mut p = Params::default();
        let fd = unsafe { libc::syscall(libc::SYS_io_uring_setup, entries as libc::c_long, &mut p as *mut Params) } as i32;
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let sq_len = p.sq_off.array as usize + p.sq_entries as usize * 4;
        let cq_len = p.cq_off.cqes as usize + p.cq_entries as usize * std::mem::size_of::<Cqe>();
        let single = p.features & FEAT_SINGLE_MMAP != 0;
        let (sq_len, cq_len) = if single { (sq_len.max(cq_len), sq_len.max(cq_len)) } else { (sq_len, cq_len) };
        let map = |len: usize, off: i64| -> io::Result<*mut u8> {
            let m = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_POPULATE, fd, off) };
            if m == libc::MAP_FAILED { Err(io::Error::last_os_error()) } else { Ok(m as *mut u8) }
        };
        let sq_ptr = map(sq_len, OFF_SQ_RING)?;
        let cq_ptr = if single { sq_ptr } else { map(cq_len, OFF_CQ_RING)? };
        let sqes_len = p.sq_entries as usize * std::mem::size_of::<Sqe>();
        let sqes = map(sqes_len, OFF_SQES)? as *mut Sqe;
        let tail = unsafe { (*(sq_ptr.add(p.sq_off.tail as usize) as *const AtomicU32)).load(Ordering::Acquire) };
        Ok(Ring { fd, sq_ptr, sq_len, cq_ptr, cq_len, sqes, sqes_len, p, sq_tail_local: tail, to_submit: 0 })
    }

    #[inline]
    fn a(&self, base: *mut u8, off: u32) -> &AtomicU32 {
        unsafe { &*(base.add(off as usize) as *const AtomicU32) }
    }

    /// Queue a read of `len` bytes at file offset `off` into `buf`. False if the submission queue is full.
    pub fn push_read(&mut self, fd: i32, buf: *mut u8, len: u32, off: u64, user_data: u64) -> bool {
        let head = self.a(self.sq_ptr, self.p.sq_off.head).load(Ordering::Acquire);
        if self.sq_tail_local.wrapping_sub(head) >= self.p.sq_entries {
            return false;
        }
        let mask = self.p.sq_entries - 1;
        let idx = self.sq_tail_local & mask;
        unsafe {
            *self.sqes.add(idx as usize) = Sqe { opcode: OP_READ, fd, off, addr: buf as u64, len, user_data, ..Default::default() };
            *(self.sq_ptr.add(self.p.sq_off.array as usize) as *mut u32).add(idx as usize) = idx;
        }
        self.sq_tail_local = self.sq_tail_local.wrapping_add(1);
        self.a(self.sq_ptr, self.p.sq_off.tail).store(self.sq_tail_local, Ordering::Release);
        self.to_submit += 1;
        true
    }

    /// Submit everything queued and wait for at least `min_complete` completions.
    pub fn submit_and_wait(&mut self, min_complete: u32) -> io::Result<()> {
        let flags = if min_complete > 0 { ENTER_GETEVENTS } else { 0 };
        let r = unsafe { libc::syscall(libc::SYS_io_uring_enter, self.fd as libc::c_long, self.to_submit as libc::c_long, min_complete as libc::c_long, flags as libc::c_long, 0usize, 0usize) };
        if r < 0 {
            let e = io::Error::last_os_error();
            if e.raw_os_error() != Some(libc::EINTR) {
                return Err(e);
            }
            return Ok(());
        }
        self.to_submit -= (r as u32).min(self.to_submit);
        Ok(())
    }

    /// Run the ring's deferred completion work without waiting (io_uring_enter with GETEVENTS and
    /// min_complete 0), so a reader that spins on the completion queue instead of blocking still sees
    /// every completion promptly. Also submits anything queued. (EIO)
    pub fn flush_events(&mut self) -> io::Result<()> {
        let r = unsafe { libc::syscall(libc::SYS_io_uring_enter, self.fd as libc::c_long, self.to_submit as libc::c_long, 0 as libc::c_long, ENTER_GETEVENTS as libc::c_long, 0usize, 0usize) };
        if r < 0 {
            let e = io::Error::last_os_error();
            if e.raw_os_error() != Some(libc::EINTR) && e.raw_os_error() != Some(libc::EAGAIN) && e.raw_os_error() != Some(libc::EBUSY) {
                return Err(e);
            }
            return Ok(());
        }
        self.to_submit -= (r as u32).min(self.to_submit);
        Ok(())
    }

    /// Submission-queue entries (= the most reads that can be in flight).
    pub fn entries(&self) -> u32 {
        self.p.sq_entries
    }

    /// Pop one completion: (user_data, res). res < 0 is -errno.
    pub fn pop(&mut self) -> Option<(u64, i32)> {
        let head = self.a(self.cq_ptr, self.p.cq_off.head).load(Ordering::Relaxed);
        let tail = self.a(self.cq_ptr, self.p.cq_off.tail).load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let mask = self.p.cq_entries - 1;
        let cqe = unsafe { *(self.cq_ptr.add(self.p.cq_off.cqes as usize) as *const Cqe).add((head & mask) as usize) };
        self.a(self.cq_ptr, self.p.cq_off.head).store(head.wrapping_add(1), Ordering::Release);
        Some((cqe.user_data, cqe.res))
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.sqes as *mut _, self.sqes_len);
            if self.cq_ptr != self.sq_ptr {
                libc::munmap(self.cq_ptr as *mut _, self.cq_len);
            }
            libc::munmap(self.sq_ptr as *mut _, self.sq_len);
            libc::close(self.fd);
        }
    }
}

/// 4096-aligned heap buffer for O_DIRECT.
pub struct AlignedBuf {
    ptr: *mut u8,
    pub len: usize,
}
unsafe impl Send for AlignedBuf {}
impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, 4096).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        AlignedBuf { ptr, len }
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, 4096).unwrap()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::AsRawFd;

    #[test]
    fn uring_reads_a_file() {
        let p = std::env::temp_dir().join(format!("engb_uring_{}", std::process::id()));
        let data: Vec<u8> = (0..16384u32).map(|i| (i % 253) as u8).collect();
        std::fs::File::create(&p).unwrap().write_all(&data).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mut ring = match Ring::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring unavailable here ({e}); skipping");
                std::fs::remove_file(&p).ok();
                return;
            }
        };
        let mut bufs: Vec<AlignedBuf> = (0..4).map(|_| AlignedBuf::new(4096)).collect();
        for (i, b) in bufs.iter_mut().enumerate() {
            assert!(ring.push_read(f.as_raw_fd(), b.as_mut_ptr(), 4096, (i * 4096) as u64, i as u64));
        }
        ring.submit_and_wait(4).unwrap();
        let mut n = 0;
        while let Some((ud, res)) = ring.pop() {
            assert_eq!(res, 4096);
            assert_eq!(bufs[ud as usize].as_slice(), &data[ud as usize * 4096..(ud as usize + 1) * 4096]);
            n += 1;
        }
        assert_eq!(n, 4);
        std::fs::remove_file(&p).ok();
    }
}

    use super::*;

    /// ISSUES 153: the abort signal behind the liveness-tied bring-up retry. A registered
    /// stream reads ALIVE while the peer holds it and DEAD after the peer closes, without
    /// consuming a single protocol byte.
    #[test]
    fn head_liveness_peek_sees_peer_close() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let ours = l.accept().unwrap().0;
        set_head_liveness(&ours);
        assert_eq!(unsafe { net_abort_check() }, 0, "fresh pair must read alive");
        drop(peer);
        // EOF lands asynchronously; allow a few seconds.
        let mut dead = false;
        for _ in 0..250 {
            if unsafe { net_abort_check() } != 0 { dead = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(dead, "peer close must flip the peek to dead");
        clear_head_liveness();
        assert_eq!(unsafe { net_abort_check() }, 0, "cleared registration must read alive");
    }

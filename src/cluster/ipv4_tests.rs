    use super::*;

    #[test]
    fn slash24_with_brd_uses_kernel_value() {
        let s = "2: eth0    inet 192.0.2.13/24 brd 192.0.2.255 scope global eth0";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(192, 0, 2, 13));
        assert_eq!(prefix, 24);
        assert_eq!(bcast, Ipv4Addr::new(192, 0, 2, 255));
    }

    #[test]
    fn slash31_no_brd_synthesizes() {
        // Peer link /31: kernel prints no `brd`; ip|~mask gives the link's other address.
        let s = "3: p2p    inet 10.0.0.0/31 scope link p2p";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 0));
        assert_eq!(prefix, 31);
        assert_eq!(bcast, Ipv4Addr::new(10, 0, 0, 1));
    }

    #[test]
    fn slash30_no_brd_synthesizes() {
        let s = "4: p2p    inet 10.0.0.2/30 scope link p2p";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 2));
        assert_eq!(prefix, 30);
        assert_eq!(bcast, Ipv4Addr::new(10, 0, 0, 3));
    }

    #[test]
    fn prefix_zero_no_panic_bcast_is_all_ones() {
        let s = "5: p2p    inet 10.0.0.7/0 scope link p2p";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 7));
        assert_eq!(prefix, 0);
        assert_eq!(bcast, Ipv4Addr::new(255, 255, 255, 255));
    }

    #[test]
    fn missing_addr_is_none() {
        assert!(ipv4_of_str("2: eth0    <BROADCAST,MULTICAST,UP> mtu 1500").is_none());
    }

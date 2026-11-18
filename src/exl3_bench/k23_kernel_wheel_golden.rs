//! Host-side proof tests (module `k23_kernel_wheel_golden`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    use super::*;

    #[test]
    fn k23_kernel_dump_vs_wheel() {
        let Ok(j) = std::fs::read_to_string("tests/k23_golden_l4e7up.json")
            else { eprintln!("[skip] lab fixture tests/k23_golden_l4e7up.json not present"); return; };
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let kb = v["kb"].as_u64().unwrap() as usize;
        let nb = v["nb"].as_u64().unwrap() as usize;
        let trellis: Vec<i16> = v["trellis_i16"].as_array().unwrap()
            .iter().map(|x| x.as_i64().unwrap() as i16).collect();
        let want: Vec<u16> = v["inner_f16_bits"].as_array().unwrap()
            .iter().map(|x| x.as_i64().unwrap() as u16).collect();
        let raw: Vec<u8> = trellis.iter().flat_map(|x| x.to_le_bytes()).collect();

        let dev = dev0().unwrap();
        load_module(&dev).unwrap();
        let rc = tmap_inv_table();
        let d_rc = dev.htod_sync_copy(&rc).unwrap();
        let f = dev.get_func(MODULE, "exl3_decode_dump").unwrap();
        let stream = fork_blocking_stream(&dev);
        let d_tr = dev.htod_sync_copy(&raw).unwrap();
        let mut d_out = unsafe { dev.alloc::<u16>(want.len()) }.unwrap();
        unsafe {
            f.clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((kb * nb) as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_tr, &d_rc, &mut d_out, nb as i32, 3i32),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&d_out).unwrap();
        assert_eq!(got.len(), want.len(), "decoded size mismatch");
        let nbad = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert!(
            nbad == 0,
            "k2=3 KERNEL decode differs from wheel golden: {nbad}/{} values, first diffs {:?}",
            got.len(),
            got.iter().zip(want.iter()).enumerate()
                .filter_map(|(i, (a, b))| if a != b { Some((i, *a, *b)) } else { None })
                .take(8).collect::<Vec<_>>()
        );
    }

//! Unit tests (module `vision_image_cache_tests`) moved out of the old inline block; text unchanged.

    use super::{request_image_identities, images_compatible};
    use crate::vision_encoder::ImageSpan;
    use crate::vision_tower::OUT_HIDDEN;

    fn span(start: usize, n: usize) -> ImageSpan { ImageSpan { start, num_tokens: n, ..Default::default() } }

    /// Build a deterministic concatenated embeds buffer for two images of width OUT_HIDDEN.
    fn two_image_embeds(n0: usize, n1: usize) -> (Vec<f32>, Vec<ImageSpan>) {
        let mut e = Vec::new();
        let mut seed = 0x12345678u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) - 0.5
        };
        for _ in 0..n0 * OUT_HIDDEN { e.push(next()); }
        for _ in 0..n1 * OUT_HIDDEN { e.push(next()); }
        (e, vec![span(60, n0), span(60 + n0, n1)])
    }

    #[test]
    fn identities_are_deterministic_and_positioned() {
        let (emb, spans) = two_image_embeds(4, 5);
        let ids = request_image_identities(Some(&emb), &spans);
        assert_eq!(ids.len(), 2, "two images -> two identities");
        assert_eq!(ids[0].start, 60);
        assert_eq!(ids[1].start, 60 + 4);
        let ids2 = request_image_identities(Some(&emb), &spans);
        assert_eq!(ids[0].hash, ids2[0].hash);
        assert_eq!(ids[1].hash, ids2[1].hash);
        // Different content at image 1 -> different hash. Use a perturbation large enough to
        // change the f32 bits (the LCG values are ~2^30, where the f32 ULP is ~256, so a +0.5
        // would round away).
        let mut emb2 = emb.clone();
        emb2[4 * OUT_HIDDEN] = -12345.678;
        let ids3 = request_image_identities(Some(&emb2), &spans);
        assert_eq!(ids3[0].hash, ids[0].hash, "image 0 unchanged");
        assert_ne!(ids3[1].hash, ids[1].hash, "image 1 changed -> hash must differ");
        assert!(request_image_identities(None, &[]).is_empty());
    }

    #[test]
    fn compatibility_detects_a_changed_image_in_the_reused_prefix() {
        let (emb, spans) = two_image_embeds(4, 5);
        let img_old = request_image_identities(Some(&emb), &spans);
        assert!(images_compatible(&img_old, &img_old, 80), "identical image key must match");
        assert!(images_compatible(&img_old, &[], 80), "text-only request may reuse any prefix");
        // A request with a DIFFERENT image at position 60 (inside the reused prefix) must NOT reuse.
        let mut emb_c = emb.clone();
        emb_c[5] = -12345.678;
        let img_new = request_image_identities(Some(&emb_c), &spans);
        assert!(!images_compatible(&img_old, &img_new, 80), "changed image in the reused prefix must break reuse");
        // An image wholly in the SUFFIX (start >= l) imposes no constraint.
        let (_, spans_suffix) = (emb_c.clone(), vec![span(70, 4), span(80, 5)]);
        let img_sfx = request_image_identities(Some(&emb_c), &spans_suffix);
        assert!(images_compatible(&img_old, &img_sfx, 60), "image wholly in the suffix imposes no constraint");
    }

//! V1.2 probe: strict-load the 333 model.visual.* tensors and validate count/shape/dtype.
fn main() {
    // the model dir is required (no built-in lab path: S-B9-REL-SMALLFIXES item 8)
    let md = std::env::args().nth(1).expect("usage: w2_vision_load <MODEL_DIR>");
    match gb10_inference::vision_tower::VisualTower::load(&md) {
        Ok(t) => {
            println!("VISION_LOAD_OK tensors={} blocks={}", t.tensor_count(), t.blocks.len());
            println!("  patch_embed_w[{}] pos_embed_w[{}] merger_fc2_w[{}]",
                t.patch_embed_w.len(), t.pos_embed_w.len(), t.merger_fc2_w.len());
            println!("  block0 qkv_w[{}] fc1_w[{}] norm1_w[{}]",
                t.blocks[0].qkv_w.len(), t.blocks[0].fc1_w.len(), t.blocks[0].norm1_w.len());
            println!("SUCCESS_VISION_LOAD");
        }
        Err(e) => { eprintln!("VISION_LOAD_ERROR: {e}"); println!("FAIL_VISION_LOAD"); }
    }
}

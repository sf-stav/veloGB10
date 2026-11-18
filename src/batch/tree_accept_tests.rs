//! Unit tests (module `tree_accept_tests`) moved out of the old inline block; text unchanged.

    use super::tree_accept_walk;

    // A tree degenerates to a chain (parent[c]=c-1). The walk must reproduce "accept longest prefix".
    #[test]
    fn chain_is_accept_longest_prefix() {
        // committed=100; drafts=[10,20,30]; target preds=[10,20,99,..] => accept 10,20, correct 30->99.
        let parent = [-1, 0, 1, 2];
        let tokens = [100u32, 10, 20, 30];
        let preds  = [10u32, 20, 99, 7];         // preds[2]=99 != tokens[3]=30 -> stop, bonus 99
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 1, 2]);          // accepted committed, 10, 20
        assert_eq!(emitted, vec![10, 20, 99]);    // 10, 20 accepted, 99 the correction
    }

    // Fork at position 1: child A (col 1) wrong, child B (col 2) right -> B rescued, then chains.
    #[test]
    fn fork_rescues_second_branch() {
        //        0(committed=100)
        //       / \
        //   1(A=10) 2(B=20)---3(B2=30)
        let parent = [-1, 0, 0, 2];
        let tokens = [100u32, 10, 20, 30];
        let preds  = [20u32, 5, 30, 88];  // after committed target wants 20 (=B), then 30 (=B2), then 88
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 2, 3]);          // walked root -> B -> B2
        assert_eq!(emitted, vec![20, 30, 88]);    // B, B2 accepted, 88 the bonus
    }

    // Neither child matches at the root -> emit just the correction, accept nothing past committed.
    #[test]
    fn no_child_matches() {
        let parent = [-1, 0, 0];
        let tokens = [100u32, 10, 20];
        let preds  = [77u32, 1, 2];               // target wants 77, neither child is 77
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0]);
        assert_eq!(emitted, vec![77]);
    }

    // Tie: two children carry the target's token -> prefer the lowest index (deterministic).
    #[test]
    fn tie_prefers_lowest_child() {
        let parent = [-1, 0, 0];
        let tokens = [100u32, 42, 42];
        let preds  = [42u32, 9, 9];
        let (path, _) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 1]);             // col 1, not col 2
    }

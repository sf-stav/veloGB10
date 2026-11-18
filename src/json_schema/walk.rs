//! `SchemaMask` internals: interning, the trie walk, per-value starts and the byte feed. Another `impl SchemaMask`
//! block (the public methods are in `mask.rs`); moved methods keep their exact text. Map in `mod.rs`.

use super::*;

impl SchemaMask {
    // -----------------------------------------------------------------------------------
    // internals
    // -----------------------------------------------------------------------------------

    pub(super) fn machine(&self, state: u32) -> Option<Arc<Machine>> {
        lock(&self.interner).states.get(state as usize).cloned()
    }

    pub(super) fn intern(&self, m: Machine) -> u32 {
        let mut it = lock(&self.interner);
        if let Some(&id) = it.index.get(&m) {
            return id;
        }
        let id = it.states.len() as u32;
        it.index.insert(m.clone(), id);
        it.states.push(Arc::new(m));
        id
    }

    pub(super) fn compute_allowed(&self, state: u32) -> Vec<u32> {
        let mut bits = vec![0u32; self.words];
        if self.words == 0 || self.vocab.nodes.is_empty() {
            return bits;
        }
        let Some(stored) = self.machine(state) else { return bits };
        self.walk_into(&stored, &mut bits);
        bits
    }

    /// [`SchemaMask::compute_allowed`] for a machine snapshot that need not be interned.
    pub(super) fn compute_allowed_machine(&self, m: &Machine) -> Vec<u32> {
        let mut bits = vec![0u32; self.words];
        if self.words == 0 || self.vocab.nodes.is_empty() {
            return bits;
        }
        self.walk_into(m, &mut bits);
        bits
    }

    fn walk_into(&self, stored: &Machine, bits: &mut Vec<u32>) {
        if stored.is_dead() {
            return;
        }
        let mut m = stored.clone();
        self.trie_walk(&mut m, 0, bits);
    }

    /// DFS the vocabulary trie from `node`, pruning the moment the machine refuses a byte.
    /// One machine is reused and rolled back per byte (see [`Roll`]) — no per-node clones and no
    /// full-vocab rescan.
    fn trie_walk(&self, m: &mut Machine, node: usize, out: &mut Vec<u32>) {
        // Iterate the shared trie by reference: cloning the child list at every node cost one
        // allocation per visited node (~1M on a 248K vocabulary inside a free string).
        let vocab = &*self.vocab;
        for &(b, child) in &vocab.nodes[node].children {
            let mut roll = Roll::new(m.stack.len());
            let ws0 = m.ws; // not covered by the Roll undo log (it records frames only)
            let alive = self.feed(m, b, Some(&mut roll));
            if alive {
                if let Some(tok) = vocab.nodes[child].token {
                    let t = tok as usize;
                    if t < self.vocab_size {
                        out[t >> 5] |= 1u32 << (t & 31);
                    }
                }
                self.trie_walk(m, child, out);
            }
            roll.restore(m);
            m.ws = ws0;
        }
    }

    /// Push the frame(s) that start a JSON value matching `si` in response to byte `b`.
    fn value_start(&self, m: &mut Machine, si: u32, b: u8) -> bool {
        let s = &self.schemas[si as usize];
        let (has_enum, any_str, any_int) = (s.has_enum, !s.enum_str.is_empty(), !s.enum_int.is_empty());
        let push_str = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Str { si, st: StrState::new() });
        };
        let push_obj = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Obj {
                si,
                phase: ObjPhase::KeyOrEnd,
                seen: Vec::new(),
                extra: Vec::new(),
            });
        };
        let push_arr = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Arr { si, n: 0, phase: ArrPhase::ItemOrEnd });
        };
        match s.ty {
            Ty::Object => {
                if b != b'{' {
                    return false;
                }
                push_obj(m, si);
                true
            }
            Ty::Array => {
                if b != b'[' {
                    return false;
                }
                push_arr(m, si);
                true
            }
            Ty::String => {
                if b != b'"' {
                    return false;
                }
                push_str(m, si);
                true
            }
            Ty::Integer | Ty::Number => match num_start(b) {
                Some((st, buf)) => {
                    if !self.num_prefix_viable(si, &buf, st) {
                        return false;
                    }
                    m.stack.push(Frame::Num { si, buf, st });
                    true
                }
                None => false,
            },
            Ty::Boolean => match b {
                b't' => {
                    m.stack.push(Frame::Lit { rest: b"rue" });
                    true
                }
                b'f' => {
                    m.stack.push(Frame::Lit { rest: b"alse" });
                    true
                }
                _ => false,
            },
            Ty::Null => {
                if b != b'n' {
                    return false;
                }
                m.stack.push(Frame::Lit { rest: b"ull" });
                true
            }
            Ty::Any => {
                // Untyped: accept whatever the schema constrains. With an enum, only the value
                // kinds the enum actually contains may start.
                if has_enum && !any_str && b == b'"' {
                    return false;
                }
                if has_enum && !any_int && (b == b'-' || b.is_ascii_digit()) {
                    return false;
                }
                // enum values are strings and integers only: an object or array never matches
                if has_enum && (b == b'{' || b == b'[') {
                    return false;
                }
                match b {
                    b'{' => {
                        push_obj(m, si);
                        true
                    }
                    b'[' => {
                        push_arr(m, si);
                        true
                    }
                    b'"' => {
                        push_str(m, si);
                        true
                    }
                    b't' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"rue" });
                        true
                    }
                    b'f' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"alse" });
                        true
                    }
                    b'n' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"ull" });
                        true
                    }
                    _ => match num_start(b) {
                        Some((st, buf)) => {
                            if !self.num_prefix_viable(si, &buf, st) {
                                return false;
                            }
                            m.stack.push(Frame::Num { si, buf, st });
                            true
                        }
                        None => false,
                    },
                }
            }
        }
    }

    fn required_satisfied(&self, si: u32, seen: &[u32]) -> bool {
        self.schemas[si as usize].required.iter().all(|r| seen.contains(r))
    }

    /// Value-string constraint check. `closing` = the terminating quote was consumed.
    fn str_value_ok(&self, si: u32, st: &StrState, closing: bool) -> bool {
        let s = &self.schemas[si as usize];
        if let Some(mx) = s.max_len.filter(|_| !st.free_len) {
            // A started escape or UTF-8 sequence already commits one more character: counting it
            // only once complete let a string at maxLength begin one, after which no byte could
            // finish it (a dead end that ends the reply mid-document).
            if st.cp as u64 + pending_char(st) > mx {
                return false;
            }
        }
        if closing {
            if !st.is_closed_interior() {
                return false;
            }
            if let Some(mn) = s.min_len.filter(|_| !st.free_len) {
                if (st.cp as u64) < mn {
                    return false;
                }
            }
            if !s.enum_str.is_empty() && !s.enum_str.iter().any(|e| canon_match(e, &st.raw).1) {
                return false;
            }
        } else if !s.enum_str.is_empty() && !s.enum_str.iter().any(|e| canon_match(e, &st.raw).0) {
            // enum prefix rule: only bytes that keep us on the way to an allowed value.
            return false;
        }
        true
    }

    fn num_ok(&self, si: u32, buf: &[u8], st: NumState) -> bool {
        if !st.terminal() {
            return false;
        }
        let s = &self.schemas[si as usize];
        let Ok(text) = std::str::from_utf8(buf) else { return false };
        let Ok(v) = text.parse::<f64>() else { return false };
        if s.ty == Ty::Integer && !matches!(st, NumState::Zero | NumState::Int) {
            return false; // integer forbids fraction/exponent
        }
        // a number matches only the enum's integer values (an untyped enum may list strings too)
        if s.has_enum && !s.enum_int.iter().any(|&e| e as f64 == v) {
            return false;
        }
        if let Some(m) = s.minimum {
            if v < m {
                return false;
            }
        }
        if let Some(m) = s.excl_min {
            if v <= m {
                return false;
            }
        }
        if let Some(m) = s.maximum {
            if v > m {
                return false;
            }
        }
        if let Some(m) = s.excl_max {
            if v >= m {
                return false;
            }
        }
        true
    }

    /// Can a number whose bytes so far are `buf` (grammar position `st`) still END UP valid?
    ///
    /// `num_ok` runs only when the number TERMINATES, so without this check the allowed set let the
    /// model run an unbounded digit sequence after a prefix that could no longer complete.
    ///
    /// The rule may prune the mask (refuse a prefix) only when no valid completion exists, or
    /// when it narrows the spelling of a value that is still reachable another way:
    ///   - an integer enum: the canonical spelling of a listed value only (as for string enums);
    ///   - `integer`: no fraction or exponent (`num_ok` refuses them), and the digits must still
    ///     reach an integer inside the bounds (the values `p*10^k ..= p*10^k + 10^k - 1`);
    ///   - `number`: a sign the bounds exclude (a nonzero negative with no negative value allowed,
    ///     a nonzero positive with no positive value allowed). Any other float prefix can still
    ///     be moved into range by `.`/`e` (`70e-1 == 7`).
    fn num_prefix_viable(&self, si: u32, buf: &[u8], st: NumState) -> bool {
        let s = &self.schemas[si as usize];
        if !s.enum_int.is_empty() {
            return s.enum_int.iter().any(|e| e.to_string().as_bytes().starts_with(buf));
        }
        let negative = buf.first() == Some(&b'-');
        let digits = if negative { &buf[1..] } else { buf };
        let lead = digits.first().copied();
        match s.ty {
            Ty::Integer => {}
            Ty::Number => {
                let neg_ok = s.minimum.map_or(true, |m| m < 0.0) && s.excl_min.map_or(true, |m| m < 0.0);
                let pos_ok = s.maximum.map_or(true, |m| m > 0.0) && s.excl_max.map_or(true, |m| m > 0.0);
                let zero_ok = s.minimum.map_or(true, |m| m <= 0.0) && s.excl_min.map_or(true, |m| m < 0.0)
                    && s.maximum.map_or(true, |m| m >= 0.0) && s.excl_max.map_or(true, |m| m > 0.0);
                let lower = s.minimum.unwrap_or(f64::NEG_INFINITY).max(s.excl_min.unwrap_or(f64::NEG_INFINITY));
                let upper = s.maximum.unwrap_or(f64::INFINITY).min(s.excl_max.unwrap_or(f64::INFINITY));
                return match lead {
                    None => neg_ok || zero_ok, // a bare `-`: `-0` or a negative
                    // `0`, `0.x`: a value in [0, 1); `-0`, `-0.x`: a value in (-1, 0]
                    Some(b'0') if negative => zero_ok || (neg_ok && upper > -1.0),
                    Some(b'0') => zero_ok || (pos_ok && lower < 1.0),
                    Some(_) if negative => neg_ok,
                    Some(_) => pos_ok,
                };
            }
            _ => return true,
        }
        if !matches!(st, NumState::NegStart | NumState::Zero | NumState::Int) {
            return false;
        }
        // the integer bounds
        let lo = s.minimum.map_or(f64::NEG_INFINITY, f64::ceil)
            .max(s.excl_min.map_or(f64::NEG_INFINITY, |m| m.floor() + 1.0));
        let hi = s.maximum.map_or(f64::INFINITY, f64::floor)
            .min(s.excl_max.map_or(f64::INFINITY, |m| m.ceil() - 1.0));
        if lo > hi {
            return false;
        }
        if digits.is_empty() {
            // a bare `-`: `-0` (= 0) or a negative integer
            return lo <= hi.min(0.0);
        }
        // `0` / `-0` cannot be extended by another digit (leading zeros are illegal JSON)
        if st == NumState::Zero {
            return self.num_ok(si, buf, st);
        }
        let Ok(text) = std::str::from_utf8(digits) else { return true };
        let Ok(p) = text.parse::<f64>() else { return true };
        // the magnitudes this prefix can still become: [p*10^k, p*10^k + 10^k - 1], k = 0, 1, ...
        let mut scale = 1.0f64;
        loop {
            let (a, b) = (p * scale, p * scale + scale - 1.0);
            let (vlo, vhi) = if negative { (-b, -a) } else { (a, b) };
            if vlo <= hi && vhi >= lo {
                return true;
            }
            // every longer number lies further from zero, so past the bound means never
            if (!negative && a > hi) || (negative && -a < lo) {
                return false;
            }
            if scale > 1e19 {
                return true; // beyond the integer range checked exactly here; `num_ok` decides
            }
            scale *= 10.0;
        }
    }

    /// Pop a terminal pending number frame (used by `is_complete`); false = it was not valid.
    pub(super) fn finalize_pending_number(&self, m: &mut Machine) -> bool {
        if let Some(Frame::Num { si, buf, st }) = m.stack.last().cloned() {
            if self.num_ok(si, &buf, st) {
                m.stack.pop();
                return true;
            }
            return false;
        }
        true
    }

    /// Can the key being parsed still become a key this object accepts? Declared keys already
    /// present (`seen`) are excluded: a duplicate is refused when the key closes, so letting it
    /// through here led into a state with no allowed continuation.
    fn key_prefix_ok(&self, si: u32, st: &StrState, seen: &[u32]) -> bool {
        let s = &self.schemas[si as usize];
        s.additional
            || s.props.iter().enumerate().any(|(i, (n, _))| !seen.contains(&(i as u32)) && canon_match(n, &st.raw).0)
    }

    /// May another key start in this object (`"` after `{`, `,` after a member)?
    fn can_add_key(&self, si: u32, seen: &[u32]) -> bool {
        let s = &self.schemas[si as usize];
        s.additional || (0..s.props.len() as u32).any(|i| !seen.contains(&i))
    }

    /// Feed ONE byte. Returns true when the byte was consumed by a live machine; false when the
    /// machine died (a [`Frame::Dead`] is pushed and absorbs everything afterwards).
    ///
    /// Frames are mutated only after being popped, which is what makes the [`Roll`] undo log and
    /// the "pop, then let the parent reprocess the same byte" loop sound.
    pub(super) fn feed(&self, m: &mut Machine, b: u8, roll: Option<&mut Roll>) -> bool {
        // Trailing whitespace after a complete document (top = DocEnd) is not counted: the
        // completed state stays one interned state, and the scheduler ends the turn there anyway.
        let structural_ws = is_ws(b) && !m.in_string() && !matches!(m.stack.last(), Some(Frame::DocEnd));
        // No whitespace before the root value: the document starts at its first value byte (a
        // leading run carries no information and only spends tokens).
        if structural_ws && m.stack.len() == 2 && matches!(m.stack[1], Frame::Value { .. }) {
            return false;
        }
        if structural_ws && m.ws >= MAX_WS_RUN {
            // Refused without touching the stack: both callers discard the machine (step) or
            // roll it back (trie_walk) on `false`.
            return false;
        }
        let alive = self.feed_inner(m, b, roll);
        if alive {
            m.ws = if structural_ws { m.ws + 1 } else { 0 };
        }
        alive
    }

    fn feed_inner(&self, m: &mut Machine, b: u8, mut roll: Option<&mut Roll>) -> bool {
        loop {
            if m.stack.is_empty() {
                m.stack.push(Frame::Dead);
                return false;
            }
            let idx = m.stack.len() - 1;
            let top = m.stack.pop().expect("checked non-empty");
            if let Some(r) = roll.as_deref_mut() {
                r.note_pop(idx, &top);
            }
            return match top {
                Frame::Dead => {
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::DocEnd => {
                    if is_ws(b) {
                        m.stack.push(Frame::DocEnd);
                        return true;
                    }
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::Value { si } => {
                    if is_ws(b) {
                        m.stack.push(Frame::Value { si });
                        return true;
                    }
                    if self.value_start(m, si, b) {
                        return true;
                    }
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::Obj { si, phase, mut seen, mut extra } => match phase {
                    ObjPhase::KeyOrEnd => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyOrEnd, seen, extra });
                            return true;
                        }
                        if b == b'}' {
                            if self.required_satisfied(si, &seen) {
                                return true; // object closed; the byte belongs to this frame
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        if b == b'"' && self.can_add_key(si, &seen) {
                            m.stack.push(Frame::Obj {
                                si,
                                phase: ObjPhase::Key(StrState::new()),
                                seen,
                                extra,
                            });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    ObjPhase::Key(mut st) => {
                        match str_advance(&mut st, b, true) {
                            Step::Open => {
                                if !self.key_prefix_ok(si, &st, &seen) {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                }
                                m.stack.push(Frame::Obj { si, phase: ObjPhase::Key(st), seen, extra });
                                true
                            }
                            Step::Closed => {
                                if !st.is_closed_interior() {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                }
                                let Ok(key) = String::from_utf8(st.dec.clone()) else {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                };
                                let s = &self.schemas[si as usize];
                                let child = match s.props.iter().position(|(n, _)| *n == key) {
                                    Some(i) => {
                                        if seen.contains(&(i as u32)) {
                                            m.stack.push(Frame::Dead);
                                            return false;
                                        }
                                        seen.push(i as u32);
                                        s.props[i].1
                                    }
                                    None => {
                                        if !s.additional || extra.iter().any(|k| *k == key) {
                                            m.stack.push(Frame::Dead);
                                            return false;
                                        }
                                        extra.push(key);
                                        0 // universal permissive schema
                                    }
                                };
                                m.stack.push(Frame::Obj {
                                    si,
                                    phase: ObjPhase::Colon { child },
                                    seen,
                                    extra,
                                });
                                true
                            }
                            Step::Bad => {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                    }
                    // After a `,` a key must follow: `}` here would be a trailing comma (invalid
                    // JSON), so it can never be emitted.
                    ObjPhase::KeyReq => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyReq, seen, extra });
                            return true;
                        }
                        if b == b'"' {
                            m.stack.push(Frame::Obj {
                                si,
                                phase: ObjPhase::Key(StrState::new()),
                                seen,
                                extra,
                            });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    ObjPhase::Colon { child } => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::Colon { child }, seen, extra });
                            return true;
                        }
                        if b == b':' {
                            // Consume the separator, then require a value above us.
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::AfterValue, seen, extra });
                            m.stack.push(Frame::Value { si: child });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                    ObjPhase::AfterValue => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::AfterValue, seen, extra });
                            return true;
                        }
                        if b == b',' && self.can_add_key(si, &seen) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyReq, seen, extra });
                            return true;
                        }
                        if b == b'}' {
                            if self.required_satisfied(si, &seen) {
                                return true;
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                },
                Frame::Arr { si, n, phase } => match phase {
                    ArrPhase::ItemOrEnd | ArrPhase::ItemReq => {
                        // Just after a `,` (`ItemReq`) a `]` would be a trailing comma: refuse it.
                        let need_item = matches!(phase, ArrPhase::ItemReq);
                        if is_ws(b) {
                            let here = if need_item { ArrPhase::ItemReq } else { ArrPhase::ItemOrEnd };
                            m.stack.push(Frame::Arr { si, n, phase: here });
                            return true;
                        }
                        if b == b']' {
                            let s = &self.schemas[si as usize];
                            if !need_item && s.min_items.map_or(true, |m0| n >= m0) {
                                return true; // array closed
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        let s = &self.schemas[si as usize];
                        if let Some(mx) = s.max_items {
                            if n >= mx {
                                m.stack.push(Frame::Dead);
                                return false;
                            }
                        }
                        let item_si = s.items.unwrap_or(0);
                        m.stack.push(Frame::Arr { si, n: n + 1, phase: ArrPhase::AfterItem });
                        m.stack.push(Frame::Value { si: item_si });
                        continue; // same byte starts the item
                    }
                    ArrPhase::AfterItem => {
                        if is_ws(b) {
                            m.stack.push(Frame::Arr { si, n, phase: ArrPhase::AfterItem });
                            return true;
                        }
                        // A `,` must leave room for one more item (maxItems), or the array can only
                        // dead-end.
                        if b == b',' && self.schemas[si as usize].max_items.map_or(true, |mx| n < mx) {
                            m.stack.push(Frame::Arr { si, n, phase: ArrPhase::ItemReq });
                            return true;
                        }
                        if b == b']' {
                            let s = &self.schemas[si as usize];
                            if s.min_items.map_or(true, |m0| n >= m0) {
                                return true;
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                },
                Frame::Str { si, mut st } => {
                    let keep = !self.schemas[si as usize].enum_str.is_empty();
                    match str_advance(&mut st, b, keep) {
                        Step::Open => {
                            if self.str_value_ok(si, &st, false) {
                                m.stack.push(Frame::Str { si, st });
                                true
                            } else {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                        Step::Closed => {
                            if self.str_value_ok(si, &st, true) {
                                true
                            } else {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                        Step::Bad => {
                            m.stack.push(Frame::Dead);
                            false
                        }
                    }
                }
                Frame::Lit { rest } => {
                    if rest.is_empty() || rest[0] != b {
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    if rest.len() == 1 {
                        return true; // literal complete
                    }
                    m.stack.push(Frame::Lit { rest: &rest[1..] });
                    true
                }
                Frame::Num { si, mut buf, st } => {
                    if let Some(next) = num_advance(st, b) {
                        buf.push(b);
                        // W2: refuse a digit that puts the number permanently out of range.
                        if !self.num_prefix_viable(si, &buf, next) {
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Num { si, buf, st: next });
                        return true;
                    }
                    // The number cannot continue: validate it, then hand the byte to the parent.
                    if self.num_ok(si, &buf, st) {
                        continue;
                    }
                    m.stack.push(Frame::Dead);
                    false
                }
            };
        }
    }
}

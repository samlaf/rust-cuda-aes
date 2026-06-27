//! AES constant tables, generated at compile time from first principles.
//!
//! Everything here is derived from the GF(2^8) field, so there are no
//! hand-transcribed magic-number tables that could carry a silent typo:
//!   - the S-box is the multiplicative inverse in GF(2^8) followed by the AES
//!     affine transform;
//!   - `T0[x] = [2.S[x], S[x], S[x], 3.S[x]]` and `T1/T2/T3` are byte rotations
//!     of `T0` (paper, Section IV-A);
//!   - `RCON[i] = 2^i` in GF(2^8).
//!
//! Because these are `const`, the values are computed by the compiler and baked
//! into the output as plain constant data — identical to hardcoding them, but
//! checked by the tests below.

// ---------------------------------------------------------------------------
// GF(2^8) arithmetic (the AES field, modulus x^8 + x^4 + x^3 + x + 1 = 0x11b).
// ---------------------------------------------------------------------------

/// Multiply two elements of GF(2^8).
const fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // reduce by the AES modulus
        }
        b >>= 1;
        i += 1;
    }
    p
}

/// Multiplicative inverse in GF(2^8) (0 maps to 0), by brute-force search.
const fn ginv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let mut b = 1u16;
    while b < 256 {
        if gmul(a, b as u8) == 1 {
            return b as u8;
        }
        b += 1;
    }
    0
}

// ---------------------------------------------------------------------------
// S-box: inverse in GF(2^8) followed by the AES affine transform.
// ---------------------------------------------------------------------------

const fn build_sbox() -> [u8; 256] {
    let mut s = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let b = ginv(i as u8);
        // affine: b ^ (b<<<1) ^ (b<<<2) ^ (b<<<3) ^ (b<<<4) ^ 0x63
        s[i] = b
            ^ b.rotate_left(1)
            ^ b.rotate_left(2)
            ^ b.rotate_left(3)
            ^ b.rotate_left(4)
            ^ 0x63;
        i += 1;
    }
    s
}

/// AES S-box (bytes). Used by the host key schedule.
pub const SBOX: [u8; 256] = build_sbox();

/// AES S-box widened to u32 (value `S[x]` in the low byte). Uploaded to the GPU
/// and used for the last round; keeping it u32 avoids any byte-slice ABI
/// concerns when passing it to the kernel.
pub const SBOX_U32: [u32; 256] = {
    let mut s = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        s[i] = SBOX[i] as u32;
        i += 1;
    }
    s
};

// ---------------------------------------------------------------------------
// T-tables. T0[x] = [2.S[x], S[x], S[x], 3.S[x]] (MSB..LSB), and
// T1/T2/T3 are byte rotations of T0:
//   T1 = ROTR(T0, 8), T2 = ROTR(T0, 16), T3 = ROTR(T0, 24).
// ---------------------------------------------------------------------------

const fn build_t0() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let s = SBOX[i];
        let s2 = gmul(s, 2);
        let s3 = gmul(s, 3);
        t[i] = ((s2 as u32) << 24) | ((s as u32) << 16) | ((s as u32) << 8) | (s3 as u32);
        i += 1;
    }
    t
}

const fn rotr_table(src: [u32; 256], r: u32) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = src[i].rotate_right(r);
        i += 1;
    }
    t
}

pub const T0: [u32; 256] = build_t0();
pub const T1: [u32; 256] = rotr_table(T0, 8);
pub const T2: [u32; 256] = rotr_table(T0, 16);
pub const T3: [u32; 256] = rotr_table(T0, 24);

/// AES-128 round constants (`RCON[i] = 2^i` in GF(2^8)).
pub const RCON: [u8; 10] = {
    let mut r = [0u8; 10];
    r[0] = 1;
    let mut i = 1;
    while i < 10 {
        r[i] = gmul(r[i - 1], 2);
        i += 1;
    }
    r
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmul_basics() {
        // Identity and the canonical xtime reduction example.
        assert_eq!(gmul(0x01, 0x57), 0x57);
        assert_eq!(gmul(0x57, 0x83), 0xc1); // FIPS-197 Section 4.2 worked example
        assert_eq!(gmul(0x80, 0x02), 0x1b); // high bit set -> reduce by 0x11b
    }

    #[test]
    fn sbox_known_values() {
        // A handful of published AES S-box entries.
        assert_eq!(SBOX[0x00], 0x63); // = affine(0), fixes the constant term
        assert_eq!(SBOX[0x01], 0x7c);
        assert_eq!(SBOX[0x10], 0xca);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(SBOX[0x7a], 0xda);
        assert_eq!(SBOX[0xff], 0x16);
    }

    #[test]
    fn sbox_is_a_permutation() {
        // A correct S-box is a bijection on 0..=255.
        let mut seen = [false; 256];
        for &v in SBOX.iter() {
            assert!(!seen[v as usize], "duplicate S-box output {v:#04x}");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn rcon_values() {
        assert_eq!(
            RCON,
            [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
        );
    }

    #[test]
    fn t0_matches_published_te0() {
        // Independent known answers from the published Rijndael Te0 table.
        assert_eq!(T0[0x00], 0xc66363a5);
        assert_eq!(T0[0x01], 0xf87c7c84);
        assert_eq!(T0[0xff], 0x2c16163a);
    }

    #[test]
    fn t0_structure() {
        // T0[x] must be exactly [2.S[x], S[x], S[x], 3.S[x]].
        for x in 0..256 {
            let s = SBOX[x] as u32;
            let s2 = gmul(SBOX[x], 2) as u32;
            let s3 = gmul(SBOX[x], 3) as u32;
            assert_eq!(T0[x], (s2 << 24) | (s << 16) | (s << 8) | s3);
        }
    }

    #[test]
    fn tables_are_rotations_of_t0() {
        for x in 0..256 {
            assert_eq!(T1[x], T0[x].rotate_right(8));
            assert_eq!(T2[x], T0[x].rotate_right(16));
            assert_eq!(T3[x], T0[x].rotate_right(24));
        }
    }

    #[test]
    fn sbox_u32_matches_sbox() {
        for x in 0..256 {
            assert_eq!(SBOX_U32[x], SBOX[x] as u32);
        }
    }
}

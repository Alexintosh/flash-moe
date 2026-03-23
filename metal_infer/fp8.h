// fp8.h -- FP8 E4M3 encode/decode for KV cache quantization
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m -- do NOT compile separately.
//
// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Exponent bias: 7.  Range: [-448, 448].  NaN: 0x7F.
// Per-layer dynamic scale: scale = absmax / 240.0 (headroom below 448 max).

// ============================================================================
// Global flag: opt-in FP8 KV cache (default off, float32 path unchanged)
// ============================================================================

static int g_use_fp8_kv = 0;

// ============================================================================
// FP8 E4M3 encode: float -> uint8_t (with per-tensor scale)
// ============================================================================

static inline uint8_t fp8_e4m3_encode(float x, float inv_scale) {
    float scaled = x * inv_scale;
    // Clamp to representable FP8 E4M3 range
    if (scaled != scaled) return 0x7F;  // NaN -> FP8 NaN
    scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);

    // Decompose into sign + magnitude
    uint32_t bits;
    memcpy(&bits, &scaled, sizeof(bits));
    uint8_t sign = (bits >> 31) & 1;
    float mag = fabsf(scaled);

    if (mag < 0.001953125f) {  // below smallest normal (2^-6 = 0.015625) sub-normals
        // Subnormals: exponent=0, mantissa encodes value / 2^-9
        // Smallest subnormal: 2^-9 = 0.001953125
        int m = (int)(mag / 0.001953125f + 0.5f);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | m);
    }

    // Normal: extract exponent and mantissa via log2
    int exp_unbiased = (int)floorf(log2f(mag));
    if (exp_unbiased < -6) exp_unbiased = -6;   // min exponent
    if (exp_unbiased > 8)  exp_unbiased = 8;    // max exponent (bias=7, stored=15)

    float frac = mag / powf(2.0f, (float)exp_unbiased) - 1.0f;  // in [0, 1)
    int mantissa = (int)(frac * 8.0f + 0.5f);  // 3 mantissa bits
    if (mantissa > 7) { mantissa = 0; exp_unbiased++; }  // carry

    int exp_biased = exp_unbiased + 7;  // bias = 7
    if (exp_biased < 0) { exp_biased = 0; mantissa = 0; }
    if (exp_biased > 15) { exp_biased = 15; mantissa = 6; }  // max normal (not 7, that's NaN with exp=15)

    // Special: exp=15 mantissa=7 is NaN in E4M3 -- clamp to max finite
    if (exp_biased == 15 && mantissa >= 7) mantissa = 6;

    return (uint8_t)((sign << 7) | (exp_biased << 3) | mantissa);
}

// ============================================================================
// FP8 E4M3 decode: uint8_t -> float (with per-tensor scale)
// ============================================================================

static inline float fp8_e4m3_decode(uint8_t x, float scale) {
    if (x == 0x7F) return __builtin_nanf("");  // NaN

    uint8_t sign = (x >> 7) & 1;
    uint8_t exp_biased = (x >> 3) & 0xF;
    uint8_t mantissa = x & 0x7;

    float val;
    if (exp_biased == 0) {
        // Subnormal: val = mantissa * 2^(1-bias-mantissa_bits) = mantissa * 2^-9
        val = (float)mantissa * 0.001953125f;  // 2^-9
    } else {
        // Normal: val = (1 + mantissa/8) * 2^(exp_biased - bias)
        val = (1.0f + (float)mantissa / 8.0f) * powf(2.0f, (float)exp_biased - 7.0f);
    }

    if (sign) val = -val;
    return val * scale;
}

// ============================================================================
// Compute absmax of a float vector (for dynamic scale derivation)
// ============================================================================

static inline float fp8_absmax(const float *vec, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(vec[i]);
        if (a > amax) amax = a;
    }
    return amax;
}

// ============================================================================
// Batch encode: float vector -> FP8 E4M3 with dynamic scale
// Returns scale (absmax / 240.0). Caller stores scale separately.
// ============================================================================

static inline float fp8_encode_vec(const float *src, uint8_t *dst, int n) {
    float amax = fp8_absmax(src, n);
    float scale = (amax > 0.0f) ? (amax / 240.0f) : 1.0f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_encode(src[i], inv_scale);
    }
    return scale;
}

// ============================================================================
// Batch decode: FP8 E4M3 -> float with scale
// ============================================================================

static inline void fp8_decode_vec(const uint8_t *src, float *dst, int n, float scale) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_decode(src[i], scale);
    }
}

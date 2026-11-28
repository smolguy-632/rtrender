// RT Renderer kernel: one work item per pixel.
//   primary ray  -> sparse 64-tree hierarchical Amanatides&Woo DDA (same traversal that won the benchmark)
//   textures     -> block atlas sampled with nearest filtering, biome tints, cutout + translucent layers
//   shadows      -> 1 ray to the sun/moon + 1 ray to the strongest nearby light block (optional)
//   water        -> mirror reflection (Fresnel, animated ripple normal) + refraction with depth absorption
// No recursion (OpenCL C forbids it): water surfaces are handed back to the kernel body, which traces the
// reflection and refraction rays with trace_color() in modes 1 and 2.
//
// Face index: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z      (face = 2*axis + (ray moves in +axis ? 1 : 0))
// mats entry: bits 0..14 palette id (0 = air), bit 15 = "exposed water surface"

#define LEVELS 4
#define TREE_SIZE 256.0f

#define F_OPAQUE 1u
#define F_CUTOUT 2u
#define F_TRANSL 4u
#define F_WATER  8u
#define F_FULL   16u
#define F_EMIT   32u
#define F_SHADOW 64u

// ---- parameter block layout (float array P) ----
#define P_INVVP   0     // 16, column-major, maps NDC -> (world - eye) space
#define P_VP      16    // 16, column-major, (world - eye) -> clip
#define P_EYE     32    // 3  eye in window-local voxel coords
#define P_SUNDIR  35    // 3
#define P_SUNCOL  38    // 3
#define P_ZENITH  41    // 3
#define P_FOG     44    // 3
#define P_FOGSTART 47
#define P_FOGEND  48
#define P_TIME    49
#define P_MAXDIST 50
#define P_SHADOWDIST 51
#define P_WAVE    52
#define P_SKYAMB  53
#define P_BLKDARK 54
#define P_SUNSHADOWS 55
#define P_WATERREFL 56
#define P_LIGHT 57
#define P_HEATMAP 58
#define P_YTOP 59       // exclusive top of the highest occupied 16-block section (window-local y)
// node layout: uint4 { maskLo, maskHi, firstChildPtr, 0 } - one 16-byte load per visited node, 32-bit mask math only
// ---- int block IP: 0 w, 1 h, 2 originX, 3 originZ, 4 atlasW, 5 atlasH, 6 blockShadows ----

#define SA __global const uint4* nodes, __global const ushort* mats, \
           __global const uchar* palFlags, __global const ushort* palUV, __global const uint* palTint, \
           __global const ushort* palBox, __global const uint* colTint, \
           __global const int* emHead, __global const int* emNext, __global const float4* emPos, \
           __global const uint* atlas, __global const float* P, __global const int* IP
#define SP nodes, mats, palFlags, palUV, palTint, palBox, colTint, emHead, emNext, emPos, atlas, P, IP

// =====================================================================================================
// sparse 64-tree resumable traversal
// =====================================================================================================
typedef struct {
    float3 o, d, inv, ainv;
    int sx, sy, sz;
    float t, t_end;
    int level, cs, node, ox, oy, oz, cx, cy, cz;
    float tx, ty, tz;
    uint2 m;
    int s_node[LEVELS], s_cell[LEVELS];      // per-level stack: node index and packed cell (cx | cy<<2 | cz<<4); t values are recomputed on ascent
    int axis;
    int done, pend;
} Tr;

inline void tr_cell(Tr* s)
{
    float3 p = s->o + s->d * s->t;
    float rcs = 1.0f / (float)s->cs;     // cs is a power of two -> exact reciprocal, same result as the division
    s->cx = clamp((int)((p.x - (float)s->ox) * rcs), 0, 3);
    s->cy = clamp((int)((p.y - (float)s->oy) * rcs), 0, 3);
    s->cz = clamp((int)((p.z - (float)s->oz) * rcs), 0, 3);
    s->tx = (s->ox + (s->cx + (s->sx > 0)) * s->cs - s->o.x) * s->inv.x;
    s->ty = (s->oy + (s->cy + (s->sy > 0)) * s->cs - s->o.y) * s->inv.y;
    s->tz = (s->oz + (s->cz + (s->sz > 0)) * s->cs - s->o.z) * s->inv.z;
}

inline void tr_init(Tr* s, __global const uint4* nodes, float3 o, float3 d, float max_t, float ytop)
{
    if (fabs(d.x) < 1e-6f) d.x = copysign(1e-6f, d.x);
    if (fabs(d.y) < 1e-6f) d.y = copysign(1e-6f, d.y);
    if (fabs(d.z) < 1e-6f) d.z = copysign(1e-6f, d.z);
    s->o = o; s->d = d; s->inv = 1.0f / d; s->ainv = fabs(s->inv);
    s->sx = d.x > 0.0f ? 1 : -1; s->sy = d.y > 0.0f ? 1 : -1; s->sz = d.z > 0.0f ? 1 : -1;
    float3 t0 = (0.0f - o) * s->inv, t1 = (TREE_SIZE - o) * s->inv;
    float3 tn = fmin(t0, t1), tf = fmax(t0, t1);
    float tmin = fmax(fmax(tn.x, tn.y), tn.z);
    s->t = fmax(tmin, 0.0f);
    s->t_end = fmin(fmin(tf.x, tf.y), fmin(tf.z, max_t));
    // nothing is occupied above ytop: an upward ray can stop as soon as it climbs past it (exact, saves the empty-sky walk)
    if (d.y > 0.0f) s->t_end = fmin(s->t_end, (ytop - o.y) * s->inv.y);
    s->done = (s->t >= s->t_end); s->pend = 0;
    s->level = LEVELS - 1; s->cs = 64; s->node = 0; s->ox = 0; s->oy = 0; s->oz = 0;
    s->m = nodes[0].xy;
    s->axis = -1;
    if (tmin > 0.0f) s->axis = (tn.x >= tn.y && tn.x >= tn.z) ? 0 : ((tn.y >= tn.z) ? 1 : 2);
    tr_cell(s);
}

// step to the next cell of the current level (ascending as needed). returns 0 when the ray left the volume.
inline int tr_advance(Tr* s, __global const uint4* nodes)
{
    int axis;
    if (s->tx < s->ty) { axis = (s->tx < s->tz) ? 0 : 2; } else { axis = (s->ty < s->tz) ? 1 : 2; }
    s->t = (axis == 0) ? s->tx : ((axis == 1) ? s->ty : s->tz);
    if (s->t >= s->t_end) return 0;
    s->axis = axis;
    int out;
    if (axis == 0)      { s->cx += s->sx; s->tx += s->cs * s->ainv.x; out = (s->cx < 0 || s->cx > 3); }
    else if (axis == 1) { s->cy += s->sy; s->ty += s->cs * s->ainv.y; out = (s->cy < 0 || s->cy > 3); }
    else                { s->cz += s->sz; s->tz += s->cs * s->ainv.z; out = (s->cz < 0 || s->cz > 3); }
    while (out) {
        if (s->level == LEVELS - 1) return 0;
        int l = ++s->level; s->cs <<= 2;
        s->node = s->s_node[l]; s->m = nodes[s->node].xy;
        int pc = s->s_cell[l];
        int pcx = pc & 3, pcy = (pc >> 2) & 3, pcz = pc >> 4;
        s->ox -= pcx * s->cs; s->oy -= pcy * s->cs; s->oz -= pcz * s->cs;
        if (axis == 0) pcx += s->sx; else if (axis == 1) pcy += s->sy; else pcz += s->sz;
        s->cx = pcx; s->cy = pcy; s->cz = pcz;
        // exit distances of the parent cell, recomputed (replaces the saved tx/ty/tz of the old stack)
        s->tx = (s->ox + (pcx + (s->sx > 0)) * s->cs - s->o.x) * s->inv.x;
        s->ty = (s->oy + (pcy + (s->sy > 0)) * s->cs - s->o.y) * s->inv.y;
        s->tz = (s->oz + (pcz + (s->sz > 0)) * s->cs - s->o.z) * s->inv.z;
        out = (pcx < 0 || pcx > 3 || pcy < 0 || pcy > 3 || pcz < 0 || pcz > 3);
    }
    return 1;
}

// Returns the next occupied voxel along the ray (any non-zero mats entry) and moves the state past it.
inline int tr_next(Tr* s, __global const uint4* nodes, __global const ushort* mats,
                   int* vx, int* vy, int* vz, uint* mat, float* tin, int* axis_out)
{
    if (s->pend) { s->pend = 0; if (!tr_advance(s, nodes)) { s->done = 1; return 0; } }
    for (int it = 0; it < 256; it++) {
        if (s->done) return 0;
        int bit = s->cx + s->cy * 4 + s->cz * 16;
        uint hiHalf = (uint)bit >> 5, b0 = (uint)bit & 31u;
        uint word = hiHalf ? s->m.y : s->m.x;
        if ((word >> b0) & 1u) {
            // popcount of the set bits below `bit` using 32-bit ops only (64-bit shifts are emulated on many GPUs)
            uint lm = (1u << b0) - 1u;
            uint below = hiHalf ? (popcount(s->m.x) + popcount(s->m.y & lm)) : popcount(s->m.x & lm);
            uint c = nodes[s->node].z + below;
            if (s->level == 0) {
                *mat = (uint)mats[c];
                *vx = s->ox + s->cx; *vy = s->oy + s->cy; *vz = s->oz + s->cz;
                *tin = s->t; *axis_out = s->axis;
                s->pend = 1;
                return 1;
            }
            int l = s->level;
            s->s_node[l] = s->node; s->s_cell[l] = s->cx | (s->cy << 2) | (s->cz << 4);
            s->ox += s->cx * s->cs; s->oy += s->cy * s->cs; s->oz += s->cz * s->cs;
            s->level--; s->cs >>= 2; s->node = (int)c; s->m = nodes[s->node].xy;
            tr_cell(s);
            continue;
        }
        if (!tr_advance(s, nodes)) { s->done = 1; return 0; }
    }
    s->done = 1;
    return 0;
}

// =====================================================================================================
// helpers
// =====================================================================================================
inline float3 vload3p(__global const float* P, int i) { return (float3)(P[i], P[i + 1], P[i + 2]); }

inline float3 face_normal(int f)
{
    switch (f) {
        case 0: return (float3)(1, 0, 0);
        case 1: return (float3)(-1, 0, 0);
        case 2: return (float3)(0, 1, 0);
        case 3: return (float3)(0, -1, 0);
        case 4: return (float3)(0, 0, 1);
        default: return (float3)(0, 0, -1);
    }
}

inline void face_uv(int f, float3 fr, float* u, float* v)
{
    switch (f) {
        case 2: *u = fr.x;        *v = fr.z;        break;   // up
        case 3: *u = fr.x;        *v = 1.0f - fr.z; break;   // down
        case 4: *u = fr.x;        *v = 1.0f - fr.y; break;   // south
        case 5: *u = 1.0f - fr.x; *v = 1.0f - fr.y; break;   // north
        case 0: *u = 1.0f - fr.z; *v = 1.0f - fr.y; break;   // east
        default: *u = fr.z;       *v = 1.0f - fr.y; break;   // west
    }
}

inline float4 sample_atlas(__global const uint* atlas, int aw, int ah, float4 r, float u, float v)
{
    float uu = mix(r.x, r.z, clamp(u, 0.0f, 0.999f)), vv = mix(r.y, r.w, clamp(v, 0.0f, 0.999f));
    int px = clamp((int)(uu * aw), 0, aw - 1), py = clamp((int)(vv * ah), 0, ah - 1);
    uint c = atlas[py * aw + px];
    return (float4)((float)(c & 255u), (float)((c >> 8) & 255u), (float)((c >> 16) & 255u), (float)(c >> 24)) * (1.0f / 255.0f);
}

// Find the leaf index in the 64-tree for voxel (vx,vy,vz) (local coords), or -1 if air/empty.
inline uchar get_light(int x, int y, int z, int ox, int oz)
{
    if (y < 0) return (uchar)0;
    if (y > 255) return (uchar)0xFF;
    return (uchar)0xF0;
}

// Does the ray really hit this voxel's geometry? (full cube: yes; partial block: bounding box; cutout: alpha test)
inline int eval_voxel(SA, float3 o, float3 d, int vx, int vy, int vz, int axis, float tin, uint id,
                      float* tHit, int* faceOut, float4* texOut)
{
    uint fl = palFlags[id];
    float3 vo = (float3)((float)vx, (float)vy, (float)vz);
    float th; int face;
    if (fl & F_FULL) {
        th = tin;
        if (axis < 0) face = 2;
        else {
            float da = (axis == 0) ? d.x : ((axis == 1) ? d.y : d.z);
            face = 2 * axis + (da > 0.0f ? 1 : 0);
        }
    } else {
        float3 bmin = (float3)((float)palBox[id * 6] / 65535.0f, (float)palBox[id * 6 + 1] / 65535.0f, (float)palBox[id * 6 + 2] / 65535.0f);
        float3 bmax = (float3)((float)palBox[id * 6 + 3] / 65535.0f, (float)palBox[id * 6 + 4] / 65535.0f, (float)palBox[id * 6 + 5] / 65535.0f);
        float3 ol = o - vo;
        float3 inv = 1.0f / ((float3)(fabs(d.x) < 1e-6f ? copysign(1e-6f, d.x) : d.x,
                                      fabs(d.y) < 1e-6f ? copysign(1e-6f, d.y) : d.y,
                                      fabs(d.z) < 1e-6f ? copysign(1e-6f, d.z) : d.z));
        float3 t0 = (bmin - ol) * inv, t1 = (bmax - ol) * inv;
        float3 tn = fmin(t0, t1), tf = fmax(t0, t1);
        float tnear = fmax(fmax(tn.x, tn.y), tn.z), tfar = fmin(fmin(tf.x, tf.y), tf.z);
        if (tnear > tfar || tfar < tin - 1e-4f) return 0;
        th = fmax(tnear, 0.0f);
        int ax = (tn.x >= tn.y && tn.x >= tn.z) ? 0 : ((tn.y >= tn.z) ? 1 : 2);
        float da = (ax == 0) ? d.x : ((ax == 1) ? d.y : d.z);
        face = 2 * ax + (da > 0.0f ? 1 : 0);
        if (tnear < tin - 1e-3f) { face = 2; }          // started inside the box
    }
    float3 hp = o + d * th;
    float3 fr = clamp(hp - vo, 0.0f, 1.0f);
    float u, v; face_uv(face, fr, &u, &v);
    int _uvb = (id * 6 + face) * 4;
    float4 _uvr = (float4)((float)palUV[_uvb] / 65535.0f, (float)palUV[_uvb + 1] / 65535.0f,
                          (float)palUV[_uvb + 2] / 65535.0f, (float)palUV[_uvb + 3] / 65535.0f);
    float4 tx = sample_atlas(atlas, IP[4], IP[5], _uvr, u, v);
    uint tt = palTint[id * 6 + face];
    uint tp = tt >> 24;
    if (tp != 0u) {
        float3 tc;
        if (tp == 4u) {
            tc = (float3)((float)((tt >> 16) & 255u), (float)((tt >> 8) & 255u), (float)(tt & 255u)) * (1.0f / 255.0f);
        } else {
            int slot = (((vx + IP[2]) >> 4) & 15) | ((((vz + IP[3]) >> 4) & 15) << 4);
            uint c = colTint[(tp - 1u) * 256u + (uint)slot];
            tc = (float3)((float)((c >> 16) & 255u), (float)((c >> 8) & 255u), (float)(c & 255u)) * (1.0f / 255.0f);
        }
        tx.xyz *= tc;
    }
    if ((fl & F_CUTOUT) && tx.w < 0.5f) return 0;
    if (!(fl & F_TRANSL)) tx.w = 1.0f;
    *tHit = th; *faceOut = face; *texOut = tx;
    return 1;
}

// any-hit visibility test (1 = unobstructed). Translucent blocks only dim the light.
inline float shadow_vis(SA, float3 o, float3 d, float maxT)
{
    Tr s; tr_init(&s, nodes, o, d, maxT, P[P_YTOP]);
    float vis = 1.0f;
    int vx, vy, vz, axis; uint mat; float tin;
    while (tr_next(&s, nodes, mats, &vx, &vy, &vz, &mat, &tin, &axis)) {
        uint id = mat & 0x7fffu;
        uint fl = palFlags[id];
        if (fl & F_WATER) { vis *= 0.9f; continue; }
        if (fl & F_TRANSL) { vis *= 0.8f; continue; }
        if (!(fl & F_SHADOW)) continue;
        if ((fl & F_FULL) && !(fl & F_CUTOUT)) return 0.0f;      // eval_voxel would always hit: skip the atlas fetch
        float th; int face; float4 tex;
        if (eval_voxel(SP, o, d, vx, vy, vz, axis, tin, id, &th, &face, &tex)) return 0.0f;
    }
    return vis;
}

inline float3 sky_color(__global const float* P, float3 d)
{
    float3 zen = vload3p(P, P_ZENITH), fogc = vload3p(P, P_FOG);
    float h = clamp(d.y, 0.0f, 1.0f);
    float3 c = mix(fogc, zen, pow(h, 0.55f));
    float sd = fmax(dot(d, vload3p(P, P_SUNDIR)), 0.0f);
    if (sd > 0.9f) c += vload3p(P, P_SUNCOL) * pow(sd, 400.0f) * 6.0f;
    return c;
}

// lit colour of a surface point (vanilla light levels + ray traced sun / light-block shadows)
inline float3 shade(SA, float3 hp, int face, float3 base, uint id, int vx, int vy, int vz)
{
    uint fl = palFlags[id];
    if (fl & F_EMIT) return base * 1.15f;
    float3 n = face_normal(face);
    int ox = IP[2], oz = IP[3];
    uchar la, lb;
    if (P[P_LIGHT] > 0.5f) {
        la = get_light(vx + (int)n.x, vy + (int)n.y, vz + (int)n.z, ox, oz);
        lb = get_light(vx, vy, vz, ox, oz);
    } else {
        la = (uchar)0xF0; lb = (uchar)0xF0;
    }
    float skyF = (float)(max((int)(la >> 4), (int)(lb >> 4))) * (1.0f / 15.0f);
    float blkF = (float)(max((int)(la & 15), (int)(lb & 15))) * (1.0f / 15.0f);

    float faceShade = (face == 2) ? 1.0f : ((face == 3) ? 0.5f : ((face < 2) ? 0.6f : 0.8f));
    float skyAmb = P[P_SKYAMB];
    float3 sunDir = vload3p(P, P_SUNDIR), sunCol = vload3p(P, P_SUNCOL);
    float ndl = dot(n, sunDir);
    float vis = 0.0f;
    if (P[P_SUNSHADOWS] > 0.5f && ndl > 0.0f && skyF > 0.02f && (sunCol.x + sunCol.y + sunCol.z) > 0.0f)
        vis = shadow_vis(SP, hp + n * 0.01f, sunDir, P[P_SHADOWDIST]);
    else if (P[P_SUNSHADOWS] <= 0.5f)
        vis = (ndl > 0.0f) ? 1.0f : 0.0f;

    float3 ambPart = (float3)(0.88f, 0.94f, 1.0f) * (skyAmb * faceShade * skyF);
    float3 sunPart = sunCol * (fmax(ndl, 0.0f) * vis * skyF);

    float3 warm = (float3)(1.0f, 0.78f, 0.45f);
    float bl = (blkF > 0.0f) ? pow(blkF, 1.6f) * 1.15f : 0.0f;
    float blkPart = bl * faceShade;
    if (IP[6] != 0 && blkF > 0.04f) {
        int csx = (int)floor(hp.x) >> 4, csy = (int)floor(hp.y) >> 4, csz = (int)floor(hp.z) >> 4;
        float4 best = (float4)(0.0f); float bestS = 0.0f;
        for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
            int sx = csx + dx, sy = csy + dy, sz = csz + dz;
            if (sx < 0 || sx > 15 || sy < 0 || sy > 15 || sz < 0 || sz > 15) continue;
            int guard = 0;
            for (int e = emHead[(sx << 8) | (sy << 4) | sz]; e >= 0 && guard < 10; e = emNext[e], guard++) {
                float4 ep = emPos[e];
                float3 L = ep.xyz - hp;
                float d2 = dot(L, L);
                if (d2 < 0.25f || d2 > 324.0f) continue;
                float nd = dot(n, L) * rsqrt(d2);
                if (nd <= 0.05f) continue;
                float sc = ep.w * nd / fmax(1.0f, d2);
                if (sc > bestS) { bestS = sc; best = ep; }
            }
        }
        if (best.w > 0.5f) {
            float3 L = best.xyz - hp;
            float dist = length(L);
            L /= dist;
            float v2 = shadow_vis(SP, hp + n * 0.01f, L, dist - 0.6f);
            float direct = (best.w / 15.0f) * fmax(dot(n, L), 0.0f) * v2 / (1.0f + 0.06f * dist * dist);
            blkPart = fmax(blkPart * mix(1.0f - P[P_BLKDARK], 1.0f, v2), direct);
        }
    }
    float3 lit = ambPart + sunPart + warm * blkPart;
    return base * fmax(lit, (float3)(0.03f, 0.03f, 0.035f));
}

typedef struct { float3 col; float alpha; float t; int water; float3 hp; float ts; float3 acc; float T; int steps; } Res;

// mode 0 primary (hands water surfaces back), 1 reflection (water = flat colour, sky on miss),
// mode 2 refraction / underwater (water skipped, absorption applied)
inline Res trace_color(SA, float3 o, float3 d, float maxDist, int mode)
{
    Res r;
    r.col = (float3)(0.0f); r.alpha = 0.0f; r.t = -1.0f; r.water = 0; r.hp = (float3)(0.0f); r.ts = 0.0f;
    r.acc = (float3)(0.0f); r.T = 1.0f; r.steps = 0;
    float fogStart = P[P_FOGSTART], fogEnd = P[P_FOGEND];
    float3 fogc = vload3p(P, P_FOG);
    Tr s; tr_init(&s, nodes, o, d, maxDist, P[P_YTOP]);
    float3 acc = (float3)(0.0f); float T = 1.0f;
    int first = 1, under = 0, nt = 0;
    int vx, vy, vz, axis; uint mat; float tin;
    while (tr_next(&s, nodes, mats, &vx, &vy, &vz, &mat, &tin, &axis)) {
        r.steps++;
        uint id = mat & 0x7fffu;
        uint fl = palFlags[id];
        if (fl & F_WATER) {
            if (first && tin < 1e-3f && mode == 0) { under = 1; mode = 2; }
            first = 0;
            if (mode == 0 && (mat & 0x8000u)) {
                float ts = tin;
                if (axis == 1 && fabs(d.y) > 1e-4f) {
                    float tp = ((float)vy + 0.89f - o.y) / d.y;
                    if (tp > tin && tp < tin + 1.8f) ts = tp;
                }
                r.water = 1; r.ts = ts; r.hp = o + d * ts; r.acc = acc; r.T = T; r.t = ts;
                return r;
            }
            if (mode == 1 && (mat & 0x8000u)) {
                acc += T * (float3)(0.05f, 0.16f, 0.28f) * (0.35f + P[P_SKYAMB]);
                T = 0.0f; r.t = tin; break;
            }
            continue;
        }
        first = 0;
        float th; int face; float4 tex;
        if (!eval_voxel(SP, o, d, vx, vy, vz, axis, tin, id, &th, &face, &tex)) continue;
        float3 hp = o + d * th;
        float3 lit = shade(SP, hp, face, tex.xyz, id, vx, vy, vz);
        float f = clamp((th - fogStart) / fmax(fogEnd - fogStart, 1.0f), 0.0f, 1.0f);
        lit = mix(lit, fogc, f * f);
        if (fl & F_TRANSL) {
            float a = tex.w;
            if (a < 0.02f) continue;
            acc += T * a * lit; T *= (1.0f - a);
            if (T < 0.03f || ++nt > 6) { T = 0.0f; r.t = th; break; }
            continue;
        }
        acc += T * lit; T = 0.0f; r.t = th;
        break;
    }
    int hit = (T < 0.0001f);
    if (mode == 0) {
        r.col = acc; r.alpha = 1.0f - T;
    } else {
        float3 skyc = hit ? (float3)(0.0f) : sky_color(P, d);
        r.col = acc + T * skyc; r.alpha = 1.0f;
    }
    if (mode == 2 || under) {
        float dist = (r.t >= 0.0f) ? r.t : maxDist;
        float3 ab = exp(-(float3)(0.30f, 0.09f, 0.04f) * dist);
        float3 deep = (float3)(0.02f, 0.10f, 0.17f) * (0.25f + P[P_SKYAMB]);
        r.col = r.col * ab + deep * ((float3)(1.0f) - ab);
        if (under) r.alpha = 1.0f;
    }
    return r;
}

inline float3 wave_normal(float2 xz, float t, float amp)
{
    float dx = amp * (cos(xz.x * 1.7f + t * 1.3f) * 0.6f + cos((xz.x + xz.y) * 2.9f - t * 1.9f) * 0.4f);
    float dz = amp * (cos(xz.y * 1.9f - t * 1.1f) * 0.6f + cos((xz.x - xz.y) * 2.3f + t * 1.7f) * 0.4f);
    return normalize((float3)(-dx, 1.0f, -dz));
}

__kernel void render(
    __global const uint4* nodes, __global const ushort* mats,
    __global const uchar* palFlags, __global const ushort* palUV, __global const uint* palTint,
    __global const ushort* palBox, __global const uint* colTint,
    __global const int* emHead, __global const int* emNext, __global const float4* emPos,
    __global const uint* atlas, __global const float* P, __global const int* IP,
    __global uint* outColor, __global float* outDepth)
{
    // 8x8 pixel tiles: neighbouring lanes trace neighbouring pixels -> far less divergence / better cache hits than
    // a 1-D row-major mapping. Host launches tilesX*tilesY*64 work items (see RTEngine.render).
    int gid = get_global_id(0);
    int w = IP[0], h = IP[1];
    int tilesX = (w + 7) >> 3;
    int tile = gid >> 6, inT = gid & 63;
    int px = ((tile % tilesX) << 3) + (inT & 7);
    int py = ((tile / tilesX) << 3) + (inT >> 3);
    if (px >= w || py >= h) return;
    gid = py * w + px;                     // output index stays row-major
    float nx = ((float)px + 0.5f) / (float)w * 2.0f - 1.0f;
    float ny = ((float)py + 0.5f) / (float)h * 2.0f - 1.0f;

    float4 a = (float4)(nx, ny, -1.0f, 1.0f), b = (float4)(nx, ny, 1.0f, 1.0f);
    float4 pa, pb;
    pa.x = P[0] * a.x + P[4] * a.y + P[8]  * a.z + P[12] * a.w;
    pa.y = P[1] * a.x + P[5] * a.y + P[9]  * a.z + P[13] * a.w;
    pa.z = P[2] * a.x + P[6] * a.y + P[10] * a.z + P[14] * a.w;
    pa.w = P[3] * a.x + P[7] * a.y + P[11] * a.z + P[15] * a.w;
    pb.x = P[0] * b.x + P[4] * b.y + P[8]  * b.z + P[12] * b.w;
    pb.y = P[1] * b.x + P[5] * b.y + P[9]  * b.z + P[13] * b.w;
    pb.z = P[2] * b.x + P[6] * b.y + P[10] * b.z + P[14] * b.w;
    pb.w = P[3] * b.x + P[7] * b.y + P[11] * b.z + P[15] * b.w;
    float3 na = pa.xyz / pa.w, nb = pb.xyz / pb.w;
    float3 eye = vload3p(P, P_EYE);
    float3 o = eye + na;
    float3 d = normalize(nb - na);
    float maxDist = P[P_MAXDIST];

    Res r = trace_color(SP, o, d, maxDist, 0);
    float3 col = r.col; float alpha = r.alpha; float tdepth = r.t;

    // debug heatmap: colour by traversal step count (blue=few .. red=many)
    if (P[P_HEATMAP] > 0.5f) {
        float tt = clamp((float)r.steps / 64.0f, 0.0f, 1.0f);
        float3 hc;
        if (tt < 0.33f) hc = mix((float3)(0.0f,0.0f,1.0f), (float3)(0.0f,1.0f,0.0f), tt / 0.33f);
        else if (tt < 0.66f) hc = mix((float3)(0.0f,1.0f,0.0f), (float3)(1.0f,1.0f,0.0f), (tt - 0.33f) / 0.33f);
        else hc = mix((float3)(1.0f,1.0f,0.0f), (float3)(1.0f,0.0f,0.0f), (tt - 0.66f) / 0.34f);
        hc = clamp(hc, 0.0f, 1.0f);
        uint pr = (uint)(hc.x * 255.0f + 0.5f), pg = (uint)(hc.y * 255.0f + 0.5f), pbv = (uint)(hc.z * 255.0f + 0.5f);
        outColor[gid] = pr | (pg << 8) | (pbv << 16) | (255u << 24);
        outDepth[gid] = 1.0f;
        return;
    }

    // sky miss: no surface hit -> skip water handling and depth transform
    if (tdepth < 0.0f) {
        col = clamp(col, 0.0f, 1.0f);
        alpha = clamp(alpha, 0.0f, 1.0f);
        uint pr = (uint)(col.x * 255.0f + 0.5f), pg = (uint)(col.y * 255.0f + 0.5f), pbv = (uint)(col.z * 255.0f + 0.5f);
        uint pa8 = (uint)(alpha * 255.0f + 0.5f);
        outColor[gid] = pr | (pg << 8) | (pbv << 16) | (pa8 << 24);
        outDepth[gid] = 1.0f;
        return;
    }

    if (r.water) {
        float3 n = wave_normal(r.hp.xz, P[P_TIME], P[P_WAVE]);
        float cosi = clamp(dot(-d, n), 0.0f, 1.0f);
        float F = 0.02f + 0.98f * pow(1.0f - cosi, 5.0f);
        float3 refl;
        if (P[P_WATERREFL] > 0.5f) {
            float3 R = d - 2.0f * dot(d, n) * n;
            R.y = fabs(R.y);
            Res rl = trace_color(SP, r.hp + n * 0.03f, normalize(R), maxDist, 1);
            refl = rl.col;
        } else {
            refl = sky_color(P, (float3)(d.x, fabs(d.y), d.z));
        }
        float eta = 1.0f / 1.33f;
        float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
        float3 rd = normalize(eta * d + (eta * cosi - sqrt(fmax(k, 0.0f))) * n);
        Res rf = trace_color(SP, r.hp - n * 0.03f, rd, maxDist, 2);
        float3 wcol = rf.col * (1.0f - F) + refl * F;
        float f = clamp((r.ts - P[P_FOGSTART]) / fmax(P[P_FOGEND] - P[P_FOGSTART], 1.0f), 0.0f, 1.0f);
        wcol = mix(wcol, vload3p(P, P_FOG), f * f);
        col = r.acc + r.T * wcol; alpha = 1.0f; tdepth = r.ts;
    }

    float depth = 1.0f;
    if (tdepth >= 0.0f) {
        float3 rel = (o + d * tdepth) - eye;
        float cz = P[2 + 16] * rel.x + P[6 + 16] * rel.y + P[10 + 16] * rel.z + P[14 + 16];
        float cw = P[3 + 16] * rel.x + P[7 + 16] * rel.y + P[11 + 16] * rel.z + P[15 + 16];
        depth = clamp((cz / cw) * 0.5f + 0.5f, 0.0f, 1.0f);
    }
    col = clamp(col, 0.0f, 1.0f);
    alpha = clamp(alpha, 0.0f, 1.0f);
    uint pr = (uint)(col.x * 255.0f + 0.5f), pg = (uint)(col.y * 255.0f + 0.5f), pbv = (uint)(col.z * 255.0f + 0.5f);
    uint pa8 = (uint)(alpha * 255.0f + 0.5f);
    outColor[gid] = pr | (pg << 8) | (pbv << 16) | (pa8 << 24);
    outDepth[gid] = depth;
}

// =====================================================================================================
// GI bake: per-cell light for the rasterised terrain.
// Celeritas' vertex shader samples the result as a 3D texture, so lighting changes never need a remesh.
//
// One hemisphere ray per sample does double duty: rays that escape count as sky visibility, rays that
// hit a surface feed the single bounce of colour bleed. GP layout:
//   0 from, 1 to      linear cell range [from,to) of the dimX*dimY*dimZ volume
//   2 dimX, 3 dimY, 4 cell size (blocks)
//   5..7 volume origin in world blocks
//   8 rays per cell   9 flags: 1 = sun shadows, 2 = bounce, 4 = emitters
// =====================================================================================================

inline float3 gi_dir6(int d)
{
    if (d == 0) return (float3)(1.0f, 0.0f, 0.0f);
    if (d == 1) return (float3)(-1.0f, 0.0f, 0.0f);
    if (d == 2) return (float3)(0.0f, 1.0f, 0.0f);
    if (d == 3) return (float3)(0.0f, -1.0f, 0.0f);
    if (d == 4) return (float3)(0.0f, 0.0f, 1.0f);
    return (float3)(0.0f, 0.0f, -1.0f);
}

// Cosine-weighted hemisphere direction around +Y, low-discrepancy in the azimuth.
inline float3 gi_hemi(int i, int n)
{
    float xi = ((float)i + 0.5f) / (float)n;
    float st = sqrt(xi), ct = sqrt(fmax(0.0f, 1.0f - xi));
    float ph = (float)i * 2.39996323f;
    return (float3)(st * cos(ph), ct, st * sin(ph));
}

// Is window-local voxel (x,y,z) occupied? Four mask lookups, no ray traversal.
inline int gi_occupied(SA, int x, int y, int z)
{
    if (x < 0 || x > 255 || y < 0 || y > 255 || z < 0 || z > 255) return 0;
    int node = 0, ox = 0, oy = 0, oz = 0, cs = 64;
    for (int l = LEVELS - 1; l >= 0; l--) {
        int cx = (x - ox) / cs, cy = (y - oy) / cs, cz = (z - oz) / cs;
        int bit = cx + cy * 4 + cz * 16;
        uint hiHalf = (uint)bit >> 5, b0 = (uint)bit & 31u;
        uint word = hiHalf ? nodes[node].y : nodes[node].x;
        if (!((word >> b0) & 1u)) return 0;
        if (l == 0) return 1;
        uint lm = (1u << b0) - 1u;
        uint below = hiHalf ? (popcount(nodes[node].x) + popcount(nodes[node].y & lm)) : popcount(nodes[node].x & lm);
        ox += cx * cs; oy += cy * cs; oz += cz * cs;
        cs >>= 2; node = (int)(nodes[node].z + below);
    }
    return 1;
}

// Light leaving a surface point towards the sky and the sun (used by the bounce term).
inline float3 gi_surface_light(SA, float3 fp)
{
    float3 skyTint = (float3)(0.88f, 0.94f, 1.0f);
    float skyAmb = P[P_SKYAMB];
    float3 sunDir = vload3p(P, P_SUNDIR), sunCol = vload3p(P, P_SUNCOL);
    float hitSky = shadow_vis(SP, fp + (float3)(0.0f, 0.02f, 0.0f), (float3)(0.0f, 1.0f, 0.0f), P[P_SHADOWDIST]);
    float hitSun = 0.0f;
    if (hitSky > 0.0f && sunDir.y > 0.0f)
        hitSun = shadow_vis(SP, fp + sunDir * 0.02f, sunDir, P[P_SHADOWDIST]);
    return skyTint * (skyAmb * hitSky) + sunCol * (hitSun * fmax(sunDir.y, 0.0f));
}

// One hemisphere ray. Returns how much sky it can still see (1 = clean escape); when bounce is enabled
// the light arriving from the first real surface it meets is accumulated into *bounce.
inline float gi_ray(SA, float3 o, float3 d, float maxT, int doBounce, float3* bounce)
{
    Tr s; tr_init(&s, nodes, o, d, maxT, P[P_YTOP]);
    float vis = 1.0f;
    int vx, vy, vz, axis; uint mat; float tin;
    while (tr_next(&s, nodes, mats, &vx, &vy, &vz, &mat, &tin, &axis)) {
        uint id = mat & 0x7fffu;
        uint fl = palFlags[id];
        if (fl & F_WATER) { vis *= 0.9f; continue; }
        if (fl & F_TRANSL) { vis *= 0.8f; continue; }
        if (!(fl & F_SHADOW)) continue;
        if (!doBounce) return 0.0f;
        float th; int face; float4 tex;
        if (!eval_voxel(SP, o, d, vx, vy, vz, axis, tin, id, &th, &face, &tex)) continue;
        *bounce += tex.xyz * gi_surface_light(SP, o + d * th - d * 0.02f);
        return 0.0f;
    }
    return vis;
}

__kernel void gi_bake(
    SA,
    __global uint* outLight,
    __global const int* GP)
{
    int gid = get_global_id(0);
    int idx = GP[0] + gid;
    if (idx >= GP[1]) return;

    int dimX = GP[2], dimY = GP[3], cell = GP[4];
    int i = idx % dimX, t = idx / dimX;
    int j = t % dimY, k = t / dimY;

    float3 lo = (float3)((float)(GP[5] + (i + 0.5f) * cell),
                         (float)(GP[6] + (j + 0.5f) * cell),
                         (float)(GP[7] + (k + 0.5f) * cell))
                - (float3)((float)IP[2], 0.0f, (float)IP[3]);

    // A cell whose centre sits inside geometry must borrow the light of the nearest open voxel: surface
    // vertices land exactly on a cell boundary, so without this every one of them would sample 50 % dark.
    if (gi_occupied(SP, (int)floor(lo.x), (int)floor(lo.y), (int)floor(lo.z))) {
        int lvx = (int)floor(lo.x), lvy = (int)floor(lo.y), lvz = (int)floor(lo.z);
        int found = 0;
        for (int d = 0; d < 6 && !found; d++) {
            float3 dd = gi_dir6(d);
            for (int s = 1; s <= cell && !found; s++) {
                int nx = lvx + (int)(dd.x * s), ny = lvy + (int)(dd.y * s), nz = lvz + (int)(dd.z * s);
                if (!gi_occupied(SP, nx, ny, nz)) {
                    lo = (float3)((float)nx + 0.5f, (float)ny + 0.5f, (float)nz + 0.5f);
                    found = 1;
                }
            }
        }
        if (!found) { outLight[idx] = 0u; return; }   // sealed inside rock: no light reaches it
    }

    int N = GP[8], flags = GP[9];
    float maxT = P[P_SHADOWDIST];
    float3 skyTint = (float3)(0.88f, 0.94f, 1.0f);
    float3 sunDir = vload3p(P, P_SUNDIR), sunCol = vload3p(P, P_SUNCOL);

    float skyVis = 0.0f;
    float3 bounce = (float3)(0.0f);
    int doBounce = (flags & 2) != 0;
    for (int r = 0; r < N; r++) {
        float3 d = gi_hemi(r, N);
        skyVis += gi_ray(SP, lo + d * 0.02f, d, maxT, doBounce, &bounce);
    }
    float invN = 1.0f / (float)N;
    skyVis *= invN;
    bounce *= invN;

    int vx, vy, vz, axis; uint mat; float tin;

    // direct sun on this cell (only worth testing when some sky is visible at all)
    float sunLit = 0.0f;
    if ((flags & 1) != 0 && skyVis > 0.0f && sunDir.y > 0.0f) {
        Tr sh; tr_init(&sh, nodes, lo + sunDir * 0.02f, sunDir, maxT, P[P_YTOP]);
        if (!tr_next(&sh, nodes, mats, &vx, &vy, &vz, &mat, &tin, &axis)) sunLit = 1.0f;
    }

    // ray traced torch / glowstone light
    float3 emi = (float3)(0.0f);
    if ((flags & 4) != 0) {
        int csx = ((int)floor(lo.x)) >> 4, csy = ((int)floor(lo.y)) >> 4, csz = ((int)floor(lo.z)) >> 4;
        if (csx >= 0 && csx <= 15 && csy >= 0 && csy <= 15 && csz >= 0 && csz <= 15) {
            float4 best = (float4)(0.0f); float bestS = 0.0f;
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
                int sx = csx + dx, sy = csy + dy, sz = csz + dz;
                if (sx < 0 || sx > 15 || sy < 0 || sy > 15 || sz < 0 || sz > 15) continue;
                int guard = 0;
                for (int e = emHead[(sx << 8) | (sy << 4) | sz]; e >= 0 && guard < 10; e = emNext[e], guard++) {
                    float4 ep = emPos[e];
                    float3 L = ep.xyz - lo;
                    float d2 = dot(L, L);
                    if (d2 < 0.25f || d2 > 324.0f) continue;
                    float sc = ep.w / fmax(1.0f, d2);
                    if (sc > bestS) { bestS = sc; best = ep; }
                }
            }
            if (best.w > 0.5f) {
                float3 L = best.xyz - lo;
                float dist = length(L);
                L /= dist;
                Tr sh; tr_init(&sh, nodes, lo + L * 0.02f, L, fmax(dist - 0.6f, 0.05f), P[P_YTOP]);
                float v2 = tr_next(&sh, nodes, mats, &vx, &vy, &vz, &mat, &tin, &axis) ? 0.0f : 1.0f;
                float direct = (best.w / 15.0f) * v2 / (1.0f + 0.06f * dist * dist);
                emi = (float3)(1.0f, 0.78f, 0.45f) * direct;
            }
        }
    }

    float3 gi = skyTint * (P[P_SKYAMB] * skyVis)
              + sunCol * (sunLit * fmax(sunDir.y, 0.0f))
              + bounce
              + emi;
    gi = clamp(gi, 0.0f, 1.0f);
    uint pr = (uint)(gi.x * 255.0f + 0.5f), pg = (uint)(gi.y * 255.0f + 0.5f), pb = (uint)(gi.z * 255.0f + 0.5f);
    outLight[idx] = pr | (pg << 8) | (pb << 16) | (255u << 24);
}

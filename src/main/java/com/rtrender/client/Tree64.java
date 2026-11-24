package com.rtrender.client;

import java.util.Arrays;

/**
 * Sparse 64-tree (4x4x4 children per node, 64-bit child mask, popcount child indexing), 4 levels = 256^3.
 * bit index = cx + 4*cy + 16*cz. Nodes are stored root, level2, level1, level0; leaf nodes point into {@code mats}.
 * mats entry = palette id | 0x8000 when it is a water voxel with no water above (exposed surface).
 */
public final class Tree64 {
    public long[] masks;
    public int[] ptrs;
    public short[] mats;
    public int nodeCount;

    private final VoxelWorld vol;
    private final int ox, oz;
    private final byte[] palFlags;
    private final long[][] lm = new long[4][];
    private final int[][] lp = new int[4][];
    private final int[] ln = new int[4];
    private short[] mb = new short[1 << 14];
    private int mn;

    private Tree64(VoxelWorld vol, int ox, int oz, byte[] palFlags) {
        this.vol = vol; this.ox = ox; this.oz = oz; this.palFlags = palFlags;
        for (int i = 0; i < 4; i++) { lm[i] = new long[1 << 10]; lp[i] = new int[1 << 10]; }
    }

    public static Tree64 build(VoxelWorld vol, int ox, int oz, byte[] palFlags) {
        Tree64 t = new Tree64(vol, ox, oz, palFlags);
        boolean any = t.node(3, 0, 0, 0);
        if (!any) t.push(3, 0L, 0);
        t.finish();
        return t;
    }

    private boolean sectionsEmpty(int x, int y, int z, int size) {
        for (int sx = 0; sx < size; sx += 16)
            for (int sy = 0; sy < size; sy += 16)
                for (int sz = 0; sz < size; sz += 16)
                    if (vol.sectionCount[VoxelWorld.secIndex(ox + x + sx, y + sy, oz + z + sz)] != 0) return false;
        return true;
    }

    private boolean isWater(short m) {
        int id = m & 0x7fff;
        return id != 0 && id < palFlags.length && (palFlags[id] & BlockPalette.F_WATER) != 0;
    }

    private boolean node(int level, int x0, int y0, int z0) {
        int cs = 1 << (2 * level);
        long mask = 0L;
        int start = (level == 0) ? mn : ln[level - 1];
        for (int cz = 0; cz < 4; cz++) {
            for (int cy = 0; cy < 4; cy++) {
                for (int cx = 0; cx < 4; cx++) {
                    int bit = cx + (cy << 2) + (cz << 4);
                    int x = x0 + cx * cs, y = y0 + cy * cs, z = z0 + cz * cs;
                    if (level == 0) {
                        short m = vol.cells[VoxelWorld.cellIndex(ox + x, y, oz + z)];
                        if (m != 0) {
                            if (isWater(m)) {
                                short above = y < 255 ? vol.cells[VoxelWorld.cellIndex(ox + x, y + 1, oz + z)] : 0;
                                if (!isWater(above)) m |= (short) 0x8000;
                            }
                            mask |= 1L << bit;
                            if (mn == mb.length) mb = Arrays.copyOf(mb, mb.length * 2);
                            if (mn >= mb.length) { mb = Arrays.copyOf(mb, mb.length * 2); }
                            mb[mn++] = m;
                        }
                    } else {
                        if (cs >= 16 && sectionsEmpty(x, y, z, cs)) continue;
                        if (node(level - 1, x, y, z)) mask |= 1L << bit;
                    }
                }
            }
        }
        if (mask == 0L) return false;
        push(level, mask, start);
        return true;
    }

    private void push(int level, long mask, int ptr) {
        int n = ln[level];
        if (n == lm[level].length) { lm[level] = Arrays.copyOf(lm[level], n * 2); lp[level] = Arrays.copyOf(lp[level], n * 2); }
        lm[level][n] = mask; lp[level][n] = ptr; ln[level] = n + 1;
    }

    private void finish() {
        int[] off = new int[4];
        off[3] = 0; off[2] = ln[3]; off[1] = off[2] + ln[2]; off[0] = off[1] + ln[1];
        nodeCount = off[0] + ln[0];
        masks = new long[nodeCount]; ptrs = new int[nodeCount];
        for (int L = 3; L >= 0; L--) {
            for (int i = 0; i < ln[L]; i++) {
                masks[off[L] + i] = lm[L][i];
                ptrs[off[L] + i] = (L == 0) ? lp[L][i] : off[L - 1] + lp[L][i];
            }
        }
        mats = Arrays.copyOf(mb, Math.max(mn + 4, 8));
    }
}

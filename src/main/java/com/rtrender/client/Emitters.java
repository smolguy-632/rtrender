package com.rtrender.client;

import java.util.Arrays;

/** Light-emitting voxels bucketed per 16^3 section (linked lists) so the kernel can find the strongest nearby light. */
public final class Emitters {
    public static final int MAX = 16384;
    public final int[] head = new int[4096];
    public int[] next = new int[MAX];
    public float[] pos = new float[MAX * 4];      // x, y, z (voxel centre, window-local), level
    public int n;

    public static Emitters collect(VoxelWorld vol, int ox, int oz, BlockPalette.Snapshot ps) {
        Emitters e = new Emitters();
        Arrays.fill(e.head, -1);
        for (int sx = 0; sx < 16; sx++) for (int sz = 0; sz < 16; sz++) for (int sy = 0; sy < 16; sy++) {
            if (vol.sectionCount[VoxelWorld.secIndex(ox + sx * 16, sy * 16, oz + sz * 16)] == 0) continue;
            int key = (sx << 8) | (sy << 4) | sz;
            for (int lx = 0; lx < 16; lx++) for (int ly = 0; ly < 16; ly++) for (int lz = 0; lz < 16; lz++) {
                int x = sx * 16 + lx, y = sy * 16 + ly, z = sz * 16 + lz;
                int id = vol.cells[VoxelWorld.cellIndex(ox + x, y, oz + z)] & 0x7fff;
                if (id == 0 || id >= ps.count) continue;
                int lv = ps.lightLevel[id];
                if (lv <= 0 || e.n >= MAX) continue;
                int i = e.n++;
                e.pos[i * 4] = x + 0.5f; e.pos[i * 4 + 1] = y + 0.5f; e.pos[i * 4 + 2] = z + 0.5f; e.pos[i * 4 + 3] = lv;
                e.next[i] = e.head[key];
                e.head[key] = i;
            }
        }
        return e;
    }
}

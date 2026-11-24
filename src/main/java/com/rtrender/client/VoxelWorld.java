package com.rtrender.client;

import net.minecraft.block.state.IBlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.biome.BiomeColorHelper;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.storage.ExtendedBlockStorage;

import java.util.Arrays;
import java.util.Comparator;

/**
 * 256^3 toroidal voxel mirror of the client world (full height, 16 chunks wide).
 * cells: palette id (bits 0..14, 0 = air / not drawn).  light: (skyLight << 4) | blockLight of that cell.
 * colTint: biome colours per chunk column slot: [0..255] grass, [256..511] foliage, [512..767] water.
 */
public final class VoxelWorld {
    public static final int SIZE = 256;

    public final short[] cells = new short[SIZE * SIZE * SIZE];
    public final byte[] light = new byte[SIZE * SIZE * SIZE];
    public final short[] sectionCount = new short[16 * 16 * 16];
    public final int[] colTint = new int[768];
    public final BlockPalette palette = new BlockPalette();

    private final byte[] sectionState = new byte[16 * 16 * 16];
    private final int[] slotCx = new int[256], slotCz = new int[256];
    public volatile int originX, originZ;
    public volatile boolean dirty;
    private int cursor;

    private static final int[] ORD_X = new int[256], ORD_Z = new int[256];
    static {
        Integer[] idx = new Integer[256];
        for (int i = 0; i < 256; i++) idx[i] = i;
        Arrays.sort(idx, new Comparator<Integer>() {
            public int compare(Integer a, Integer b) { return Integer.compare(d2(a), d2(b)); }
            private int d2(int i) { int x = (i & 15) - 8, z = (i >> 4) - 8; return x * x + z * z; }
        });
        for (int i = 0; i < 256; i++) { ORD_X[i] = idx[i] & 15; ORD_Z[i] = idx[i] >> 4; }
    }

    public VoxelWorld() { reset(); }

    public synchronized void reset() {
        Arrays.fill(cells, (short) 0);
        Arrays.fill(light, (byte) 0);
        Arrays.fill(sectionCount, (short) 0);
        Arrays.fill(sectionState, (byte) 0);
        Arrays.fill(slotCx, Integer.MIN_VALUE);
        Arrays.fill(slotCz, Integer.MIN_VALUE);
        Arrays.fill(colTint, 0xFFFFFF);
        palette.clear();
        cursor = 0; originX = 0; originZ = 0; dirty = true;
    }

    public static int cellIndex(int wx, int wy, int wz) { return ((wx & 255) << 16) | (wy << 8) | (wz & 255); }
    public static int secIndex(int wx, int wy, int wz) { return (((wx >> 4) & 15) << 8) | ((wy >> 4) << 4) | ((wz >> 4) & 15); }

    public void recenter(int playerChunkX, int playerChunkZ) {
        int ocx = playerChunkX - 8, ocz = playerChunkZ - 8;
        if ((ocx << 4) != originX || (ocz << 4) != originZ) dirty = true;
        originX = ocx << 4; originZ = ocz << 4;
        for (int dz = 0; dz < 16; dz++) for (int dx = 0; dx < 16; dx++) {
            int cx = ocx + dx, cz = ocz + dz;
            int slot = (cx & 15) | ((cz & 15) << 4);
            if (slotCx[slot] != cx || slotCz[slot] != cz) {
                clearColumn(cx, cz);
                slotCx[slot] = cx; slotCz[slot] = cz;
            }
        }
    }

    private void clearColumn(int cx, int cz) {
        int bz = cz << 4;
        for (int x = 0; x < 16; x++) {
            int wx = (cx << 4) + x;
            for (int y = 0; y < 256; y++) {
                int base = cellIndex(wx, y, bz);
                Arrays.fill(cells, base, base + 16, (short) 0);
                Arrays.fill(light, base, base + 16, (byte) 0);
            }
        }
        for (int sy = 0; sy < 16; sy++) {
            int si = secIndex(cx << 4, sy << 4, bz);
            sectionCount[si] = 0; sectionState[si] = 0;
        }
        dirty = true;
    }

    public boolean isReady(World world, int pcx, int pcz, int radius) {
        int ocx = originX >> 4, ocz = originZ >> 4;
        for (int dz = 0; dz < 16; dz++) for (int dx = 0; dx < 16; dx++) {
            int cx = ocx + dx, cz = ocz + dz;
            if (Math.abs(cx - pcx) > radius || Math.abs(cz - pcz) > radius) continue;
            if (world.getChunkProvider().getLoadedChunk(cx, cz) == null) continue;
            for (int sy = 0; sy < 16; sy++)
                if (sectionState[secIndex(cx << 4, sy << 4, cz << 4)] == 0) return false;
        }
        return true;
    }

    public void scan(World world, double budget, int playerCx, int playerCz, int radius) {
        double used = 0; int guard = 0;
        while (used < budget && guard < 4096) {
            int c = cursor; cursor = (cursor + 1) & 4095; guard++;
            int sy = c & 15, ci = c >> 4;
            int cx = (originX >> 4) + ORD_X[ci], cz = (originZ >> 4) + ORD_Z[ci];
            if (Math.abs(cx - playerCx) > radius || Math.abs(cz - playerCz) > radius) continue;
            Chunk ch = world.getChunkProvider().getLoadedChunk(cx, cz);
            if (ch == null) continue;
            if (sy == 0) updateColors(world, cx, cz);
            used += scanSection(world, ch, cx, cz, sy);
        }
    }

    private void updateColors(World w, int cx, int cz) {
        BlockPos p = new BlockPos((cx << 4) + 8, 64, (cz << 4) + 8);
        int slot = (cx & 15) | ((cz & 15) << 4);
        int g = colTint[slot], f = colTint[256 + slot], wt = colTint[512 + slot];
        try {
            g = BiomeColorHelper.getGrassColorAtPos(w, p);
            f = BiomeColorHelper.getFoliageColorAtPos(w, p);
            wt = BiomeColorHelper.getWaterColorAtPos(w, p);
        } catch (Throwable ignored) {}
        if (colTint[slot] != g || colTint[256 + slot] != f || colTint[512 + slot] != wt) dirty = true;
        colTint[slot] = g; colTint[256 + slot] = f; colTint[512 + slot] = wt;
    }

    private double scanSection(World world, Chunk ch, int cx, int cz, int sy) {
        ExtendedBlockStorage s = ch.getBlockStorageArray()[sy];
        int bx = cx << 4, by = sy << 4, bz = cz << 4, cnt = 0;
        boolean changed = false;
        double cost;
        if (s == null) {
            for (int x = 0; x < 16; x++) for (int y = 0; y < 16; y++) {
                int base = cellIndex(bx + x, by + y, bz);
                for (int z = 0; z < 16; z++) {
                    if (cells[base + z] != 0 || light[base + z] != (byte) 0xF0) changed = true;
                    cells[base + z] = 0; light[base + z] = (byte) 0xF0;
                }
            }
            cost = 0.03;
        } else {
            boolean empty = s.isEmpty();
            boolean sky = true;
            try { s.getSkyLight(0, 0, 0); } catch (RuntimeException e) { sky = false; }
            for (int x = 0; x < 16; x++) for (int y = 0; y < 16; y++) for (int z = 0; z < 16; z++) {
                int i = cellIndex(bx + x, by + y, bz + z);
                int sl = sky ? s.getSkyLight(x, y, z) : 0;
                byte lv = (byte) ((sl << 4) | s.getBlockLight(x, y, z));
                if (light[i] != lv) { light[i] = lv; changed = true; }
                short nid = 0;
                if (!empty) {
                    IBlockState st = s.get(x, y, z);
                    nid = (short) palette.idFor(st, world, bx + x, by + y, bz + z);
                }
                if (cells[i] != nid) { cells[i] = nid; changed = true; }
                if (nid != 0) cnt++;
            }
            cost = empty ? 0.3 : 1.0;
        }
        int si = secIndex(bx, by, bz);
        if (sectionCount[si] != (short) cnt) changed = true;
        sectionCount[si] = (short) cnt;
        if (sectionState[si] == 0) changed = true;
        sectionState[si] = 1;
        if (changed) dirty = true;
        return cost;
    }

    /** Immediate block update from the world listener (main thread). */
    public void setBlock(World w, int wx, int wy, int wz, IBlockState st) {
        if (wy < 0 || wy > 255) return;
        int cx = wx >> 4, cz = wz >> 4;
        if (cx < (originX >> 4) || cx >= (originX >> 4) + 16 || cz < (originZ >> 4) || cz >= (originZ >> 4) + 16) return;
        short nid = (short) palette.idFor(st, w, wx, wy, wz);
        int idx = cellIndex(wx, wy, wz);
        short old = cells[idx];
        if (old == nid) return;
        int si = secIndex(wx, wy, wz);
        if (old == 0) sectionCount[si]++; else if (nid == 0) sectionCount[si]--;
        cells[idx] = nid;
        dirty = true;
    }
}

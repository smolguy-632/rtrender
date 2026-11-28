package com.rtrender.client;

import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL12;
import org.lwjgl.opengl.GL13;
import org.lwjgl.opengl.GL20;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The GI volume: a DIM^3 grid of baked light covering the render distance, handed to Celeritas as a GL 3D
 * texture that its vertex shader samples at the world position of every vertex.
 *
 * <p>Because lighting lives in a texture instead of in the mesh, changing it never triggers a remesh - chunks
 * only rebuild when blocks actually change. The volume is refreshed in small batches so the GPU cost per game
 * tick stays inside {@code RTConfig.giFrameBudgetUs}.</p>
 *
 * <p>Byte layout is x fastest, then y, then z, which is exactly the ordering glTexImage3D expects.</p>
 */
public final class GIVolume {
    /** Cells per axis. 64^3 * 4 bytes = 1 MB. */
    public static final int DIM = 64;
    public static final int CELLS = DIM * DIM * DIM;
    private static final int BYTES = CELLS * 4;
    /** Texture unit the GI sampler is bound to (unit 0 keeps the block atlas). */
    public static final int UNIT = 15;

    /** Blocks per cell - chosen so the volume always spans the render distance. */
    public int cell = 4;
    /** World block coordinate of cell (0,0,0)'s corner. */
    public int originX, originY, originZ;
    /** A full pass over the volume has completed at least once, so the shader may switch to GI. */
    public boolean bakedOnce;

    private int tex;
    private ByteBuffer cpu;
    private byte[] scratch;
    private int cursor;
    private boolean placed;
    private int boundProgram = -1;
    private int locTex = -1, locOrigin = -1, locStrength = -1;

    // ------------------------------------------------------------------ geometry

    public int span() { return DIM * cell; }

    /** Recompute the cell size and re-centre the volume on the player. Returns true if the texture exists. */
    public boolean ensure(int renderDistanceBlocks, double px, double py, double pz, float skyAmb) {
        int want = Math.max(2, (2 * renderDistanceBlocks + DIM - 1) / DIM);
        if (want != cell) {           // render distance changed -> start over at the new resolution
            cell = want;
            cpu = null;
            bakedOnce = false;
            cursor = 0;
            placed = false;
        }
        if (cpu == null) {
            cpu = ByteBuffer.allocateDirect(BYTES).order(ByteOrder.nativeOrder());
            scratch = new byte[BYTES];
            fillDefault(skyAmb);
        }

        int snap = cell * 4;
        int span = span();
        int nx = Math.floorDiv((int) Math.floor(px - span / 2.0), snap) * snap;
        int ny = Math.floorDiv((int) Math.floor(py - span / 2.0), snap) * snap;
        int nz = Math.floorDiv((int) Math.floor(pz - span / 2.0), snap) * snap;
        // a volume tall enough to cover the world is pinned to y = 0 instead of following the player vertically
        if (span >= 256) ny = 0;

        if (nx != originX || ny != originY || nz != originZ) {
            if (placed) shift((originX - nx) / cell, (originY - ny) / cell, (originZ - nz) / cell, skyAmb);
            originX = nx; originY = ny; originZ = nz;
            placed = true;
        }

        if (tex == 0) createTexture();
        return tex != 0;
    }

    /** Fill every cell with open-sky light (the default for cells the volume has not reached yet). */
    private void fillDefault(float skyAmb) {
        byte[] p = new byte[4];
        p[0] = (byte) clamp255(0.88f * skyAmb);
        p[1] = (byte) clamp255(0.94f * skyAmb);
        p[2] = (byte) clamp255(1.00f * skyAmb);
        p[3] = (byte) 255;
        cpu.clear();
        for (int i = 0; i < CELLS; i++) cpu.put(p);
        cpu.clear();
    }

    private static int clamp255(float v) { return (int) Math.max(0, Math.min(255, Math.round(v * 255f))); }

    /**
     * Slide the baked contents to follow a new origin. Cells that move in from outside the old volume get the
     * open-sky default. dx/dy/dz are cell deltas: new[i] = old[i - d].
     */
    private void shift(int dx, int dy, int dz, float skyAmb) {
        if (dx == 0 && dy == 0 && dz == 0) return;
        cpu.clear();
        scratch = scratch == null ? new byte[BYTES] : scratch;
        cpu.get(scratch);
        byte d0 = (byte) clamp255(0.88f * skyAmb), d1 = (byte) clamp255(0.94f * skyAmb), d2 = (byte) clamp255(1.00f * skyAmb);
        cpu.clear();
        for (int k = 0; k < DIM; k++) {
            int sk = k - dz;
            for (int j = 0; j < DIM; j++) {
                int sj = j - dy;
                boolean rowOk = sk >= 0 && sk < DIM && sj >= 0 && sj < DIM;
                for (int i = 0; i < DIM; i++) {
                    int si = i - dx;
                    int dst = ((k * DIM + j) * DIM + i) << 2;
                    if (rowOk && si >= 0 && si < DIM) {
                        int src = ((sk * DIM + sj) * DIM + si) << 2;
                        cpu.put(dst, scratch[src]);
                        cpu.put(dst + 1, scratch[src + 1]);
                        cpu.put(dst + 2, scratch[src + 2]);
                        cpu.put(dst + 3, scratch[src + 3]);
                    } else {
                        cpu.put(dst, d0); cpu.put(dst + 1, d1); cpu.put(dst + 2, d2); cpu.put(dst + 3, (byte) 255);
                    }
                }
            }
        }
        cpu.clear();
    }

    // ------------------------------------------------------------------ GL

    private void createTexture() {
        tex = GL11.glGenTextures();
        if (tex == 0) return;
        int prev = GL11.glGetInteger(GL13.GL_ACTIVE_TEXTURE);
        GL13.glActiveTexture(GL13.GL_TEXTURE0 + UNIT);
        GL11.glBindTexture(GL11.GL_TEXTURE_3D, tex);
        GL11.glPixelStorei(GL11.GL_UNPACK_ALIGNMENT, 1);
        GL12.glTexImage3D(GL11.GL_TEXTURE_3D, 0, GL11.GL_RGBA8, DIM, DIM, DIM, 0,
                GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, (ByteBuffer) null);
        GL11.glTexParameteri(GL11.GL_TEXTURE_3D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL11.GL_TEXTURE_3D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL11.GL_TEXTURE_3D, GL11.GL_TEXTURE_WRAP_S, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexParameteri(GL11.GL_TEXTURE_3D, GL11.GL_TEXTURE_WRAP_T, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexParameteri(GL11.GL_TEXTURE_3D, GL11.GL_TEXTURE_WRAP_R, GL12.GL_CLAMP_TO_EDGE);
        GL11.glBindTexture(GL11.GL_TEXTURE_3D, 0);
        GL13.glActiveTexture(prev);
        // push the freshly filled buffer so the very first frame is not undefined
        cpu.clear();
        GL12.glTexSubImage3D(GL11.GL_TEXTURE_3D, 0, 0, 0, 0, DIM, DIM, DIM,
                GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, cpu);
        cpu.clear();
    }

    /** Upload the baked z range [zFrom, zTo). */
    public void upload(int zFrom, int zTo) {
        if (tex == 0 || zTo <= zFrom) return;
        int prev = GL11.glGetInteger(GL13.GL_ACTIVE_TEXTURE);
        GL13.glActiveTexture(GL13.GL_TEXTURE0 + UNIT);
        GL11.glBindTexture(GL11.GL_TEXTURE_3D, tex);
        cpu.clear();
        cpu.position(zFrom * DIM * DIM * 4);
        GL12.glTexSubImage3D(GL11.GL_TEXTURE_3D, 0, 0, 0, zFrom, DIM, DIM, zTo - zFrom,
                GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, cpu);
        cpu.clear();
        GL11.glBindTexture(GL11.GL_TEXTURE_3D, 0);
        GL13.glActiveTexture(prev);
    }

    /** Point the currently bound chunk program at this volume. Safe to call every pass. */
    public void bind(int program) {
        if (program <= 0) return;
        if (program != boundProgram) {
            boundProgram = program;
            locTex = GL20.glGetUniformLocation(program, "u_GIVol");
            locOrigin = GL20.glGetUniformLocation(program, "u_GIOrigin");
            locStrength = GL20.glGetUniformLocation(program, "u_GIStrength");
        }
        if (locTex < 0) return;
        int prev = GL11.glGetInteger(GL13.GL_ACTIVE_TEXTURE);
        GL13.glActiveTexture(GL13.GL_TEXTURE0 + UNIT);
        GL11.glBindTexture(GL11.GL_TEXTURE_3D, tex);
        GL13.glActiveTexture(prev);
        GL20.glUniform1i(locTex, UNIT);
        if (locOrigin >= 0)
            GL20.glUniform4f(locOrigin, originX, originY, originZ, (float) span());
        if (locStrength >= 0)
            GL20.glUniform1f(locStrength, bakedOnce ? 1f : 0f);
    }

    /** Drop everything when leaving a world (the GL context may go away with it). */
    public void reset() {
        if (tex != 0) {
            GL11.glDeleteTextures(tex);
            tex = 0;
        }
        cpu = null;
        scratch = null;
        cursor = 0;
        placed = false;
        bakedOnce = false;
        boundProgram = -1;
        originX = originY = originZ = 0;
    }

    // ------------------------------------------------------------------ bake scheduling

    public int cursor() { return cursor; }

    public void advanceCursor(int cells) {
        cursor += cells;
        if (cursor >= CELLS) cursor = 0;
    }

    public void invalidate() { bakedOnce = false; cursor = 0; }

    public ByteBuffer buffer() { return cpu; }
}

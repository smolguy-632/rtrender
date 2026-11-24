package com.rtrender.client;

import net.minecraft.block.Block;
import net.minecraft.block.BlockDoublePlant;
import net.minecraft.block.BlockGrass;
import net.minecraft.block.BlockLeaves;
import net.minecraft.block.BlockReed;
import net.minecraft.block.BlockTallGrass;
import net.minecraft.block.material.Material;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.block.model.BakedQuad;
import net.minecraft.client.renderer.block.model.IBakedModel;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.client.renderer.texture.TextureMap;
import net.minecraft.init.Blocks;
import net.minecraft.util.BlockRenderLayer;
import net.minecraft.util.EnumBlockRenderType;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.ColorizerFoliage;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Block state -> small integer id (1..32767, 0 = air / not drawn by the ray tracer) plus everything the kernel needs:
 * per-face atlas UV rectangle, tint class, flags, partial-block bounding box and light level.
 * Face order: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z.   Main (client) thread only.
 */
public final class BlockPalette {
    public static final int F_OPAQUE = 1, F_CUTOUT = 2, F_TRANSL = 4, F_WATER = 8, F_FULL = 16, F_EMIT = 32, F_SHADOW = 64;
    private static final EnumFacing[] FACES = {EnumFacing.EAST, EnumFacing.WEST, EnumFacing.UP, EnumFacing.DOWN, EnumFacing.SOUTH, EnumFacing.NORTH};

    private final IdentityHashMap<IBlockState, Integer> ids = new IdentityHashMap<IBlockState, Integer>();
    private int count = 1;
    private int[] flags = new int[1024];
    private float[] uv = new float[1024 * 6 * 4];
    private int[] tint = new int[1024 * 6];
    private float[] box = new float[1024 * 8];
    private byte[] lightLevel = new byte[1024];

    /** Immutable copy for the worker thread / GPU upload. */
    public static final class Snapshot {
        public int count;
        public int[] flags, tint;
        public float[] uv, box;
        public byte[] lightLevel;
    }

    public synchronized void clear() {
        ids.clear();
        count = 1;
        Arrays.fill(flags, 0);
    }

    public synchronized Snapshot snapshot() {
        Snapshot s = new Snapshot();
        s.count = count;
        s.flags = Arrays.copyOf(flags, count);
        s.tint = Arrays.copyOf(tint, count * 6);
        s.uv = Arrays.copyOf(uv, count * 24);
        s.box = Arrays.copyOf(box, count * 8);
        s.lightLevel = Arrays.copyOf(lightLevel, count);
        return s;
    }

    public int idFor(IBlockState st, World w, int x, int y, int z) {
        Integer id = ids.get(st);
        if (id == null) {
            id = create(st, w, new BlockPos(x, y, z));
            ids.put(st, id);
        }
        return id;
    }

    private int allocate() {
        if (count >= 32767) return 0;
        if (count == flags.length) {
            int n = flags.length * 2;
            flags = Arrays.copyOf(flags, n);
            uv = Arrays.copyOf(uv, n * 24);
            tint = Arrays.copyOf(tint, n * 6);
            box = Arrays.copyOf(box, n * 8);
            lightLevel = Arrays.copyOf(lightLevel, n);
        }
        int id = count++;
        flags[id] = 0;
        return id;
    }

    private synchronized int create(IBlockState st, World w, BlockPos pos) {
        EnumBlockRenderType rt = st.getRenderType();
        if (rt == EnumBlockRenderType.INVISIBLE || rt == EnumBlockRenderType.ENTITYBLOCK_ANIMATED) return 0;
        Minecraft mc = Minecraft.getMinecraft();
        TextureMap atlas = mc.getTextureMapBlocks();
        Block b = st.getBlock();
        Material mat = st.getMaterial();
        int light = 0;
        try { light = st.getLightValue(); } catch (Throwable ignored) {}
        int id = allocate();
        if (id == 0) return 0;
        lightLevel[id] = (byte) Math.min(15, light);

        if (rt == EnumBlockRenderType.LIQUID) {
            boolean lava = mat == Material.LAVA;
            TextureAtlasSprite sp = atlas.getAtlasSprite(lava ? "minecraft:blocks/lava_still" : "minecraft:blocks/water_still");
            for (int f = 0; f < 6; f++) {
                setUV(id, f, sp);
                tint[id * 6 + f] = lava ? 0 : (3 << 24);
            }
            flags[id] = lava ? (F_OPAQUE | F_FULL | F_EMIT | F_SHADOW) : (F_WATER | F_FULL);
            if (lava) lightLevel[id] = 15;
            return id;
        }

        IBakedModel model = mc.getBlockRendererDispatcher().getModelForState(st);
        for (int f = 0; f < 6; f++) {
            TextureAtlasSprite sp = null;
            int ti = -1;
            try {
                List<BakedQuad> q = model.getQuads(st, FACES[f], 0L);
                if (q != null && !q.isEmpty()) { sp = q.get(0).getSprite(); ti = q.get(0).getTintIndex(); }
                else {
                    List<BakedQuad> g = model.getQuads(st, null, 0L);
                    if (g != null) for (BakedQuad bq : g) if (bq.getFace() == FACES[f]) { sp = bq.getSprite(); ti = bq.getTintIndex(); break; }
                    if (sp == null && g != null && !g.isEmpty()) { sp = g.get(0).getSprite(); ti = g.get(0).getTintIndex(); }
                }
            } catch (Throwable ignored) {}
            if (sp == null) sp = model.getParticleTexture();
            if (sp == null) sp = atlas.getMissingSprite();
            setUV(id, f, sp);
            tint[id * 6 + f] = tintFor(st, w, pos, b, mat, ti);
        }

        boolean opaque = st.isOpaqueCube();
        boolean full = opaque || st.isFullCube();
        int fl = 0;
        BlockRenderLayer layer = BlockRenderLayer.SOLID;
        try { layer = b.getRenderLayer(); } catch (Throwable ignored) {}
        if (opaque) fl |= F_OPAQUE | F_FULL | F_SHADOW;
        else {
            if (full) fl |= F_FULL;
            if (layer == BlockRenderLayer.TRANSLUCENT) fl |= F_TRANSL;
            else {
                fl |= F_SHADOW;
                if (layer == BlockRenderLayer.CUTOUT || layer == BlockRenderLayer.CUTOUT_MIPPED) fl |= F_CUTOUT;
            }
        }
        if (!full) {
            AxisAlignedBB bb = null;
            try { bb = st.getBoundingBox(w, pos); } catch (Throwable ignored) {}
            if (bb == null) bb = Block.FULL_BLOCK_AABB;
            box[id * 8] = (float) bb.minX; box[id * 8 + 1] = (float) bb.minY; box[id * 8 + 2] = (float) bb.minZ;
            box[id * 8 + 4] = (float) bb.maxX; box[id * 8 + 5] = (float) bb.maxY; box[id * 8 + 6] = (float) bb.maxZ;
        }
        if (light > 0) fl |= F_EMIT;
        flags[id] = fl;
        return id;
    }

    private void setUV(int id, int f, TextureAtlasSprite sp) {
        int o = (id * 6 + f) * 4;
        float inset = 1e-4f;
        uv[o] = sp.getMinU() + inset; uv[o + 1] = sp.getMinV() + inset;
        uv[o + 2] = sp.getMaxU() - inset; uv[o + 3] = sp.getMaxV() - inset;
    }

    /** tint word: bits 24..27 = class (0 none, 1 grass, 2 foliage, 3 water, 4 fixed rgb in low 24 bits). */
    private int tintFor(IBlockState st, World w, BlockPos pos, Block b, Material mat, int tintIndex) {
        if (tintIndex < 0) return 0;
        if (b instanceof BlockGrass || b instanceof BlockTallGrass || b instanceof BlockDoublePlant || b instanceof BlockReed) return 1 << 24;
        int c = 0xFFFFFF;
        try { c = Minecraft.getMinecraft().getBlockColors().colorMultiplier(st, w, pos, tintIndex); } catch (Throwable ignored) {}
        if (mat == Material.LEAVES || b instanceof BlockLeaves) {
            if (c == ColorizerFoliage.getFoliageColorPine() || c == ColorizerFoliage.getFoliageColorBirch()) return (4 << 24) | (c & 0xFFFFFF);
            return 2 << 24;
        }
        if (mat == Material.WATER) return 3 << 24;
        return (4 << 24) | (c & 0xFFFFFF);
    }
}

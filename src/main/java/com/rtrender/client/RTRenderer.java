package com.rtrender.client;

import com.rtrender.RTConfig;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.SoundCategory;
import net.minecraft.util.SoundEvent;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.IWorldEventListener;
import net.minecraft.world.World;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.opengl.GL11;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Orchestrates the ray traced terrain:
 *  main thread : mirrors blocks into the voxel volume (time sliced, nearest first) + listens to block updates
 *  worker      : rebuilds the sparse 64-tree + emitter lists when something changed
 *  render hook : (called from the RenderGlobal.renderBlockLayer mixin) uploads the newest snapshot, traces one
 *                ray per pixel on the GPU, and draws colour + depth back with glDrawPixels so entities, particles,
 *                the hand, clouds and tile entities composite correctly against the ray traced world.
 */
public final class RTRenderer {
    public static final RTRenderer INSTANCE = new RTRenderer();

    /** Immutable scene snapshot handed from the worker to the render thread. */
    public static final class Published {
        public Tree64 tree;
        public BlockPalette.Snapshot palette;
        public Emitters emitters;
        public byte[] light;          // live array (benign tearing)
        public int[] colTint;
        public int originX, originZ;
    }

    private static final Logger LOG = LogManager.getLogger("RTRender");

    private final VoxelWorld vol = new VoxelWorld();
    private final AtomicBoolean busy = new AtomicBoolean();
    private final AtomicReference<Published> pending = new AtomicReference<Published>();
    private final ExecutorService pool = Executors.newSingleThreadExecutor(new ThreadFactory() {
        public Thread newThread(Runnable r) { Thread t = new Thread(r, "RTRender-Worker"); t.setDaemon(true); return t; }
    });
    private final IWorldEventListener listener = new Listener();

    private RTEngine engine;
    private volatile boolean failed;
    private volatile boolean atlasDirty = true;
    private Published current;
    private World lastWorld;
    private int sinceLast;
    private boolean everPublished;
    private float scale = -1f;
    private long frameCounter;
    private double[] cachedVp, cachedInv;
    // cached sky/fog/celestial (refreshed when world time advances by >= 1 tick)
    private double cachedTime = -1;
    private float cachedSunY, cachedSunX, cachedRain, cachedDayF, cachedNightF, cachedSkyAmb;
    private Vec3d cachedZen, cachedFog;
    private float[] cachedSdir, cachedScol;

    private ByteBuffer colorBuf, depthBuf;
    private int bufPixels;
    private final FloatBuffer matBuf = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asFloatBuffer();

    private RTRenderer() {}

    // ------------------------------------------------------------------------------ events (main thread)
    public void onWorldLoad(World w) { w.addEventListener(listener); }

    public void onWorldUnload(World w) {
        w.removeEventListener(listener);
        current = null; pending.set(null); everPublished = false;
    }

    public void onAtlasReload() {
        atlasDirty = true; current = null; pending.set(null); everPublished = false;
        vol.reset();
    }

    public void onClientTick(Minecraft mc) {
        if (failed || mc.world == null || mc.player == null) return;
        if (mc.world != lastWorld) {
            lastWorld = mc.world; vol.reset(); current = null; pending.set(null); everPublished = false; sinceLast = 0;
        }
        int pcx = MathHelper.floor(mc.player.posX) >> 4, pcz = MathHelper.floor(mc.player.posZ) >> 4;
        vol.recenter(pcx, pcz);
        // scan faster until the first complete snapshot exists
        vol.scan(mc.world, RTConfig.scanBudgetPerTick * (everPublished ? 1.0 : 6.0), pcx, pcz, RTConfig.radiusChunks);

        sinceLast++;
        if (busy.get() || sinceLast < RTConfig.publishIntervalTicks) return;
        boolean want = everPublished ? vol.dirty : vol.isReady(mc.world, pcx, pcz, RTConfig.radiusChunks);
        if (!want || !busy.compareAndSet(false, true)) return;
        sinceLast = 0; vol.dirty = false;
        final BlockPalette.Snapshot ps = vol.palette.snapshot();
        final int ox = vol.originX, oz = vol.originZ;
        final int[] col = vol.colTint.clone();
        pool.execute(new Runnable() {
            public void run() {
                try {
                    long t0 = System.nanoTime();
                    Published p = new Published();
                    p.tree = Tree64.build(vol, ox, oz, ps.flags, RTConfig.accountForLight);
                    p.emitters = Emitters.collect(vol, ox, oz, ps);
                    p.palette = ps; p.light = vol.light; p.colTint = col; p.originX = ox; p.originZ = oz;
                    pending.set(p);
                    everPublished = true;
                    if (RTConfig.debugLog)
                        LOG.info("scene publish: {} nodes, {} emitters, {} palette entries, {} ms",
                                p.tree.nodeCount, p.emitters.n, ps.count, (System.nanoTime() - t0) / 1e6);
                } catch (Throwable t) {
                    LOG.error("scene build failed", t);
                } finally {
                    busy.set(false);
                }
            }
        });
    }

    // ------------------------------------------------------------------------------ render hook
    /** Called by the mixin for every terrain layer. True = ray tracer owns terrain this frame (vanilla/Celeritas is skipped). */
    public boolean prepare() {
        if (!RTConfig.enabled || failed) return false;
        Minecraft mc = Minecraft.getMinecraft();
        if (mc.world == null || mc.player == null) return false;
        try {
            if (engine == null) {
                engine = RTEngine.create(RTConfig.openclDevice, RTConfig.skipD3D12Devices);
                LOG.info("RT Renderer using OpenCL device: {}", engine.deviceName);
            }
            if (atlasDirty) uploadAtlas(mc);
            Published p = pending.getAndSet(null);
            if (p != null) { engine.uploadScene(p); current = p; }
            return current != null && engine.hasAtlas() && engine.hasScene();
        } catch (Throwable t) {
            LOG.error("RT Renderer disabled (falling back to the normal terrain renderer)", t);
            failed = true;
            return false;
        }
    }

    private void uploadAtlas(Minecraft mc) {
        GlStateManager.bindTexture(mc.getTextureMapBlocks().getGlTextureId());
        int w = GL11.glGetTexLevelParameteri(GL11.GL_TEXTURE_2D, 0, GL11.GL_TEXTURE_WIDTH);
        int h = GL11.glGetTexLevelParameteri(GL11.GL_TEXTURE_2D, 0, GL11.GL_TEXTURE_HEIGHT);
        ByteBuffer bb = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder());
        GL11.glGetTexImage(GL11.GL_TEXTURE_2D, 0, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, bb);
        engine.uploadAtlas(bb, w, h);
        atlasSize = new int[]{w, h};
        atlasDirty = false;
    }

    private double[] readMatrix(int which) {
        matBuf.clear();
        GL11.glGetFloat(which, matBuf);
        double[] m = new double[16];
        for (int i = 0; i < 16; i++) m[i] = matBuf.get(i);
        return m;
    }

    private static float smooth(float a, float b, float x) {
        float t = Math.max(0f, Math.min(1f, (x - a) / (b - a)));
        return t * t * (3f - 2f * t);
    }

    public void renderTerrain(float pt) {
        Minecraft mc = Minecraft.getMinecraft();
        World world = mc.world;
        Entity view = mc.getRenderViewEntity();
        if (view == null || current == null) return;
        try {
            double[] proj = readMatrix(GL11.GL_PROJECTION_MATRIX), mv = readMatrix(GL11.GL_MODELVIEW_MATRIX);
            // cache the inverse when matrices are unchanged (avoids a 4x4 invert per frame)
            double[] vp = Mat4.mul(proj, mv);
            double[] inv = cachedInv;
            boolean same = cachedVp != null && cachedVp.length == vp.length;
            if (same) for (int i = 0; i < vp.length; i++) if (cachedVp[i] != vp[i]) { same = false; break; }
            if (!same) { inv = Mat4.invert(vp); if (inv == null) return; cachedVp = vp; cachedInv = inv; }

            // Which point is the origin of the space inverse(proj*modelview) returns? Vanilla bakes translate(0,-eyeHeight,0)
            // into the modelview matrix (origin = feet), but do not assume it: inv(MV) * (0,0,0,1) is the camera position in that
            // space, so in first person it is ~(0, eyeHeight, 0) when the origin is the feet and ~(0,0,0) when it is the eye.
            double[] imv = Mat4.invert(mv);
            double camRelY = imv != null ? imv[13] : 0.0;
            boolean originIsFeet = mc.gameSettings.thirdPersonView != 0 || camRelY > 0.5 * view.getEyeHeight();
            if (RTConfig.debugLog && !conventionLogged) {
                conventionLogged = true;
                LOG.info("modelview camera offset y = {} (eyeHeight {}) -> origin is {}", camRelY, view.getEyeHeight(),
                        originIsFeet ? "FEET" : "EYE");
            }
            double ex = view.lastTickPosX + (view.posX - view.lastTickPosX) * pt;
            double ey = view.lastTickPosY + (view.posY - view.lastTickPosY) * pt + (originIsFeet ? 0.0 : view.getEyeHeight());
            double ez = view.lastTickPosZ + (view.posZ - view.lastTickPosZ) * pt;

            // ---- sun / moon / sky (cached: sky/fog/celestial change slowly)
            double nowT = world.getTotalWorldTime() + pt;
            if (nowT != cachedTime) {
                cachedTime = nowT;
                float ang = world.getCelestialAngle(pt) * (float) (Math.PI * 2.0);
                cachedSunY = (float) Math.cos(ang); cachedSunX = -(float) Math.sin(ang);
                cachedRain = world.getRainStrength(pt);
                cachedDayF = smooth(-0.05f, 0.25f, cachedSunY);
                cachedNightF = smooth(0.0f, 0.25f, -cachedSunY);
                if (cachedSunY > -0.04f) {
                    cachedSdir = new float[]{cachedSunX, cachedSunY, 0f};
                    float warm = smooth(0.0f, 0.4f, cachedSunY);
                    float k = (float) RTConfig.sunStrength * cachedDayF * (1f - 0.85f * cachedRain);
                    cachedScol = new float[]{k * 1.0f, k * (0.55f + 0.40f * warm), k * (0.30f + 0.55f * warm)};
                } else {
                    cachedSdir = new float[]{-cachedSunX, -cachedSunY, 0f};
                    float k = 0.55f * cachedNightF * (1f - 0.8f * cachedRain);
                    cachedScol = new float[]{0.30f * k, 0.38f * k, 0.62f * k};
                }
                cachedSkyAmb = (0.16f + (0.70f - 0.16f) * cachedDayF) * (1f - 0.3f * cachedRain);
                cachedZen = world.getSkyColor(view, pt); cachedFog = world.getFogColor(pt);
            }
            float sunY = cachedSunY, sunX = cachedSunX, rain = cachedRain, dayF = cachedDayF, nightF = cachedNightF;
            float[] sdir = cachedSdir, scol = cachedScol;
            float skyAmb = cachedSkyAmb;
            Vec3d zen = cachedZen, fog = cachedFog;

            // ---- resolution
            int dw = mc.displayWidth, dh = mc.displayHeight;
            if (scale < 0f) scale = (float) RTConfig.renderScale;
            if (!RTConfig.dynamicResolution) scale = (float) RTConfig.renderScale;
            int w = Math.max(64, (int) (dw * scale)), h = Math.max(48, (int) (dh * scale));
            ensureBuffers((long) w * h);

            // ---- parameter block
            Published p = current;
            float[] P = new float[64];
            for (int i = 0; i < 16; i++) { P[i] = (float) inv[i]; P[16 + i] = (float) vp[i]; }
            P[32] = (float) (ex - p.originX); P[33] = (float) ey; P[34] = (float) (ez - p.originZ);
            P[35] = sdir[0]; P[36] = sdir[1]; P[37] = sdir[2];
            P[38] = scol[0]; P[39] = scol[1]; P[40] = scol[2];
            P[41] = (float) zen.x; P[42] = (float) zen.y; P[43] = (float) zen.z;
            P[44] = (float) fog.x; P[45] = (float) fog.y; P[46] = (float) fog.z;
            float far = (float) Math.min(mc.gameSettings.renderDistanceChunks * 16.0, RTConfig.maxRayDistance);
            P[47] = far * 0.55f; P[48] = far;
            P[49] = (float) ((world.getTotalWorldTime() + pt) / 20.0);
            P[50] = (float) RTConfig.maxRayDistance; P[51] = (float) RTConfig.shadowDistance;
            P[52] = (float) RTConfig.waveStrength; P[53] = skyAmb; P[54] = (float) RTConfig.blockShadowDarkness;
            P[55] = RTConfig.sunShadows ? 1f : 0f; P[56] = RTConfig.waterReflections ? 1f : 0f;
            P[57] = RTConfig.accountForLight ? 1f : 0f;
            int[] IP = {w, h, p.originX, p.originZ, 0, 0, RTConfig.blockLightShadows ? 1 : 0};
            IP[4] = atlasSize[0]; IP[5] = atlasSize[1];

            double ms = engine.render(P, IP, w, h, colorBuf, depthBuf);
            blit(w, h, dw, dh);

            if (RTConfig.dynamicResolution) {
                float f = (float) Math.pow(RTConfig.targetFrameMs / Math.max(ms, 0.5), 0.3);
                scale = Math.max(0.2f, Math.min(1.0f, scale * Math.max(0.92f, Math.min(1.08f, f))));
            }
            if (RTConfig.debugLog && (++frameCounter % 120) == 1)
                LOG.info("frame {}x{} (scale {}) took {} ms", w, h, scale, ms);
        } catch (Throwable t) {
            LOG.error("RT Renderer frame failed, disabling", t);
            failed = true;
        }
    }

    private int[] atlasSize = new int[]{1, 1};
    private boolean conventionLogged;

    private void ensureBuffers(long pixels) {
        if (pixels > bufPixels) {
            colorBuf = ByteBuffer.allocateDirect((int) (pixels * 4)).order(ByteOrder.nativeOrder());
            depthBuf = ByteBuffer.allocateDirect((int) (pixels * 4)).order(ByteOrder.nativeOrder());
            bufPixels = (int) pixels;
        }
    }

    // ------------------------------------------------------------------------------ GL composite
    private void blit(int w, int h, int dw, int dh) {
        boolean tex = GL11.glIsEnabled(GL11.GL_TEXTURE_2D), light = GL11.glIsEnabled(GL11.GL_LIGHTING),
                fog = GL11.glIsEnabled(GL11.GL_FOG), alpha = GL11.glIsEnabled(GL11.GL_ALPHA_TEST),
                cull = GL11.glIsEnabled(GL11.GL_CULL_FACE), blend = GL11.glIsEnabled(GL11.GL_BLEND),
                depthT = GL11.glIsEnabled(GL11.GL_DEPTH_TEST);

        GlStateManager.matrixMode(GL11.GL_PROJECTION);
        GlStateManager.pushMatrix();
        GlStateManager.loadIdentity();
        GlStateManager.ortho(0.0, dw, 0.0, dh, -1.0, 1.0);
        GlStateManager.matrixMode(GL11.GL_MODELVIEW);
        GlStateManager.pushMatrix();
        GlStateManager.loadIdentity();

        GlStateManager.disableTexture2D();
        GlStateManager.disableLighting();
        GlStateManager.disableFog();
        GlStateManager.disableAlpha();
        GlStateManager.disableCull();
        GlStateManager.disableBlend();
        GL11.glPixelZoom(dw / (float) w, dh / (float) h);
        GL11.glRasterPos2i(0, 0);

        // depth first (colour writes off): entities / particles / hand then depth-test against the ray traced world
        GlStateManager.colorMask(false, false, false, false);
        GlStateManager.enableDepth();
        GlStateManager.depthFunc(GL11.GL_ALWAYS);
        GlStateManager.depthMask(true);
        depthBuf.position(0);
        GL11.glDrawPixels(w, h, GL11.GL_DEPTH_COMPONENT, GL11.GL_FLOAT, depthBuf);

        // colour (premultiplied alpha: sky pixels have alpha 0 and keep the vanilla sky)
        GlStateManager.colorMask(true, true, true, true);
        GlStateManager.depthMask(false);
        GlStateManager.disableDepth();
        GlStateManager.enableBlend();
        GlStateManager.blendFunc(GlStateManager.SourceFactor.ONE, GlStateManager.DestFactor.ONE_MINUS_SRC_ALPHA);
        colorBuf.position(0);
        GL11.glRasterPos2i(0, 0);
        GL11.glDrawPixels(w, h, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, colorBuf);
        GL11.glPixelZoom(1f, 1f);

        // restore
        GlStateManager.blendFunc(GlStateManager.SourceFactor.SRC_ALPHA, GlStateManager.DestFactor.ONE_MINUS_SRC_ALPHA);
        GlStateManager.depthFunc(GL11.GL_LEQUAL);
        GlStateManager.depthMask(true);
        if (depthT) GlStateManager.enableDepth(); else GlStateManager.disableDepth();
        if (blend) GlStateManager.enableBlend(); else GlStateManager.disableBlend();
        if (cull) GlStateManager.enableCull(); else GlStateManager.disableCull();
        if (alpha) GlStateManager.enableAlpha(); else GlStateManager.disableAlpha();
        if (fog) GlStateManager.enableFog(); else GlStateManager.disableFog();
        if (light) GlStateManager.enableLighting(); else GlStateManager.disableLighting();
        if (tex) GlStateManager.enableTexture2D(); else GlStateManager.disableTexture2D();
        GlStateManager.matrixMode(GL11.GL_PROJECTION);
        GlStateManager.popMatrix();
        GlStateManager.matrixMode(GL11.GL_MODELVIEW);
        GlStateManager.popMatrix();
    }

    // ------------------------------------------------------------------------------ block update listener
    private final class Listener implements IWorldEventListener {
        public void notifyBlockUpdate(World w, BlockPos pos, IBlockState oldState, IBlockState newState, int flags) {
            vol.setBlock(w, pos.getX(), pos.getY(), pos.getZ(), newState);
        }
        public void notifyLightSet(BlockPos pos) { vol.dirty = true; }
        public void markBlockRangeForRenderUpdate(int x1, int y1, int z1, int x2, int y2, int z2) {}
        public void playSoundToAllNearExcept(EntityPlayer pl, SoundEvent s, SoundCategory c, double x, double y, double z, float v, float pi) {}
        public void playRecord(SoundEvent s, BlockPos pos) {}
        public void spawnParticle(int id, boolean ignoreRange, double x, double y, double z, double xs, double ys, double zs, int... params) {}
        public void spawnParticle(int id, boolean ignoreRange, boolean minimal, double x, double y, double z, double xs, double ys, double zs, int... params) {}
        public void onEntityAdded(Entity e) {}
        public void onEntityRemoved(Entity e) {}
        public void broadcastSound(int soundID, BlockPos pos, int data) {}
        public void playEvent(EntityPlayer pl, int type, BlockPos pos, int data) {}
        public void sendBlockBreakProgress(int breakerId, BlockPos pos, int progress) {}
    }
}
package com.rtrender;

import net.minecraftforge.common.config.Config;

@Config(modid = Tags.MODID, name = "rtrender")
public class RTConfig {
    @Config.Comment("Master switch. When off (or while the GPU / first snapshot is not ready) vanilla/Celeritas terrain is used.")
    public static boolean enabled = true;

    @Config.Comment("Terrain renderer. OFF = Celeritas rasterised terrain lit by GPU ray traced GI (fast, the 120 fps path). " +
            "ON = the original full screen ray tracer (slow, pretty).")
    public static boolean fullRaytrace = false;

    // ------------------------------------------------------------------ raster + GPU GI
    @Config.Comment("Raster mode: light terrain vertices with GPU ray traced GI instead of the vanilla lightmap (full light replacement).")
    public static boolean giEnabled = true;

    @Config.Comment("Raster mode: hemisphere rays per GI cell. They serve double duty - rays that escape count as sky " +
            "visibility, rays that hit a surface feed the one-bounce colour bleed. Raising this smooths the result but " +
            "slows the refresh proportionally.")
    @Config.RangeInt(min = 1, max = 64)
    public static int giRaysPerCell = 7;

    @Config.Comment("Raster mode: ray traced sun/moon shadows in the GI pass.")
    public static boolean giSunShadows = true;

    @Config.Comment("Raster mode: one bounce of colour bleed (grass bounce onto walls, etc). Costs one extra ray per hit.")
    public static boolean giBounce = true;

    @Config.Comment("Raster mode: ray traced light from torches, glowstone and other emitters.")
    public static boolean giEmitters = true;

    @Config.Comment("Raster mode: GPU time in microseconds one GI bake may spend per game tick. The whole volume " +
            "refresh rate adapts to this, so a weak GPU gets a slower refresh instead of dropped frames.")
    @Config.RangeInt(min = 100, max = 10000)
    public static int giFrameBudgetUs = 800;

    @Config.Comment("Raster mode: ticks one full GI volume refresh should take on a fast GPU. The budget above wins " +
            "if the GPU is too slow to keep up.")
    @Config.RangeInt(min = 4, max = 400)
    public static int giUpdateIntervalTicks = 40;

    @Config.Comment("Fraction of the window resolution that is ray traced (the image is scaled up). Start value when dynamic resolution is on.")
    @Config.RangeDouble(min = 0.2, max = 1.0)
    public static double renderScale = 0.5;

    @Config.Comment("Automatically lower/raise the ray traced resolution to hit targetFrameMs.")
    public static boolean dynamicResolution = true;

    @Config.Comment("Frame time budget for the ray tracer in milliseconds (33 = 30 fps).")
    @Config.RangeDouble(min = 4, max = 200)
    public static double targetFrameMs = 33;

    @Config.Comment("Primary ray length in blocks (the voxel window is 256 blocks wide, so at most ~128 are useful).")
    @Config.RangeDouble(min = 16, max = 256)
    public static double maxRayDistance = 128;

    @Config.Comment("Ray traced sun / moon shadows (1 shadow ray per pixel).")
    public static boolean sunShadows = true;

    @Config.Comment("Ray traced shadows from the strongest nearby light block (1 extra ray per lit pixel). Costly on weak GPUs.")
    public static boolean blockLightShadows = true;

    @Config.Comment("Shadow ray length in blocks.")
    @Config.RangeDouble(min = 8, max = 256)
    public static double shadowDistance = 96;

    @Config.Comment("How much vanilla block light is darkened where the strongest light block is occluded.")
    @Config.RangeDouble(min = 0, max = 1)
    public static double blockShadowDarkness = 0.6;

    @Config.Comment("Water: mirror reflections (Fresnel) and refraction. Off = sky-only reflection.")
    public static boolean waterReflections = true;

    @Config.Comment("Ripple amplitude of the water normal.")
    @Config.RangeDouble(min = 0, max = 0.2)
    public static double waveStrength = 0.035;

    @Config.Comment("Brightness of direct sunlight.")
    @Config.RangeDouble(min = 0, max = 3)
    public static double sunStrength = 1.0;

    @Config.Comment("World->voxel mirroring budget per tick (1.0 = one full 16^3 section; empty ones are nearly free). Nearest chunks first.")
    @Config.RangeDouble(min = 1, max = 512)
    public static double scanBudgetPerTick = 32;

    @Config.Comment("Radius in chunks that is mirrored (window is 16 chunks wide).")
    @Config.RangeInt(min = 2, max = 8)
    public static int radiusChunks = 8;

    @Config.Comment("Minimum ticks between scene re-publishes (tree rebuild + upload) when blocks changed.")
    @Config.RangeInt(min = 1, max = 100)
    public static int publishIntervalTicks = 3;

    @Config.Comment("OpenCL GPU index, -1 = first usable GPU.")
    @Config.RangeInt(min = -1, max = 16)
    public static int openclDevice = -1;

    @Config.Comment("Ignore Microsoft D3D12 OpenCL emulation devices.")
    public static boolean skipD3D12Devices = true;

    @Config.Comment("Account for vanilla block light levels (sky + block light). Off = fully lit surfaces (faster, less GPU work).")
    public static boolean accountForLight = true;

    @Config.Comment("Render a heatmap of per-pixel ray traversal cost instead of the scene (blue=few steps .. red=many).")
    public static boolean debugHeatmap = false;

    @Config.Comment("Log timings now and then.")
    public static boolean debugLog = false;
}

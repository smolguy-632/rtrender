package com.rtrender;

import net.minecraftforge.common.config.Config;

@Config(modid = Tags.MODID, name = "rtrender")
public class RTConfig {
    @Config.Comment("Master switch. When off (or while the GPU / first snapshot is not ready) vanilla/Celeritas terrain is used.")
    public static boolean enabled = true;

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

    @Config.Comment("Log timings now and then.")
    public static boolean debugLog = false;
}

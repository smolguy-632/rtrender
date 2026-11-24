package com.rtrender;

import com.rtrender.client.RTRenderer;
import net.minecraft.client.Minecraft;
import net.minecraftforge.client.event.TextureStitchEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.common.config.Config;
import net.minecraftforge.common.config.ConfigManager;
import net.minecraftforge.event.world.WorldEvent;
import net.minecraftforge.fml.client.event.ConfigChangedEvent;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;

@Mod(modid = Tags.MODID, name = Tags.MODNAME, version = Tags.VERSION,
     acceptedMinecraftVersions = "[1.12.2]", clientSideOnly = true,
     dependencies = "required-after:mixinbooter;after:celeritas")
public class RTRenderMod {
    @Mod.EventHandler
    public void preInit(FMLPreInitializationEvent e) {
        MinecraftForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void onConfigChanged(ConfigChangedEvent.OnConfigChangedEvent e) {
        if (Tags.MODID.equals(e.getModID())) ConfigManager.sync(Tags.MODID, Config.Type.INSTANCE);
    }

    @SubscribeEvent
    public void onTick(TickEvent.ClientTickEvent e) {
        if (e.phase == TickEvent.Phase.END && RTConfig.enabled) RTRenderer.INSTANCE.onClientTick(Minecraft.getMinecraft());
    }

    @SubscribeEvent
    public void onWorldLoad(WorldEvent.Load e) {
        if (e.getWorld().isRemote) RTRenderer.INSTANCE.onWorldLoad(e.getWorld());
    }

    @SubscribeEvent
    public void onWorldUnload(WorldEvent.Unload e) {
        if (e.getWorld().isRemote) RTRenderer.INSTANCE.onWorldUnload(e.getWorld());
    }

    /** Resource reload / atlas re-stitch: sprite UVs changed, so rebuild palette + volume. */
    @SubscribeEvent
    public void onStitch(TextureStitchEvent.Post e) {
        if (e.getMap() == Minecraft.getMinecraft().getTextureMapBlocks()) RTRenderer.INSTANCE.onAtlasReload();
    }
}

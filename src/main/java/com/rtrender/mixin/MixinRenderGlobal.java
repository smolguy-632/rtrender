package com.rtrender.mixin;

import com.rtrender.RTConfig;
import com.rtrender.client.RTRenderer;
import net.minecraft.client.renderer.RenderGlobal;
import net.minecraft.entity.Entity;
import net.minecraft.util.BlockRenderLayer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(value = RenderGlobal.class, priority = 2500)
public abstract class MixinRenderGlobal {
    @Inject(method = "renderBlockLayer(Lnet/minecraft/util/BlockRenderLayer;DILnet/minecraft/entity/Entity;)I",
            at = @At("HEAD"), cancellable = true)
    private void rtrender$replaceTerrain(BlockRenderLayer layer, double partialTicks, int pass, Entity view,
                                         CallbackInfoReturnable<Integer> cir) {
        if (!RTConfig.fullRaytrace) return;   // raster mode: Celeritas draws the terrain, rtrender only feeds GI
        if (RTRenderer.INSTANCE.prepare()) {
            if (layer == BlockRenderLayer.SOLID) RTRenderer.INSTANCE.renderTerrain((float) partialTicks);
            cir.setReturnValue(1);
        }
    }
}

package com.rtrender.mixin;

import com.rtrender.client.RTRenderer;
import net.minecraft.client.renderer.RenderGlobal;
import net.minecraft.entity.Entity;
import net.minecraft.util.BlockRenderLayer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Replaces the terrain pass. Celeritas @Overwrite-s renderBlockLayer with default priority 1000; this mixin has
 * priority 2500, so it is applied AFTER Celeritas' overwrite and its HEAD injection ends up inside Celeritas' body.
 * Without Celeritas it injects into vanilla's method. setupTerrain is deliberately NOT touched: Celeritas still
 * needs it to find visible chunks for tile-entity (chest, sign...) rendering.
 */
@Mixin(value = RenderGlobal.class, priority = 2500)
public abstract class MixinRenderGlobal {
    @Inject(method = "renderBlockLayer(Lnet/minecraft/util/BlockRenderLayer;DILnet/minecraft/entity/Entity;)I",
            at = @At("HEAD"), cancellable = true)
    private void rtrender$replaceTerrain(BlockRenderLayer layer, double partialTicks, int pass, Entity view,
                                         CallbackInfoReturnable<Integer> cir) {
        if (RTRenderer.INSTANCE.prepare()) {
            if (layer == BlockRenderLayer.SOLID) RTRenderer.INSTANCE.renderTerrain((float) partialTicks);
            cir.setReturnValue(1);
        }
    }
}

package com.rtrender.mixin;

import com.rtrender.client.GIVolume;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL20;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Binds the GI volume texture + its origin/strength uniforms to Celeritas' currently bound chunk program.
 * Both hooks run while the program is active, so glUniform calls land on the right object.
 */
@Mixin(targets = "org.embeddedt.embeddium.impl.render.chunk.shader.DefaultChunkShaderInterface")
public abstract class MixinChunkShaderInterface {
    @Inject(method = "setupState(Lorg/embeddedt/embeddium/impl/render/chunk/terrain/TerrainRenderPass;)V", at = @At("TAIL"))
    private void rtrender$bindGi(Object pass, CallbackInfo ci) {
        int prog = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        if (prog > 0) GIVolume.INSTANCE.bind(prog);
    }

    @Inject(method = "setTextureSlot(Lorg/embeddedt/embeddium/impl/render/chunk/shader/ChunkShaderTextureSlot;I)V", at = @At("TAIL"))
    private void rtrender$bindGiSlot(Object slot, int unit, CallbackInfo ci) {
        int prog = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        if (prog > 0) GIVolume.INSTANCE.bind(prog);
    }
}

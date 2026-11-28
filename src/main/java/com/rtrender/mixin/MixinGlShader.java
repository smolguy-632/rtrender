package com.rtrender.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.invoke.arg.Args;
import org.spongepowered.asm.mixin.injection.ModifyArgs;

/**
 * Patches Celeritas' terrain vertex shader so the per-vertex lightmap multiply becomes a mix between the
 * vanilla lightmap and the GPU ray traced GI volume. Uniforms are declared here and bound per pass by
 * MixinChunkShaderInterface.
 */
@Mixin(targets = "org.embeddedt.embeddium.impl.gl.shader.GlShader", remap = false)
public abstract class MixinGlShader {
    @ModifyArgs(method = "<init>", at = @At("HEAD"))
    private void rtrender$patch(Args args) {
        Object src = args.get(2);
        if (!(src instanceof String)) return;
        String source = (String) src;
        if (!source.contains("_sample_lightmap")) return;
        String uniforms = "uniform sampler3D u_GIVol;\nuniform vec4 u_GIOrigin;\nuniform float u_GIStrength;\n\n";
        int mi = source.indexOf("void main()");
        if (mi >= 0 && !source.contains("uniform sampler3D u_GIVol;"))
            source = source.substring(0, mi) + uniforms + source.substring(mi);
        String old = "    vec4 lightColor = _sample_lightmap(u_LightTex, _vert_tex_light_coord);\n    v_Color = _vert_color * lightColor;\n    v_RdhFactor.rgb *= lightColor.rgb;";
        String neu = "    vec4 lightColor = _sample_lightmap(u_LightTex, _vert_tex_light_coord);\n" +
                "    vec3 _gi = texture(u_GIVol, clamp((position - u_GIOrigin.xyz) / u_GIOrigin.w, 0.0, 1.0)).rgb;\n" +
                "    lightColor.rgb = mix(lightColor.rgb, _gi, u_GIStrength);\n" +
                "    v_Color = _vert_color * lightColor;\n" +
                "    v_RdhFactor.rgb *= lightColor.rgb;";
        if (source.contains(old)) source = source.replace(old, neu);
        else if (source.contains("    v_Color = _vert_color * lightColor;")) {
            source = source.replace("    v_Color = _vert_color * lightColor;",
                    "    vec3 _gi = texture(u_GIVol, clamp((position - u_GIOrigin.xyz) / u_GIOrigin.w, 0.0, 1.0)).rgb;\n" +
                    "    lightColor.rgb = mix(lightColor.rgb, _gi, u_GIStrength);\n" +
                    "    v_Color = _vert_color * lightColor;");
        }
        args.set(2, source);
    }
}

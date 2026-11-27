package com.rtrender.client;

import org.jocl.*;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

/** OpenCL host side for the ray tracer (JOCL: Forge/LWJGL2 and Cleanroom/LWJGL3). Render thread only. */
public final class RTEngine {
    public final String deviceName;
    private final cl_context ctx;
    private final cl_command_queue queue;
    private final cl_program program;
    private final cl_kernel kernel;
    private final Buf bMasks, bPtrs, bMats, bFlags, bUV, bTint, bBox, bCol, bEmHead, bEmNext, bEmPos,
            bAtlas, bP, bIP, bOutC, bOutD;
    private int atlasW, atlasH;
    private boolean hasScene, hasAtlas;

    private final class Buf {
        cl_mem mem; long cap;
        final long flags;
        Buf(long flags) { this.flags = flags; }
        void ensure(long need) {
            need = Math.max(need, 64L);
            if (mem != null && need <= cap) return;
            if (mem != null) clReleaseMemObject(mem);
            mem = clCreateBuffer(ctx, flags, need, null, null);
            cap = need;
        }
    }

    private RTEngine(cl_platform_id plat, cl_device_id dev, String name) throws Exception {
        this.deviceName = name;
        cl_context_properties props = new cl_context_properties();
        props.addProperty(CL_CONTEXT_PLATFORM, plat);
        ctx = clCreateContext(props, 1, new cl_device_id[]{dev}, null, null, null);
        queue = clCreateCommandQueue(ctx, dev, 0, null);
        program = clCreateProgramWithSource(ctx, 1, new String[]{readKernel()}, null, null);
        try {
            clBuildProgram(program, 0, null, null, null, null);
        } catch (CLException e) {
            long[] sz = new long[1];
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, null, sz);
            byte[] log = new byte[(int) sz[0]];
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
            throw new IllegalStateException("OpenCL build failed:\n" + new String(log), e);
        }
        kernel = clCreateKernel(program, "render", null);
        long ro = CL_MEM_READ_ONLY, rw = CL_MEM_READ_WRITE;
        bMasks = new Buf(ro); bPtrs = new Buf(ro); bMats = new Buf(ro); bFlags = new Buf(ro); bUV = new Buf(ro);
        bTint = new Buf(ro); bBox = new Buf(ro); bCol = new Buf(ro); bEmHead = new Buf(ro);
        bEmNext = new Buf(ro); bEmPos = new Buf(ro); bAtlas = new Buf(ro); bP = new Buf(ro); bIP = new Buf(ro);
        bOutC = new Buf(rw); bOutD = new Buf(rw);
    }

    private static String readKernel() throws Exception {
        try (InputStream in = RTEngine.class.getResourceAsStream("/assets/rtrender/rtrender.cl")) {
            ByteArrayOutputStream bo = new ByteArrayOutputStream();
            byte[] buf = new byte[4096]; int r;
            while ((r = in.read(buf)) > 0) bo.write(buf, 0, r);
            return new String(bo.toByteArray(), "UTF-8");
        }
    }

    private static String info(cl_device_id d, int param) {
        long[] sz = new long[1];
        clGetDeviceInfo(d, param, 0, null, sz);
        byte[] b = new byte[(int) sz[0]];
        clGetDeviceInfo(d, param, b.length, Pointer.to(b), null);
        return new String(b, 0, Math.max(0, b.length - 1)).trim();
    }

    public static RTEngine create(int index, boolean skipD3D12) throws Exception {
        CL.setExceptionsEnabled(true);
        int[] np = new int[1];
        clGetPlatformIDs(0, null, np);
        cl_platform_id[] plats = new cl_platform_id[np[0]];
        clGetPlatformIDs(plats.length, plats, null);
        List<cl_platform_id> lp = new ArrayList<cl_platform_id>();
        List<cl_device_id> ld = new ArrayList<cl_device_id>();
        List<String> ln = new ArrayList<String>();
        for (cl_platform_id p : plats) {
            int[] nd = new int[1];
            try { clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, null, nd); } catch (CLException e) { continue; }
            if (nd[0] == 0) continue;
            cl_device_id[] ds = new cl_device_id[nd[0]];
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, ds.length, ds, null);
            for (cl_device_id d : ds) {
                String vendor = info(d, CL_DEVICE_VENDOR).toLowerCase();
                String ver = info(d, CL_DEVICE_VERSION).toLowerCase();
                if (skipD3D12 && (vendor.contains("microsoft") || ver.contains("d3d12"))) continue;
                lp.add(p); ld.add(d); ln.add(info(d, CL_DEVICE_NAME));
            }
        }
        if (ld.isEmpty()) throw new IllegalStateException("No usable OpenCL GPU found");
        int i = index < 0 ? 0 : Math.min(index, ld.size() - 1);
        return new RTEngine(lp.get(i), ld.get(i), ln.get(i));
    }

    private void write(Buf b, Pointer p, long bytes) {
        b.ensure(bytes);
        clEnqueueWriteBuffer(queue, b.mem, true, 0, bytes, p, 0, null, null);
    }

    public boolean hasAtlas() { return hasAtlas; }
    public boolean hasScene() { return hasScene; }

    public void uploadAtlas(ByteBuffer rgba, int w, int h) {
        rgba.position(0);
        write(bAtlas, Pointer.to(rgba), 4L * w * h);
        atlasW = w; atlasH = h; hasAtlas = true;
    }

    public void uploadScene(RTRenderer.Published p) {
        Tree64 t = p.tree;
        write(bMasks, Pointer.to(t.masks), 8L * t.nodeCount);
        write(bPtrs, Pointer.to(t.ptrs), 4L * t.nodeCount);
        write(bMats, Pointer.to(t.mats), 2L * t.mats.length);
        BlockPalette.Snapshot ps = p.palette;
        write(bFlags, Pointer.to(ps.flags), 1L * ps.count);
        short[] _uv = new short[ps.count * 24];
        for (int i = 0; i < _uv.length; i++) _uv[i] = (short) Math.round(ps.uv[i] * 65535.0f);
        write(bUV, Pointer.to(_uv), 2L * _uv.length);
        write(bTint, Pointer.to(ps.tint), 4L * ps.count * 6);
        short[] _bx = new short[ps.count * 6];
        for (int i = 0; i < _bx.length; i++) _bx[i] = (short) Math.round(ps.box[i] * 65535.0f);
        write(bBox, Pointer.to(_bx), 2L * _bx.length);
        write(bCol, Pointer.to(p.colTint), 4L * p.colTint.length);
        Emitters e = p.emitters;
        write(bEmHead, Pointer.to(e.head), 4L * e.head.length);
        write(bEmNext, Pointer.to(e.next), 4L * Math.max(1, e.n));
        write(bEmPos, Pointer.to(e.pos), 16L * Math.max(1, e.n));
        hasScene = true;
    }

    private int arg;
    private void argMem(Buf b) { clSetKernelArg(kernel, arg++, Sizeof.cl_mem, Pointer.to(b.mem)); }

    /** Traces w*h pixels and reads colour (RGBA8, premultiplied) and window depth (float) back. @return ms */
    public double render(float[] P, int[] IP, int w, int h, ByteBuffer color, ByteBuffer depth) {
        long t0 = System.nanoTime();
        write(bP, Pointer.to(P), 4L * P.length);
        write(bIP, Pointer.to(IP), 4L * IP.length);
        long px = (long) w * h;
        bOutC.ensure(4L * px); bOutD.ensure(4L * px);
        arg = 0;
        argMem(bMasks); argMem(bPtrs); argMem(bMats); argMem(bFlags); argMem(bUV); argMem(bTint); argMem(bBox);
        argMem(bCol); argMem(bEmHead); argMem(bEmNext); argMem(bEmPos); argMem(bAtlas);
        argMem(bP); argMem(bIP); argMem(bOutC); argMem(bOutD);
        // the kernel maps work items to 8x8 pixel tiles (coherent rays); partial edge tiles are masked off in the kernel
        long tiles = (long) ((w + 7) >> 3) * ((h + 7) >> 3);
        clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[]{tiles * 64L}, null, 0, null, null);
        color.position(0); depth.position(0);
        clEnqueueReadBuffer(queue, bOutC.mem, true, 0, 4L * px, Pointer.to(color), 0, null, null);
        clEnqueueReadBuffer(queue, bOutD.mem, true, 0, 4L * px, Pointer.to(depth), 0, null, null);
        return (System.nanoTime() - t0) / 1e6;
    }
}

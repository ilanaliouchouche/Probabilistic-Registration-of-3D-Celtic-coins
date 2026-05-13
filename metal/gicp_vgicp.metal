#include <metal_stdlib>
using namespace metal;

struct VoxelBucket {
    int x;
    int y;
    int z;
    int index;
};

struct Params {
    uint source_count;
    uint voxel_count;
    uint bucket_count;
    uint offset_count;

    float voxel_resolution;
    uint bucket_mask;
    float max_corr_dist_sq;
    uint _pad0;

    float r00;
    float r01;
    float r02;
    float tx;

    float r10;
    float r11;
    float r12;
    float ty;

    float r20;
    float r21;
    float r22;
    float tz;
};

struct ThreadAccum {
    float h[21];
    float g[6];
    float err;
    float count;
};

inline float3 transform_point(const constant Params& p, float3 x) {
    return float3(
        p.r00 * x.x + p.r01 * x.y + p.r02 * x.z + p.tx,
        p.r10 * x.x + p.r11 * x.y + p.r12 * x.z + p.ty,
        p.r20 * x.x + p.r21 * x.y + p.r22 * x.z + p.tz
    );
}

inline int3 voxel_coord(float3 x, float resolution) {
    return int3(
        int(floor(x.x / resolution)),
        int(floor(x.y / resolution)),
        int(floor(x.z / resolution))
    );
}

inline uint hash_coord(int3 c) {
    uint x = as_type<uint>(c.x);
    uint y = as_type<uint>(c.y);
    uint z = as_type<uint>(c.z);
    return x * 73856093u ^ y * 19349663u ^ z * 83492791u;
}

inline int lookup_voxel(int3 coord,
                        const device VoxelBucket* buckets,
                        uint bucket_count,
                        uint bucket_mask) {
    uint h = hash_coord(coord);
    uint max_scan = min(bucket_count, (uint)128);
    for (uint i = 0; i < max_scan; ++i) {
        uint idx = (h + i) & bucket_mask;
        VoxelBucket b = buckets[idx];
        if (b.index < 0) {
            return -1;
        }
        if (b.x == coord.x && b.y == coord.y && b.z == coord.z) {
            return b.index;
        }
    }
    return -1;
}

inline void load_cov_rm(const device float* covs, uint idx, thread float c[9]) {
    const device float* p = covs + idx * 9;
    for (int i = 0; i < 9; ++i) c[i] = p[i];
}

inline void build_R_rm(const constant Params& p, thread float R[9]) {
    R[0] = p.r00; R[1] = p.r01; R[2] = p.r02;
    R[3] = p.r10; R[4] = p.r11; R[5] = p.r12;
    R[6] = p.r20; R[7] = p.r21; R[8] = p.r22;
}

inline void matmul3_rm(const thread float A[9], const thread float B[9], thread float C[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            C[r * 3 + c] =
                A[r * 3 + 0] * B[0 * 3 + c] +
                A[r * 3 + 1] * B[1 * 3 + c] +
                A[r * 3 + 2] * B[2 * 3 + c];
        }
    }
}

inline void transpose3_rm(const thread float A[9], thread float At[9]) {
    At[0] = A[0]; At[1] = A[3]; At[2] = A[6];
    At[3] = A[1]; At[4] = A[4]; At[5] = A[7];
    At[6] = A[2]; At[7] = A[5]; At[8] = A[8];
}

inline bool inverse3_rm(const thread float A[9], thread float invA[9]) {
    float a00 = A[0], a01 = A[1], a02 = A[2];
    float a10 = A[3], a11 = A[4], a12 = A[5];
    float a20 = A[6], a21 = A[7], a22 = A[8];

    float c00 = a11 * a22 - a12 * a21;
    float c01 = a02 * a21 - a01 * a22;
    float c02 = a01 * a12 - a02 * a11;
    float c10 = a12 * a20 - a10 * a22;
    float c11 = a00 * a22 - a02 * a20;
    float c12 = a02 * a10 - a00 * a12;
    float c20 = a10 * a21 - a11 * a20;
    float c21 = a01 * a20 - a00 * a21;
    float c22 = a00 * a11 - a01 * a10;

    float det = a00 * c00 + a01 * c10 + a02 * c20;
    if (!isfinite(det) || fabs(det) < 1e-9f) {
        return false;
    }

    float inv_det = 1.0f / det;
    invA[0] = c00 * inv_det; invA[1] = c01 * inv_det; invA[2] = c02 * inv_det;
    invA[3] = c10 * inv_det; invA[4] = c11 * inv_det; invA[5] = c12 * inv_det;
    invA[6] = c20 * inv_det; invA[7] = c21 * inv_det; invA[8] = c22 * inv_det;
    return true;
}

inline void add_upper_6x6(thread ThreadAccum& a, const thread float H[36], float w) {
    a.h[0] += w * H[0];
    a.h[1] += w * H[1];
    a.h[2] += w * H[2];
    a.h[3] += w * H[3];
    a.h[4] += w * H[4];
    a.h[5] += w * H[5];

    a.h[6] += w * H[7];
    a.h[7] += w * H[8];
    a.h[8] += w * H[9];
    a.h[9] += w * H[10];
    a.h[10] += w * H[11];

    a.h[11] += w * H[14];
    a.h[12] += w * H[15];
    a.h[13] += w * H[16];
    a.h[14] += w * H[17];

    a.h[15] += w * H[21];
    a.h[16] += w * H[22];
    a.h[17] += w * H[23];

    a.h[18] += w * H[28];
    a.h[19] += w * H[29];

    a.h[20] += w * H[35];
}

kernel void vgicp_accumulate(
    const device float4* source_points [[buffer(0)]],
    const device float* source_covs [[buffer(1)]],
    const device VoxelBucket* voxel_buckets [[buffer(2)]],
    const device float4* voxel_means [[buffer(3)]],
    const device float* voxel_covs [[buffer(4)]],
    const device uint* voxel_counts [[buffer(5)]],
    const device int4* offsets [[buffer(6)]],
    const constant Params& params [[buffer(7)]],
    device float* partial_out [[buffer(8)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tg_size_v [[threads_per_threadgroup]]) {

    const uint gid = gid3.x;
    const uint lid = lid3.x;
    const uint tg_size = tg_size_v.x;

    threadgroup float t_h[256 * 21];
    threadgroup float t_g[256 * 6];
    threadgroup float t_err[256];
    threadgroup float t_cnt[256];

    ThreadAccum acc;
    for (int i = 0; i < 21; ++i) acc.h[i] = 0.0f;
    for (int i = 0; i < 6; ++i) acc.g[i] = 0.0f;
    acc.err = 0.0f;
    acc.count = 0.0f;

    if (gid < params.source_count) {
        float3 p = source_points[gid].xyz;
        float3 Rp = transform_point(params, p);

        float Cs[9];
        load_cov_rm(source_covs, gid, Cs);

        float R[9];
        build_R_rm(params, R);
        float Rt[9];
        transpose3_rm(R, Rt);

        float tmp[9];
        float Crot[9];
        matmul3_rm(R, Cs, tmp);
        matmul3_rm(tmp, Rt, Crot);

        const int3 base = voxel_coord(Rp, params.voxel_resolution);

        for (uint oi = 0; oi < params.offset_count; ++oi) {
            int3 qcoord = base + offsets[oi].xyz;
            int voxel_idx = lookup_voxel(qcoord,
                                         voxel_buckets,
                                         params.bucket_count,
                                         params.bucket_mask);
            if (voxel_idx < 0) {
                continue;
            }

            float Ct[9];
            load_cov_rm(voxel_covs, (uint)voxel_idx, Ct);

            float C[9];
            for (int i = 0; i < 9; ++i) C[i] = Ct[i] + Crot[i];

            float W[9];
            if (!inverse3_rm(C, W)) {
                continue;
            }

            float3 q = voxel_means[(uint)voxel_idx].xyz;
            float3 e = q - Rp;
            if (params.max_corr_dist_sq > 0.0f && dot(e, e) > params.max_corr_dist_sq) {
                continue;
            }

            // J (3x6)
            float J[18];
            J[0] = 1.0f; J[1] = 0.0f; J[2] = 0.0f; J[3] = 0.0f;   J[4] = Rp.z;  J[5] = -Rp.y;
            J[6] = 0.0f; J[7] = 1.0f; J[8] = 0.0f; J[9] = -Rp.z;  J[10] = 0.0f;  J[11] = Rp.x;
            J[12] = 0.0f; J[13] = 0.0f; J[14] = 1.0f; J[15] = Rp.y; J[16] = -Rp.x; J[17] = 0.0f;

            const float w = sqrt((float)max(voxel_counts[(uint)voxel_idx], 1u));

            float We[3];
            We[0] = W[0] * e.x + W[1] * e.y + W[2] * e.z;
            We[1] = W[3] * e.x + W[4] * e.y + W[5] * e.z;
            We[2] = W[6] * e.x + W[7] * e.y + W[8] * e.z;

            for (int r = 0; r < 6; ++r) {
                acc.g[r] += w * (J[0 * 6 + r] * We[0] + J[1 * 6 + r] * We[1] + J[2 * 6 + r] * We[2]);
            }

            float Hfull[36];
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) {
                    float s = 0.0f;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            s += J[a * 6 + r] * W[a * 3 + b] * J[b * 6 + c];
                        }
                    }
                    Hfull[r * 6 + c] = s;
                }
            }

            add_upper_6x6(acc, Hfull, w);

            acc.err += w * (e.x * We[0] + e.y * We[1] + e.z * We[2]);
            acc.count += 1.0f;
        }
    }

    if (lid < 256) {
        for (int i = 0; i < 21; ++i) t_h[lid * 21 + i] = acc.h[i];
        for (int i = 0; i < 6; ++i) t_g[lid * 6 + i] = acc.g[i];
        t_err[lid] = acc.err;
        t_cnt[lid] = acc.count;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            uint b = lid + stride;
            for (int i = 0; i < 21; ++i) t_h[lid * 21 + i] += t_h[b * 21 + i];
            for (int i = 0; i < 6; ++i) t_g[lid * 6 + i] += t_g[b * 6 + i];
            t_err[lid] += t_err[b];
            t_cnt[lid] += t_cnt[b];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        uint base_out = tg.x * 29;
        for (int i = 0; i < 21; ++i) partial_out[base_out + i] = t_h[i];
        for (int i = 0; i < 6; ++i) partial_out[base_out + 21 + i] = t_g[i];
        partial_out[base_out + 27] = t_err[0];
        partial_out[base_out + 28] = t_cnt[0];
    }
}

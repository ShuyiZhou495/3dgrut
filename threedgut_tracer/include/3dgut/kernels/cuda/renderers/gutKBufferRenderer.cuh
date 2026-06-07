// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

struct HitParticle {
    static constexpr float InvalidHitT = -1.0f;
    int idx                            = -1;
    float hitT                         = InvalidHitT;
    float alpha                        = 0.0f;
    uint32_t contributor               = 0;
    tcnn::vec3 normal                  = tcnn::vec3::zero();
};

template <int K>
struct HitParticleKBuffer {
    __device__ HitParticleKBuffer() {
        m_numHits = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle();
        }
    }

    // insert a new hit into the kbuffer.
    // if the buffer is full overwrite the closest entry
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        if (isFull) {
            m_kbuffer[0].hitT = HitParticle::InvalidHitT;
        } else {
            m_numHits++;
        }
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
        }
    }

    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    inline __device__ uint32_t numHits() const {
        return m_numHits;
    }

    inline __device__ bool full() const {
        return m_numHits == K;
    }

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];
    }

private:
    HitParticle m_kbuffer[K];
    uint32_t m_numHits;
};

template <>
struct HitParticleKBuffer<0> {
    constexpr inline __device__ void insert(HitParticle& hitParticle) const {}
    constexpr inline __device__ HitParticle operator[](int) const { return HitParticle(); }
    constexpr inline __device__ uint32_t numHits() const { return 0; }
    constexpr inline __device__ bool full() const { return true; }
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const { return hitParticle; }
};

template <typename Particles, typename Params, bool Backward = false>
struct GUTKBufferRenderer : Params {

    using DensityParameters    = typename Particles::DensityParameters;
    using DensityRawParameters = typename Particles::DensityRawParameters;
    using TFeaturesVec         = typename Particles::TFeaturesVec;

    using TRayPayload         = RayPayload<Particles::FeaturesDim>;
    using TRayPayloadBackward = RayPayloadBackward<Particles::FeaturesDim>;

    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };

    struct PrefetchedRawParticleData {
        uint32_t idx;
        TFeaturesVec features;
        DensityRawParameters densityParameters;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles,
        const TFeaturesVec* __restrict__ particleFeatures,
        TFeaturesVec* __restrict__ particleFeaturesGradient) {

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;
            if constexpr (Params::PerRayParticleFeatures) {
                particles.featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                              hitParticle.alpha,
                                                              hitAlphaGrad,
                                                              hitParticle.idx,
                                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                              ray.featuresBackward,
                                                              ray.featuresGradient);
            } else {
                TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                particles.featuresIntegrateBwd(hitParticle.alpha,
                                               hitAlphaGrad,
                                               particleFeatures[hitParticle.idx],
                                               particleFeaturesGradientVec,
                                               ray.featuresBackward,
                                               ray.featuresGradient);
#pragma unroll
                for (int i = 0; i < Particles::FeaturesDim; ++i) {
                    atomicAdd(&(particleFeaturesGradient[hitParticle.idx][i]), particleFeaturesGradientVec[i]);
                }
            }

            particles.densityProcessHitBwdToBuffer<false>(ray.origin,
                                                          ray.direction,
                                                          hitParticle.idx,
                                                          hitParticle.alpha,
                                                          hitAlphaGrad,
                                                          ray.transmittanceBackward,
                                                          ray.transmittanceGradient,
                                                          hitParticle.hitT,
                                                          ray.hitTBackward,
                                                          ray.hitTGradient);

            ray.transmittance *= (1.0 - hitParticle.alpha);

        } else {
            if (ray.transmittance > 0.5f) {
                ray.depthInitT = hitParticle.hitT;
            }
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,
                                              ray.transmittance,
                                              hitParticle.hitT,
                                              ray.hitT,
                                              &hitParticle.normal,
                                              &ray.normal);

            particles.featureIntegrateFwd(hitWeight,
                                          Params::PerRayParticleFeatures ? particles.featuresFromBuffer(hitParticle.idx, ray.direction) : tcnn::max(particleFeatures[hitParticle.idx], 0.f),
                                          ray.features);

            if (hitWeight > 0.0f)
                ray.countHit();
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <typename TRay>
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       const float* __restrict__ /*particlesGlobalDepthPtr*/,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                       float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                       threedgut::MemoryHandles parametersGradient                        = {}) {

        using namespace threedgut;

        const uint32_t tileIdx                       = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices   = sortedTileRangeIndicesPtr[tileIdx];
        uint32_t tileNumParticlesToProcess           = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess        = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);
        const TFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBuffer(params, ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        } else {
            evalKBuffer(params, ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        }
    }

    template <typename TRay>
    static inline __device__ bool gggsPixelCandidate(const threedgut::RenderParameters& params,
                                                     const TRay& ray,
                                                     const uint32_t particleIdx,
                                                     const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                     const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr) {
        if ((particlesProjectedPositionPtr == nullptr) || (particlesProjectedConicOpacityPtr == nullptr)) {
            return true;
        }

        const tcnn::vec2 projectedPosition = particlesProjectedPositionPtr[particleIdx];
        const tcnn::vec4 conicOpacity      = particlesProjectedConicOpacityPtr[particleIdx];
        if (conicOpacity.w <= 0.0f) {
            return false;
        }

        const float pixelX = static_cast<float>(ray.idx % params.resolution.x);
        const float pixelY = static_cast<float>(ray.idx / params.resolution.x);
        const float dx     = projectedPosition.x - pixelX;
        const float dy     = projectedPosition.y - pixelY;
        const float power  = -0.5f * (conicOpacity.x * dx * dx + conicOpacity.z * dy * dy) - conicOpacity.y * dx * dy;
        if (power > 0.0f) {
            return false;
        }

        const float alpha = fminf(0.99f, conicOpacity.w * expf(power));
        return alpha >= Params::AlphaThreshold;
    }

    template <typename TRay>
    static inline __device__ float gggsTransmittance(TRay& ray,
                                                     const threedgut::RenderParameters& params,
                                                     Particles& particles,
                                                     const tcnn::uvec2& tileParticleRangeIndices,
                                                     const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                     const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                     const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                                     const uint32_t lastContributor,
                                                     const float depth) {
        float transmittance = 1.0f;
        uint32_t contributor = 0;

        for (uint32_t sortedIndex = tileParticleRangeIndices.x; sortedIndex < tileParticleRangeIndices.y; ++sortedIndex) {
            const uint32_t particleIdx = sortedTileParticleIdxPtr[sortedIndex];
            if (particleIdx == threedgut::GUTParameters::InvalidParticleIdx) {
                break;
            }
            contributor++;
            if ((lastContributor > 0) && (contributor > lastContributor)) {
                break;
            }
            if (!gggsPixelCandidate(params, ray, particleIdx, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr)) {
                continue;
            }

            const auto densityParameters = particles.fetchDensityParameters(particleIdx);
            threedgut::GGGSRayProfile profile;
            if (!particles.gggsDepthProfile(ray.origin,
                                            ray.direction,
                                            densityParameters,
                                            ray.tMinMax.x,
                                            ray.tMinMax.y,
                                            false,
                                            profile)) {
                continue;
            }

            transmittance *= particles.gggsDepthProfileTransmittance(profile, depth);
        }

        return transmittance;
    }

    template <typename TRay>
    static inline __device__ float gggsLogTransmittanceDepthDerivative(TRay& ray,
                                                                       const threedgut::RenderParameters& params,
                                                                       Particles& particles,
                                                                       const tcnn::uvec2& tileParticleRangeIndices,
                                                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                                                       const uint32_t lastContributor,
                                                                       const float depth) {
        float derivative = 0.0f;
        uint32_t contributor = 0;

        for (uint32_t sortedIndex = tileParticleRangeIndices.x; sortedIndex < tileParticleRangeIndices.y; ++sortedIndex) {
            const uint32_t particleIdx = sortedTileParticleIdxPtr[sortedIndex];
            if (particleIdx == threedgut::GUTParameters::InvalidParticleIdx) {
                break;
            }
            contributor++;
            if ((lastContributor > 0) && (contributor > lastContributor)) {
                break;
            }
            if (!gggsPixelCandidate(params, ray, particleIdx, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr)) {
                continue;
            }

            const auto densityParameters = particles.fetchDensityParameters(particleIdx);
            threedgut::GGGSRayProfile profile;
            if (!particles.gggsDepthProfile(ray.origin,
                                            ray.direction,
                                            densityParameters,
                                            ray.tMinMax.x,
                                            ray.tMinMax.y,
                                            false,
                                            profile)) {
                continue;
            }

            derivative += particles.gggsDepthProfileDLogSDt(profile, depth);
        }

        return derivative;
    }

    template <typename TRay>
    static inline __device__ void initializeGGGSMedianDepth(TRay& ray,
                                                            const threedgut::RenderParameters& params,
                                                            Particles& particles,
                                                            const tcnn::uvec2& tileParticleRangeIndices,
                                                            const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                            const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                            const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr) {
        if (!ray.isValid()) {
            return;
        }

        const float fallbackDepthInitT = ray.depthInitT;
        ray.depthInitT = 0.0f;
        ray.gggsLastContributor = 0;
        float transmittance = 1.0f;
        float depthInitT = 0.0f;
        uint32_t lastContributor = 0;
        uint32_t contributor = 0;

        for (uint32_t sortedIndex = tileParticleRangeIndices.x; sortedIndex < tileParticleRangeIndices.y; ++sortedIndex) {
            const uint32_t particleIdx = sortedTileParticleIdxPtr[sortedIndex];
            if (particleIdx == threedgut::GUTParameters::InvalidParticleIdx) {
                break;
            }
            contributor++;

            const auto densityParameters = particles.fetchDensityParameters(particleIdx);
            float rgbAlpha = 0.0f;
            float rgbHitT = 0.0f;
            if (!particles.densityHit(ray.origin,
                                      ray.direction,
                                      densityParameters,
                                      rgbAlpha,
                                      rgbHitT,
                                      nullptr)) {
                continue;
            }

            threedgut::GGGSRayProfile profile;
            if (!particles.gggsDepthProfileForInit(ray.origin,
                                                   ray.direction,
                                                   densityParameters,
                                                   ray.tMinMax.x,
                                                   ray.tMinMax.y,
                                                   profile)) {
                continue;
            }

            if ((rgbAlpha < Params::AlphaThreshold) ||
                (rgbHitT <= ray.tMinMax.x) ||
                (rgbHitT >= ray.tMinMax.y) ||
                (profile.tPeak <= ray.tMinMax.x) ||
                (profile.tPeak >= ray.tMinMax.y)) {
                continue;
            }

            if (transmittance > 0.5f) {
                depthInitT = profile.tPeak;
                lastContributor = contributor;
            }
            const float nextTransmittance = transmittance * (1.0f - rgbAlpha);
            if ((transmittance > 0.5f) && (nextTransmittance <= 0.5f)) {
                break;
            }
            transmittance = nextTransmittance;
            if (transmittance < Particles::MinTransmittanceThreshold) {
                break;
            }
        }

        if (lastContributor > 0) {
            ray.depthInitT = depthInitT;
            ray.gggsLastContributor = lastContributor;
        } else if (fallbackDepthInitT > ray.tMinMax.x) {
            ray.depthInitT = fallbackDepthInitT;
        }
    }

    template <typename TRay>
    static inline __device__ void resolveGGGSMedianDepth(TRay& ray,
                                                         const threedgut::RenderParameters& params,
                                                         Particles& particles,
                                                         const tcnn::uvec2& tileParticleRangeIndices,
                                                         const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                         const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                         const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr) {
        if (!ray.isValid()) {
            return;
        }

        constexpr float SampleRange = 0.4f;
        constexpr float MinTransmittanceForDepth = 0.45f;
        constexpr int MaxBracketExpansions = 5;
        constexpr int Split = 8;
        constexpr int SplitIterations = 5;
        constexpr uint32_t AllPixelCandidates = 0;
        const float finalTransmittance = gggsTransmittance(ray,
                                                           params,
                                                           particles,
                                                           tileParticleRangeIndices,
                                                           sortedTileParticleIdxPtr,
                                                           particlesProjectedPositionPtr,
                                                           particlesProjectedConicOpacityPtr,
                                                           AllPixelCandidates,
                                                           ray.tMinMax.y);
        ray.gggsDebug = {ray.depthInitT, finalTransmittance, 0.0f, 0.0f};
        if (ray.depthInitT <= ray.tMinMax.x) {
            ray.gggsDebug.z = 1.0f;
            return;
        }
        float lo = fmaxf(ray.depthInitT - SampleRange, ray.tMinMax.x);
        float hi = fminf(ray.depthInitT + SampleRange, ray.tMinMax.y);
        if (hi <= lo) {
            ray.gggsDebug.z = 1.0f;
            return;
        }

        const float debugLo = lo;
        const float debugMid = ray.depthInitT;
        const float debugHi = hi;
        ray.gggsTransmittanceDebug = {
            gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, debugLo),
            gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, debugMid),
            gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, debugHi),
            static_cast<float>(ray.gggsLastContributor)};
        if (finalTransmittance > MinTransmittanceForDepth) {
            ray.gggsDebug.z = 2.0f;
            return;
        }

        float bracketTLo = ray.gggsTransmittanceDebug.x;
        float bracketTHi = ray.gggsTransmittanceDebug.z;
        float sampleRange = SampleRange;
        bool bracketed = (bracketTLo >= 0.5f) && (bracketTHi <= 0.5f);
        for (int expand = 0; !bracketed && (expand < MaxBracketExpansions); ++expand) {
            if ((lo <= ray.tMinMax.x) && (hi >= ray.tMinMax.y)) {
                break;
            }
            sampleRange *= 2.0f;
            lo = fmaxf(ray.depthInitT - sampleRange, ray.tMinMax.x);
            hi = fminf(ray.depthInitT + sampleRange, ray.tMinMax.y);
            bracketTLo = gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, lo);
            bracketTHi = gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, hi);
            bracketed = (bracketTLo >= 0.5f) && (bracketTHi <= 0.5f);
        }

        if (!bracketed) {
            ray.gggsDebug.z = bracketTLo < 0.5f ? 3.0f : 4.0f;
            return;
        }

        float T[Split + 1];
        for (int iter = 0; iter < SplitIterations; ++iter) {
            const float interval = (hi - lo) / static_cast<float>(Split);
#pragma unroll
            for (int i = 0; i <= Split; ++i) {
                const float t = lo + static_cast<float>(i) * interval;
                T[i] = gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, t);
            }

            int startId = 0;
#pragma unroll
            for (int i = 1; i < Split; ++i) {
                startId = T[i] >= 0.5f ? i : startId;
            }

            hi = lo + static_cast<float>(startId + 1) * interval;
            lo = lo + static_cast<float>(startId) * interval;
        }

        const float TLo = gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, lo);
        const float THi = gggsTransmittance(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, hi);
        const float wHi = fminf(fmaxf((TLo - 0.5f) / fmaxf(TLo - THi, 1.0e-7f), 0.0f), 1.0f);
        ray.hitT = wHi * hi + (1.0f - wHi) * lo;
        ray.gggsDebug.z = 0.0f;
        ray.gggsDebug.w = ray.hitT - ray.depthInitT;
    }

    template <typename TRay>
    static inline __device__ void evalKBuffer(const threedgut::RenderParameters& params,
                                              TRay& ray,
                                              Particles& particles,
                                              const tcnn::uvec2& tileParticleRangeIndices,
                                              uint32_t tileNumBlocksToProcess,
                                              uint32_t tileNumParticlesToProcess,
                                              const uint32_t tileThreadIdx,
                                              const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                              const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                              const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                              const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                              TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        using namespace threedgut;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleKBuffer<Params::KHitBufferSize> hitParticleKBuffer;
        uint32_t contributor = 0;

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }
                contributor++;

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;
                hitParticle.contributor = contributor;
                if (particles.densityHit(ray.origin,
                                         ray.direction,
                                         particleData.densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT,
                                         &hitParticle.normal) &&
                    (hitParticle.hitT > ray.tMinMax.x) &&
                    (hitParticle.hitT < ray.tMinMax.y)) {

                    if (hitParticleKBuffer.full()) {
                        const HitParticle hitToProcess = hitParticleKBuffer.closestHit(hitParticle);
                        processHitParticle(ray,
                                           hitToProcess,
                                           particles,
                                           particleFeaturesBuffer,
                                           particleFeaturesGradientBuffer);
                        ray.gggsLastContributor = ray.gggsLastContributor > hitToProcess.contributor ? ray.gggsLastContributor : hitToProcess.contributor;
                    }
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                const HitParticle hitToProcess = hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i];
                processHitParticle(ray,
                                   hitToProcess,
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer);
                ray.gggsLastContributor = ray.gggsLastContributor > hitToProcess.contributor ? ray.gggsLastContributor : hitToProcess.contributor;
            }
        }

        if constexpr (!Backward) {
            if (params.depthMode == threedgut::RenderParameters::GGGSMedianDepth) {
                initializeGGGSMedianDepth(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr);
                resolveGGGSMedianDepth(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr);
            }
        }
    }

    // Fine-grained balanced forward rendering: Gaussian-wise parallelism with warp-level optimization
    template <typename TRay>
    static inline __device__ void evalForwardNoKBufferBalanced(
        const threedgut::RenderParameters& params,
        TRay& ray,
        const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
        const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
        const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
        const float* __restrict__ particlesGlobalDepthPtr,
        const float* __restrict__ particlesPrecomputedFeaturesPtr,
        const tcnn::uvec2& tile,
        const tcnn::uvec2& tileGrid,
        const int laneId,
        threedgut::MemoryHandles parameters,
        tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr     = nullptr,
        tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr = nullptr,
        float* __restrict__ particlesGlobalDepthGradPtr                = nullptr,
        float* __restrict__ particlesPrecomputedFeaturesGradPtr        = nullptr,
        threedgut::MemoryHandles parametersGradient                    = {}) {

        using namespace threedgut;

        // Get tile data: each warp processes particles from a single 16x16 tile
        const uint32_t tileIdx                     = tile.y * tileGrid.x + tile.x;
        const tcnn::uvec2 tileParticleRangeIndices = sortedTileRangeIndicesPtr[tileIdx];

        uint32_t tileNumParticlesToProcess = tileParticleRangeIndices.y - tileParticleRangeIndices.x;

        // Setup feature buffers based on rendering mode
        const TFeaturesVec* particleFeaturesBuffer =
            Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer =
            (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        // Initialize particle system
        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        static_assert(Params::KHitBufferSize == 0, "evalForwardNoKBufferBalanced only supports K=0 (no hit buffer). Use evalKBuffer for K>0 cases.");

        // Warp-aligned processing: round up to multiple of WarpSize to avoid divergence
        constexpr uint32_t WarpSize   = GUTParameters::Tiling::WarpSize; // 32 threads per warp
        uint32_t alignedParticleCount = ((tileNumParticlesToProcess + WarpSize - 1) / WarpSize) * WarpSize;

        // Main loop: Gaussian-wise parallelism - WarpSize threads process Gaussians, single ray
        for (uint32_t j = laneId; j < alignedParticleCount; j += WarpSize) {
            if (!ray.isAlive())
                break;

            float hitAlpha           = 0.0f;
            float hitT               = 0.0f;
            tcnn::vec3 hitNormal     = tcnn::vec3::zero();
            TFeaturesVec hitFeatures = TFeaturesVec::zero();
            bool validHit            = false;

            // Step 1: Each thread tests one Gaussian intersection
            if (j < tileNumParticlesToProcess) {
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + j;
                const uint32_t particleIdx          = sortedTileParticleIdxPtr[toProcessSortedIndex];

                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    auto densityParams = particles.fetchDensityParameters(particleIdx);

                    if (particles.densityHit(ray.origin,
                                             ray.direction,
                                             densityParams,
                                             hitAlpha,
                                             hitT,
                                             &hitNormal) &&
                        (hitT > ray.tMinMax.x) &&
                        (hitT < ray.tMinMax.y)) {

                        validHit = true;

                        // Get Gaussian features
                        if constexpr (Params::PerRayParticleFeatures) {
                            hitFeatures = particles.featuresFromBuffer(particleIdx, ray.direction);
                        } else {
                            hitFeatures = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                        }
                    }
                }
            }

            // Skip if no hits in this warp batch
            constexpr uint32_t WarpMask = GUTParameters::Tiling::WarpMask; // 0xFFFFFFFF for full warp
            if (__all_sync(WarpMask, !validHit))
                continue;

            // Step 2: Compute per-thread transmittance contribution
            float localTransmittance = validHit ? (1.0f - hitAlpha) : 1.0f;

            // Step 3: Warp-level prefix scan for cumulative transmittance
            for (uint32_t offset = 1; offset < WarpSize; offset <<= 1) {
                float n = __shfl_up_sync(WarpMask, localTransmittance, offset);
                if (laneId >= offset) {
                    localTransmittance *= n;
                }
            }

            // Get overall batch transmittance impact
            float batchTransmittance = __shfl_sync(WarpMask, localTransmittance, WarpSize - 1);
            float newTransmittance   = ray.transmittance * batchTransmittance;

            // Step 4: Early termination detection - find exact termination point
            unsigned int earlyTerminationMask = __ballot_sync(WarpMask,
                                                              validHit && (ray.transmittance * localTransmittance) < Particles::MinTransmittanceThreshold);

            bool shouldTerminate = false;
            int terminationLane  = -1;

            if (earlyTerminationMask) {
                terminationLane = __ffs(earlyTerminationMask) - 1; // Find first terminating lane
                shouldTerminate = true;
                ray.kill();
            }

            // Step 5: Warp reduction for feature accumulation
            TFeaturesVec accumulatedFeatures = TFeaturesVec::zero();
            float accumulatedHitT            = 0.0f;
            tcnn::vec3 accumulatedNormal     = tcnn::vec3::zero();
            uint32_t accumulatedHitCount     = 0;
            uint32_t accumulatedLastContributor = 0;
            uint32_t depthInitContributor = 0;
            float depthInitCandidate = 0.0f;

            // Only accumulate contributions before (and including) termination point
            bool shouldContribute = validHit && (!shouldTerminate || laneId <= terminationLane);

            if (shouldContribute) {
                // Use precomputed prefix transmittance, excluding current particle
                float prefixTransmittance   = (laneId > 0) ? (localTransmittance / (1.0f - hitAlpha)) : 1.0f;
                float particleTransmittance = ray.transmittance * prefixTransmittance;
                float hitWeight             = hitAlpha * particleTransmittance;

                // Compute weighted contributions
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    accumulatedFeatures[featIdx] = hitFeatures[featIdx] * hitWeight;
                }
                accumulatedHitT     = hitT * hitWeight;
                accumulatedNormal   = hitNormal * hitWeight;
                accumulatedHitCount = (hitWeight > 0.0f) ? 1 : 0;
                accumulatedLastContributor = j + 1;
                if (particleTransmittance > 0.5f) {
                    depthInitContributor = j + 1;
                    depthInitCandidate = hitT;
                }
            }

            // Step 6: Warp-level reduction (tree-based sum)
            for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                for (uint32_t offset = WarpSize / 2; offset > 0; offset >>= 1) {
                    accumulatedFeatures[featIdx] += __shfl_down_sync(WarpMask, accumulatedFeatures[featIdx], offset);
                }
            }

            for (uint32_t offset = WarpSize / 2; offset > 0; offset >>= 1) {
                accumulatedHitT += __shfl_down_sync(WarpMask, accumulatedHitT, offset);
                accumulatedNormal.x += __shfl_down_sync(WarpMask, accumulatedNormal.x, offset);
                accumulatedNormal.y += __shfl_down_sync(WarpMask, accumulatedNormal.y, offset);
                accumulatedNormal.z += __shfl_down_sync(WarpMask, accumulatedNormal.z, offset);
                accumulatedHitCount += __shfl_down_sync(WarpMask, accumulatedHitCount, offset);

                const uint32_t otherLastContributor = __shfl_down_sync(WarpMask, accumulatedLastContributor, offset);
                accumulatedLastContributor = accumulatedLastContributor > otherLastContributor ? accumulatedLastContributor : otherLastContributor;
                const uint32_t otherDepthInitContributor = __shfl_down_sync(WarpMask, depthInitContributor, offset);
                const float otherDepthInitCandidate = __shfl_down_sync(WarpMask, depthInitCandidate, offset);
                if (otherDepthInitContributor > depthInitContributor) {
                    depthInitContributor = otherDepthInitContributor;
                    depthInitCandidate = otherDepthInitCandidate;
                }
            }

            // Step 7: Only lane 0 updates ray state (avoid race conditions)
            if (laneId == 0) {
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    ray.features[featIdx] += accumulatedFeatures[featIdx];
                }
                ray.hitT += accumulatedHitT;
                ray.normal += accumulatedNormal;
                ray.countHit(accumulatedHitCount);
                ray.gggsLastContributor = ray.gggsLastContributor > accumulatedLastContributor ? ray.gggsLastContributor : accumulatedLastContributor;
                if (depthInitContributor > 0) {
                    ray.depthInitT = depthInitCandidate;
                }
            }

            // Step 8: Update ray transmittance
            ray.transmittance = newTransmittance;

            // Break on early termination
            if (shouldTerminate) {
                break;
            }
        }

        if (params.depthMode == threedgut::RenderParameters::GGGSMedianDepth) {
            initializeGGGSMedianDepth(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr);
            resolveGGGSMedianDepth(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr);
        }
    }

    template <typename TRay>
    static inline __device__ float gggsDepthBackwardGamma(const threedgut::RenderParameters& params,
                                                          TRay& ray,
                                                          Particles& particles,
                                                          const tcnn::uvec2& tileParticleRangeIndices,
                                                          const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                          const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                          const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr) {
        if (params.depthMode != threedgut::RenderParameters::GGGSMedianDepth ||
            ray.hitTBackward <= ray.tMinMax.x ||
            ray.hitTBackward >= ray.tMinMax.y ||
            ray.hitTGradient == 0.0f) {
            return 0.0f;
        }

        constexpr uint32_t AllPixelCandidates = 0;
        const float dLogTDt = gggsLogTransmittanceDepthDerivative(ray, params, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr, AllPixelCandidates, ray.hitTBackward);
        if (fabsf(dLogTDt) < 1.0e-7f) {
            return 0.0f;
        }

        return -ray.hitTGradient / dLogTDt;
    }

    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(const threedgut::RenderParameters& params,
                                                        TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                        const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                                        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;
        __shared__ PrefetchedRawParticleData prefetchedRawParticlesData[GUTParameters::Tiling::BlockSize];
        const float gggsDepthGamma = gggsDepthBackwardGamma(params, ray, particles, tileParticleRangeIndices, sortedTileParticleIdxPtr, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr);
        const float expectedDepthGradient = params.depthMode == threedgut::RenderParameters::GGGSMedianDepth ? 0.0f : ray.hitTGradient;

        if (__syncthreads_or(gggsDepthGamma != 0.0f)) {
            uint32_t depthNumParticlesToProcess = tileNumParticlesToProcess;
            for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, depthNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
                if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                    const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                    if (particleIdx != GUTParameters::InvalidParticleIdx) {
                        prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                        prefetchedRawParticlesData[tileThreadIdx].idx = particleIdx;
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                    }
                } else {
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
                __syncthreads();

                for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, depthNumParticlesToProcess); j++) {
                    const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                    if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                        break;
                    }
                    if (!gggsPixelCandidate(params, ray, particleData.idx, particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr)) {
                        continue;
                    }

                    DensityRawParameters densityRawParametersGrad;
                    densityRawParametersGrad.density    = 0.0f;
                    densityRawParametersGrad.position   = make_float3(0.0f);
                    densityRawParametersGrad.quaternion = make_float4(0.0f);
                    densityRawParametersGrad.scale      = make_float3(0.0f);

                    if (gggsDepthGamma != 0.0f) {
                        particles.processGGGSDepthBwd(ray.origin,
                                                      ray.direction,
                                                      particleData.densityParameters,
                                                      &densityRawParametersGrad,
                                                      ray.tMinMax.x,
                                                      ray.tMinMax.y,
                                                      ray.hitTBackward,
                                                      ray.tMinMax.x,
                                                      gggsDepthGamma);
                    }
                    particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
                }
                __syncthreads();
            }
        }

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                    if constexpr (Params::PerRayParticleFeatures) {
                        prefetchedRawParticlesData[tileThreadIdx].features = TFeaturesVec::zero();
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].features = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                    }
                    prefetchedRawParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                    break;
                }

                const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();
                    break;
                }

                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;
                densityRawParametersGrad.position   = make_float3(0.0f);
                densityRawParametersGrad.quaternion = make_float4(0.0f);
                densityRawParametersGrad.scale      = make_float3(0.0f);

                TFeaturesVec featuresGrad = TFeaturesVec::zero();

                if (ray.isAlive()) {
                    particles.processHitBwd<Params::PerRayParticleFeatures>(
                        ray.origin,
                        ray.direction,
                        particleData.idx,
                        particleData.densityParameters,
                        &densityRawParametersGrad,
                        particleData.features,
                        &featuresGrad,
                        ray.transmittance,
                        ray.transmittanceBackward,
                        ray.transmittanceGradient,
                        ray.features,
                        ray.featuresBackward,
                        ray.featuresGradient,
                        ray.hitT,
                        ray.hitTBackward,
                        expectedDepthGradient);
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                if constexpr (!Params::PerRayParticleFeatures) {
                    particles.processHitBwdUpdateFeaturesGradient(particleData.idx, featuresGrad,
                                                                  particleFeaturesGradientBuffer, tileThreadIdx);
                }
                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};

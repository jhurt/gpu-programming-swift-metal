// Copyright 2026 Jason Hurt
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

kernel void metalGrid1D(uint threadPositionInGrid [[thread_position_in_grid]],
                        uint threadsPerGrid [[threads_per_grid]]) {
    os_log_default.log("threadPositionInGrid %u, threadsPerGrid %u", threadPositionInGrid, threadsPerGrid);
}

kernel void metalGrid2D(uint2 threadPositionInGrid [[thread_position_in_grid]],
                        uint2 threadsPerGrid [[threads_per_grid]]) {
    os_log_default.log("threadPositionInGrid (%u, %u), threadsPerGrid (%u, %u)", threadPositionInGrid.x, threadPositionInGrid.y, threadsPerGrid.x, threadsPerGrid.y);
}

kernel void metalGrid3D(uint3 threadPositionInGrid [[thread_position_in_grid]],
                        uint3 threadsPerGrid [[threads_per_grid]]) {
    os_log_default.log("threadPositionInGrid (%u, %u, %u), threadsPerGrid (%u, %u, %u)", threadPositionInGrid.x, threadPositionInGrid.y, threadPositionInGrid.z, threadsPerGrid.x, threadsPerGrid.y, threadsPerGrid.z);
}

kernel void metalThreadgroup1D(uint threadPositionInGrid [[thread_position_in_grid]],
                               uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                               uint threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint threadsPerThreadgroup [[threads_per_threadgroup]],
                               uint threadsPerGrid [[threads_per_grid]]) {
    os_log_default.log("threadPositionInGrid %u, threadIndexInThreadgroup %u, threadgroupPositionInGrid: %u, threadsPerThreadgroup %u, threadsPerGrid %u", threadPositionInGrid, threadIndexInThreadgroup, threadgroupPositionInGrid, threadsPerThreadgroup, threadsPerGrid);
}

kernel void metalThreadgroup2D(uint2 threadPositionInGrid [[thread_position_in_grid]],
                               uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                               uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
                               uint2 threadsPerGrid [[threads_per_grid]]) {
    os_log_default.log("threadPositionInGrid (%u, %u), threadIndexInThreadgroup %u, threadgroupPositionInGrid: (%u %u), threadsPerThreadgroup (%u, %u), threadsPerGrid (%u, %u)", threadPositionInGrid.x, threadPositionInGrid.y, threadIndexInThreadgroup, threadgroupPositionInGrid.x, threadgroupPositionInGrid.y, threadsPerThreadgroup.x, threadsPerThreadgroup.y, threadsPerGrid.x, threadsPerGrid.y);
}

kernel void metalThreadgroup3D(uint3 threadPositionInGrid [[thread_position_in_grid]],
                               uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                               uint3 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint3 threadsPerThreadgroup [[threads_per_threadgroup]],
                               uint3 threadsPerGrid [[threads_per_grid]]) {
    os_log_default.log("threadPositionInGrid (%u, %u, %u), threadIndexInThreadgroup %u, threadgroupPositionInGrid: (%u, %u %u), threadsPerThreadgroup (%u, %u, %u), threadsPerGrid (%u, %u, %u)", threadPositionInGrid.x, threadPositionInGrid.y, threadPositionInGrid.z, threadIndexInThreadgroup, threadgroupPositionInGrid.x, threadgroupPositionInGrid.y, threadgroupPositionInGrid.z, threadsPerThreadgroup.x, threadsPerThreadgroup.y, threadsPerThreadgroup.z, threadsPerGrid.x, threadsPerGrid.y, threadsPerGrid.z);
}

kernel void metalThreadgroupUniformity(uint2 dispatchThreadsPerThreadgroup [[dispatch_threads_per_threadgroup]],
                                       uint2 threadsPerThreadgroup [[threads_per_threadgroup]]) {
    os_log_default.log("dispatchThreadsPerThreadgroup (%u %u), threadsPerThreadgroup (%u, %u)", dispatchThreadsPerThreadgroup.x, dispatchThreadsPerThreadgroup.y, threadsPerThreadgroup.x, threadsPerThreadgroup.y);
}

kernel void metalSimdgroup(uint threadIndexInSimdGroup [[thread_index_in_simdgroup]],
                           uint simdgroupIndexInThreadgroup [[simdgroup_index_in_threadgroup]],
                           uint2 dispatchThreadsPerThreadgroup [[dispatch_threads_per_threadgroup]],
                           uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
                           uint dispatchSimdgroupsPerThreadgroup [[dispatch_simdgroups_per_threadgroup]],
                           uint simdgroupsPerThreadgroup [[simdgroups_per_threadgroup]]) {
    os_log_default.log("simdgroupIndexInThreadgroup %u, threadIndexInSimdGroup %u, dispatchThreadsPerThreadgroup (%u %u), threadsPerThreadgroup (%u, %u), dispatchSimdgroupsPerThreadgroup %u, simdgroupsPerThreadgroup %u", simdgroupIndexInThreadgroup, threadIndexInSimdGroup, dispatchThreadsPerThreadgroup.x, dispatchThreadsPerThreadgroup.y, threadsPerThreadgroup.x, threadsPerThreadgroup.y, dispatchSimdgroupsPerThreadgroup, simdgroupsPerThreadgroup);
}

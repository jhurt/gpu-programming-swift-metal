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

constant uint TILE_WIDTH = 16;
constant uint TILE_HEIGHT = 16;

uint readCell(int2 cellCoordinate, const device uint8_t* grid, uint2 gridSizePixels) {
    if (cellCoordinate.x < 0 || cellCoordinate.x >= int(gridSizePixels.x) ||
        cellCoordinate.y < 0 || cellCoordinate.y >= int(gridSizePixels.y)) {
        return 0;
    }
    
    uint index = cellCoordinate.y * gridSizePixels.x + cellCoordinate.x;
    return grid[index];
}

bool tileHasLife(uint2 tileCoordinate, const device uint8_t* grid, uint2 gridSizePixels, uint2 gridSizeTiles) {
    if (tileCoordinate.x < 0 || tileCoordinate.x >= gridSizeTiles.x ||
        tileCoordinate.y < 0 || tileCoordinate.y >= gridSizeTiles.y) {
        return false;
    }
    
    uint startX = tileCoordinate.x * TILE_WIDTH;
    uint startY = tileCoordinate.y * TILE_HEIGHT;
    for (uint y = 0; y < TILE_HEIGHT; y++) {
        for (uint x = 0; x < TILE_WIDTH; x++) {
            uint index = (startY + y) * gridSizePixels.x + (startX + x);
            if (grid[index] > 0) {
                return true;
            }
        }
    }
    return false;
}

bool atLeastOneNeighboringTileHasLife(uint2 tileCoordinate, const device uint8_t* grid, uint2 gridSizePixels, uint2 gridSizeTiles) {
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint2 neighborTileCoordinate = uint2(int(tileCoordinate.x) + dx, int(tileCoordinate.y) + dy);
            if (tileHasLife(neighborTileCoordinate, grid, gridSizePixels, gridSizeTiles)) {
                return true;
            }
        }
    }
    return false;
}

kernel void classifyTiles(device uint* interestingTileList [[buffer(0)]],
                          device atomic_uint* interestingTileCount [[buffer(1)]],
                          const device uint8_t* currentGrid [[buffer(2)]],
                          constant uint2& gridSizePixels [[buffer(3)]],
                          constant uint2& gridSizeTiles [[buffer(4)]],
                          uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) convert this thread's position to a 2D tile coordinate
    uint tileX = threadPositionInGrid % gridSizeTiles.x;
    uint tileY = threadPositionInGrid / gridSizeTiles.x;
    uint2 tileCoordinate = uint2(tileX, tileY);
    
    // (2) check if this is an interesting tile
    if (atLeastOneNeighboringTileHasLife(tileCoordinate, currentGrid, gridSizePixels, gridSizeTiles)) {
        // (3) add this tile to `interestingTileList` if it is interesting
        uint listIndex = atomic_fetch_add_explicit(interestingTileCount, 1, memory_order_relaxed);
        interestingTileList[listIndex] = threadPositionInGrid;
    }
}

struct ClassifyTilesICBArguments {
    command_buffer indirectCommandBuffer [[id(0)]];
    compute_pipeline_state computePipelineState [[id(1)]];
};

kernel void classifyTilesICB(const device ClassifyTilesICBArguments &args [[buffer(0)]],
                             device uint* interestingTileList [[buffer(1)]],
                             device atomic_uint* interestingTileCount [[buffer(2)]],
                             const device uint8_t* currentGrid [[buffer(3)]],
                             device uint8_t* nextGrid [[buffer(4)]],
                             const device uint2& gridSizePixels [[buffer(5)]],
                             const device uint2& gridSizeTiles [[buffer(6)]],
                             device atomic_uint* completedThreadgroupCount [[buffer(7)]],
                             uint threadPositionInGrid [[thread_position_in_grid]],
                             uint threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                             uint threadsPerThreadgroup [[threads_per_threadgroup]],
                             uint threadgroupsPerGrid [[threadgroups_per_grid]]) {
    // (1) convert this thread's position to a 2D tile coordinate
    uint tileX = threadPositionInGrid % gridSizeTiles.x;
    uint tileY = threadPositionInGrid / gridSizeTiles.x;
    uint2 tileCoordinate = uint2(tileX, tileY);
    
    // (2) check if this is an interesting tile
    if (atLeastOneNeighboringTileHasLife(tileCoordinate, currentGrid, gridSizePixels, gridSizeTiles)) {
        // (3) add this tile to `interestingTileList` if it is interesting
        uint listIndex = atomic_fetch_add_explicit(interestingTileCount, 1, memory_order_relaxed);
        interestingTileList[listIndex] = threadPositionInGrid;
    }
    
    // (4) ensure all threads in this threadgroup have finished writing to `interestingTileList`
    threadgroup_barrier(mem_flags::mem_device);
    
    // (5) the first thread in each threadgroup may end up encoding the dispatch
    if (threadPositionInThreadgroup != 0) {
        return;
    }
    
    // (6) increment the atomic `completedThreadgroupCount`
    uint completedThreadgroups = atomic_fetch_add_explicit(completedThreadgroupCount, 1, memory_order_relaxed) + 1;
    
    // (7) if all threadgroups have completed, this thread will encode the dispatch
    if (completedThreadgroups == threadgroupsPerGrid) {
        // (8) check the interesting tile count, if it's zero then do nothing, all cells have died
        uint tileCount = atomic_load_explicit(interestingTileCount, memory_order_relaxed);
        if (tileCount > 0) {
            // (9) get the compute command associated with the indirect command buffer
            compute_command computeCommand(args.indirectCommandBuffer, 0);
            computeCommand.reset();
            // (10) bind the buffers to the `executeSimulation` kernel
            computeCommand.set_kernel_buffer(interestingTileList, 0);
            computeCommand.set_kernel_buffer(currentGrid, 1);
            computeCommand.set_kernel_buffer(nextGrid, 2);
            computeCommand.set_kernel_buffer(&gridSizePixels, 3);
            computeCommand.set_kernel_buffer(&gridSizeTiles, 4);
            
            // (11) set the `executeSimulation` compute pipeline state on the compute command
            computeCommand.set_compute_pipeline_state(args.computePipelineState);

            // (12) dispatch one threadgroup per interesting tile, tileWidth * tileHeight threads per threadgroup, one thread per cell for the `executeSimulation` kernel
            uint threadgroupsPerGridX = (uint)ceil(sqrt((float)tileCount));
            uint threadgroupsPerGridY = (tileCount + threadgroupsPerGridX - 1) / threadgroupsPerGridX;
            uint3 threadgroupsPerGrid = uint3(threadgroupsPerGridX, threadgroupsPerGridY, 1);
            uint3 threadsPerThreadgroup = uint3(TILE_WIDTH, TILE_HEIGHT, 1);
            computeCommand.concurrent_dispatch_threadgroups(threadgroupsPerGrid, threadsPerThreadgroup);
        }
    }
}

kernel void executeSimulation(device uint* interestingTileList [[buffer(0)]],
                              const device uint8_t* currentGrid [[buffer(1)]],
                              device uint8_t* nextGrid [[buffer(2)]],
                              constant uint2& gridSizePixels [[buffer(3)]],
                              constant uint2& gridSizeTiles [[buffer(4)]],
                              uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                              uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                              uint2 threadgroupsPerGrid [[threadgroups_per_grid]]) {
    // (1) get the 2D tile index for this thread's threadgroup
    uint tileIndex = interestingTileList[threadgroupPositionInGrid.y * threadgroupsPerGrid.x + threadgroupPositionInGrid.x];
    uint2 tileCoordinate = uint2(tileIndex % gridSizeTiles.x, tileIndex / gridSizeTiles.x);
    // (2) get the 2D cell coordinate for this thread based on its position in the threadgroup
    uint2 cellCoordinate = tileCoordinate * uint2(TILE_WIDTH, TILE_HEIGHT) + threadPositionInThreadgroup;
    if (cellCoordinate.x >= gridSizePixels.x || cellCoordinate.y >= gridSizePixels.y) {
        return;
    }
    
    // (3) count the amount of neighboring cells with life
    int liveNeighborCount = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            int2 neighborCoordinate = int2(cellCoordinate) + int2(x, y);
            if (readCell(neighborCoordinate, currentGrid, gridSizePixels) > 0) {
                liveNeighborCount++;
            }
        }
    }
    
    // (4) determine this thread's cell's state based on Conway's Game of Life rules
    uint currentState = readCell(int2(cellCoordinate), currentGrid, gridSizePixels);
    uint nextState = 0;
    if (currentState == 1) {
        if (liveNeighborCount == 2 || liveNeighborCount == 3) {
            nextState = 1;
        }
    } else {
        if (liveNeighborCount == 3) {
            nextState = 1;
        }
    }
    
    // (5) write this thread's cell's next state to `nextGrid`
    uint cellIndex = cellCoordinate.y * gridSizePixels.x + cellCoordinate.x;
    nextGrid[cellIndex] = uint8_t(nextState);
}

kernel void executeSimulation2(device uint* interestingTileList [[buffer(0)]],
                               const device uint8_t* currentGrid [[buffer(1)]],
                               device uint8_t* nextGrid [[buffer(2)]],
                               constant uint2& gridSizePixels [[buffer(3)]],
                               constant uint2& gridSizeTiles [[buffer(4)]],
                               uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                               uint2 threadgroupsPerGrid [[threadgroups_per_grid]]) {
    uint tileIndex = interestingTileList[threadgroupPositionInGrid.y * threadgroupsPerGrid.x + threadgroupPositionInGrid.x];
    uint2 tileCoordinate = uint2(tileIndex % gridSizeTiles.x, tileIndex / gridSizeTiles.x);
    uint2 cellCoordinate = tileCoordinate * uint2(TILE_WIDTH, TILE_HEIGHT) + threadPositionInThreadgroup;
    if (cellCoordinate.x >= gridSizePixels.x || cellCoordinate.y >= gridSizePixels.y) {
        return;
    }
    
    int interestingNeighbors = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            int2 neighborCoordinate = int2(cellCoordinate) + int2(x, y);
            if (readCell(neighborCoordinate, currentGrid, gridSizePixels) > 0) {
                interestingNeighbors++;
            }
        }
    }
    
    uint currentState = readCell(int2(cellCoordinate), currentGrid, gridSizePixels);
    uint nextState = 0;
    if (currentState == 1) {
        // slightly different rules than Conway's
        if (interestingNeighbors == 2 || interestingNeighbors == 3 || interestingNeighbors == 1) {
            nextState = 1;
        }
    } else {
        // dead, maybe spawn
        if (interestingNeighbors == 3) {
            nextState = 1;
        }
    }
    
    uint cellIndex = cellCoordinate.y * gridSizePixels.x + cellCoordinate.x;
    nextGrid[cellIndex] = uint8_t(nextState);
}

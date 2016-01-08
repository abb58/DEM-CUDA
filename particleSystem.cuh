/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

	void copyArrayFromDevice(void *host, const void *device, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    
    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos,
                         float *vel,
                         float *force,
                         float deltaTime,
                         uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     float *sortedForce,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     float *oldForce,
                                     uint   numParticles,
                                     uint   numCells);

    void collide(float *newForce,
                 float *sortedPos,
                 float *sortedVel,
                 float *sortedForce,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);
                 
    void vintegrateSystem(float *vel,
                         float *force,
                         float deltaTime,
                         uint numParticles);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}

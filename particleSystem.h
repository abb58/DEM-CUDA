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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
            FORCE,
        };

        int    getNumParticles() const
        {
            return m_numParticles;
        }

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        
        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        
        float3 getCellSize()
        {
            return m_params.cellSize;
        }
        
        float *getArray(ParticleArray array);
        void   setArray(ParticleArray array, const float *data, int start, int count);
        void update(float deltaTime);
        void reset(ParticleConfig config);
	void dumpParticles(uint start, uint count, uint file_count);
	void writeParticles(FILE* fpout, uint start, uint count, uint file_count);
		
    protected: // methods
        ParticleSystem() {}
        
        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);

    protected: // data
        bool m_bInitialized;
        uint m_numParticles;

        // CPU data
        float *m_hPos;				// particle positions
        float *m_hVel;				// particle velocities
	float *m_hForce;			// particle forces
	int *m_hColor;			// particle forces
		
        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVel;
		float *m_dForce;
		
        float *m_dSortedPos;
        float *m_dSortedVel;
		float *m_dSortedForce;
		
        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__

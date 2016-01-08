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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_hForce(0),
    m_dPos(0),
    m_dVel(0),
    m_dForce(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f/m_gridSize.x;    // what is the physical meaning of the equation

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);  //this seems to belong to cuda coding part

    m_params.spring = 3.4e8;
    m_params.damping = 1.0e6;
    m_params.shear = 7.0e7;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.1f;

    m_params.gravity = make_float3(0.0f, -1.0f, 0.0f);

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];    // particle positions
    m_hVel = new float[m_numParticles*4];   // particle velocity
    m_hForce = new float[m_numParticles*4];     //particle forces
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
    memset(m_hForce, 0, m_numParticles*4*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    allocateArray((void **)&m_dPos, memSize);
    allocateArray((void **)&m_dVel, memSize);
    allocateArray((void **)&m_dForce, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
    allocateArray((void **)&m_dSortedForce, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));  // what is hush value?
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));  // particle index

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hForce;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dPos);
    freeArray(m_dVel);
    freeArray(m_dForce);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
	freeArray(m_dSortedForce);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    // update constants
    setParameters(&m_params);

    // integrate
    integrateSystem(
        m_dPos,
        m_dVel,
        m_dForce,
        deltaTime,
        m_numParticles);

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dSortedForce,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPos,
        m_dVel,
        m_dForce,
        m_numParticles,
        m_numGridCells);

    // process collisions
    collide(
        m_dForce,
        m_dSortedPos,
        m_dSortedVel,
        m_dSortedForce,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);

    // integrate velocity
    vintegrateSystem(
        m_dVel,
        m_dForce,
        deltaTime,
        m_numParticles);

}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;

        case FORCE:
            hdata = m_hForce;
            ddata = m_dForce;
            break;
    }

    copyArrayFromDevice(hdata, ddata, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
  assert(m_bInitialized);

  switch (array){
  default:
  case POSITION:
    copyArrayToDevice(m_dPos, data, start*4*sizeof(float), count*4*sizeof(float));
    break;

  case VELOCITY:
    copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
      break;

  case FORCE:
    copyArrayToDevice(m_dForce, data, start*4*sizeof(float), count*4*sizeof(float));
    break;
  }
}


// Set the positions, velocities and forces for the CPU data varaibles
void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);
    uint i=0;
    // z=5, y=40; x=20

    // bottom plane
    for (uint z=0; z<size[2]; z++){
      for (uint x=0; x<size[0]; x++) {
	i = (z*size[1]*size[0]) + x;
	m_hPos[i*4+0] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;  //frand is a random value (0,1)
	m_hPos[i*4+1] = 1.0;
	m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	m_hPos[i*4+3] = 1.0f;

	m_hVel[i*4]   = 0.0f;
	m_hVel[i*4+1] = 0.0f;
	m_hVel[i*4+2] = 0.0f;
	m_hVel[i*4+3] = 0.0f;

	m_hForce[i*4] = 0.0f;
	m_hForce[i*4+1] = 0.0f;
	m_hForce[i*4+2] = 0.0f;
	m_hForce[i*4+3] = 0.0f;
      }
    }


    // top plane
    for (uint z=0; z<size[2]; z++){
      for (uint x=0; x<size[0]; x++) {
	i = ((size[1]-1)*size[0]) + (z*size[1]*size[0]) + x;

	m_hPos[i*4+0] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;  //frand is a random value (0,1)
	m_hPos[i*4+1] = size[1];
	m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	m_hPos[i*4+3] = 1.0f;

	m_hVel[i*4]   = 0.0f;
	m_hVel[i*4+1] = 0.0f;
	m_hVel[i*4+2] = 0.0f;
	m_hVel[i*4+3] = 0.0f;

	m_hForce[i*4] = 0.0f;
	m_hForce[i*4+1] = 0.0f;
	m_hForce[i*4+2] = 0.0f;
	m_hForce[i*4+3] = 0.0f;
      }
    }

    // middle layer
    for (uint z=0; z<size[2]; z++){
      for (uint y=1; y<size[1]-1; y++) {
	for (uint x=0; x<size[0]; x++) {
	  i = (z*size[1]*size[0]) + (y*size[0]) + x;

	  m_hPos[i*4]   = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;  //frand is a random value (0,1)
	  m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	  m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	  m_hPos[i*4+3] = 1.0f;

	  m_hVel[i*4] = 0.0f;
	  m_hVel[i*4+1] = 0.0f;
	  m_hVel[i*4+2] = 0.0f;
	  m_hVel[i*4+3] = 0.0f;

	  m_hForce[i*4] = 0.0f;
	  m_hForce[i*4+1] = 0.0f;
	  m_hForce[i*4+2] = 0.0f;
	  m_hForce[i*4+3] = 0.0f;

	}
      }
    }

    /*
    for (uint z=0; z<size[2]; z++){
      for (uint y=0; y<size[1]; y++) {
	for (uint x=0; x<size[0]; x++) {
	  if (i < 110){
	    m_hPos[i*4]   = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;  //frand is a random value (0,1)
	    m_hPos[i*4+1] = 1.0;
	    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	    m_hPos[i*4+3] = 1.0f;

	    m_hVel[i*4] = 0.0f;
	    m_hVel[i*4+1] = 0.0f;
	    m_hVel[i*4+2] = 0.0f;
	    m_hVel[i*4+3] = 0.0f;

	    m_hForce[i*4] = 0.0f;
	    m_hForce[i*4+1] = 0.0f;
	    m_hForce[i*4+2] = 0.0f;
	    m_hForce[i*4+3] = 0.0f;
	  }

	  else if (i < 220){
	    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;  //frand is a random value (0,1)
	    m_hPos[i*4+1] = 39.0;
	    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	    m_hPos[i*4+3] = 1.0f;

	    m_hVel[i*4] = 0.0f;
	    m_hVel[i*4+1] = 0.0f;
	    m_hVel[i*4+2] = 0.0f;
	    m_hVel[i*4+3] = 0.0f;

	    m_hForce[i*4] = 0.0f;
	    m_hForce[i*4+1] = 0.0f;
	    m_hForce[i*4+2] = 0.0f;
	    m_hForce[i*4+3] = 0.0f;
	  }

	  else if (i < 1820 && i>=220){
	    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;  //frand is a random value (0,1)
	    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter+ (spacing * 10.0) ;// start from y=10
	    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
	    m_hPos[i*4+3] = 1.0f;

	    m_hVel[i*4] = 0.0f;
	    m_hVel[i*4+1] = 0.0f;
	    m_hVel[i*4+2] = 0.0f;
	    m_hVel[i*4+3] = 0.0f;

	    m_hForce[i*4] = 0.0f;
	    m_hForce[i*4+1] = 0.0f;
	    m_hForce[i*4+2] = 0.0f;
	    m_hForce[i*4+3] = 0.0f;
	  }
	  else break;
	  i++;
	  std::cout<<i<<std::endl;
	}
      }
    }
    */
}

void
ParticleSystem::dumpParticles(uint start, uint count, uint file_count)
{
    char outfile[256];
    FILE *fpout;

    copyArrayFromDevice(m_hPos,   m_dPos,   sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel,   m_dVel,   sizeof(float)*4*count);
    copyArrayFromDevice(m_hForce, m_dForce, sizeof(float)*4*count);

    sprintf(outfile, "file%04d", file_count);
    if ((fpout = fopen(outfile, "w")) == NULL) {
      printf("Cannot Open File\n");
      exit(1);
    }

    for (uint i=start; i<start+count; i++){
      fprintf(fpout, "%.4f %.4f %.4f %.4f %.4f %.4f 0.0 0.0 0.0 %.4f 0.0 5\n", m_hPos[i*4+0]*64.0f, m_hPos[i*4+1]*64.0f, m_hPos[i*4+2]*64.0f, m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hPos[i*4+3]);
    }

    fclose(fpout);
}

void
ParticleSystem::reset(ParticleConfig config)
{
    float jitter = m_params.particleRadius*0.01f;
    uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
    uint gridSize[3];
    //gridSize[0] = 20;
    //gridSize[1] = 40;
    //gridSize[2] = 5;
    gridSize[0] = gridSize[1] = gridSize[2] = s;

    initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
    setArray(FORCE, m_hForce, 0, m_numParticles);
}

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

/*
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particleSystem.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   16384

const uint width = 640, height = 480;

uint numParticles = 0;
uint3 gridSize;
int numIterations = 0; // run until exit

// simulation parameters
float timestep = 0.000007f;
int iterations = 1;

ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

float modelView[16];

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;

const char *sSDKsample = "CUDA Particle Dynamics Simulation";

extern "C" void cudaInit(int argc, char **argv);

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize)
{
    psystem = new ParticleSystem(numParticles, gridSize);
    psystem->reset(ParticleSystem::CONFIG_GRID);

    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
}

void runBenchmark(int iterations, char *exec_path)
{
  int file_count=0, iterationsPerFrame = (int)(1.0/(30.0*timestep));
        
  printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
  //abb58: 1. what are you trying to sync???
  cudaDeviceSynchronize();
  sdkStartTimer(&timer);

  for (int i = 0; i < iterations; ++i){
    psystem->update(timestep);
        
    if (i % iterationsPerFrame == 0) {
      psystem->dumpParticles(0, numParticles, file_count);
      file_count++;
    }
  }

  cudaDeviceSynchronize();
  sdkStopTimer(&timer);
  float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)iterations);
}

inline float frand()
{
    return rand() / (float) RAND_MAX;  // value between 0 to 1
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{

    printf("%s Starting...\n\n", sSDKsample);

    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;
    numIterations = 0;

    gridSize.x = gridSize.y = gridSize.z = gridDim;

    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    printf("particles: %d\n", numParticles);

    cudaInit(argc, argv);

    initParticleSystem(numParticles, gridSize);

    printf("%e\n", timestep);
    psystem->dumpParticles(0, numParticles-1, 0);

    if (numIterations <= 0)
    {
    	numIterations = (int)(2.0/timestep);
    }
    std::cout << "1. I am here \n";
    runBenchmark(numIterations, argv[0]);
    std::cout << "2. I am here \n";

    if (psystem){
      delete psystem;
      cleanup();
    }

    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}


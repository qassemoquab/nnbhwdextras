
__global__ void SCUclipWeightsKernel(float* wdataptr, float normbound, int str0)
{
   /* blockIdx.x  = [ 0, op    ] ()
      threadIdx.x = [ 0, 31    ] ()
   */

   wdataptr += blockIdx.x*str0;

   volatile __shared__ float sqrsums[32];
   int i;
   float sqrsum=0;
   float current;
   const int numel=str0;
   for(i=threadIdx.x; i<numel; i+=blockDim.x)
   {
      current=wdataptr[i];
      sqrsum+=current*current;
   }

   sqrsums[threadIdx.x]=sqrsum;
   
   // NVCC : Y U NO __SHFL ?
   if (threadIdx.x < 16)
   {
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 16];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 8];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 4];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 2];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 1];
      sqrsums[threadIdx.x + 1] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 2] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 4] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 8] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 16] = sqrsums[threadIdx.x];
   }

   sqrsum=sqrsums[threadIdx.x];   


   if(sqrsum>normbound*normbound)
   {
      float scale = normbound/sqrt(sqrsum); 
      for(i=threadIdx.x; i<numel; i+=blockDim.x)
      {
         wdataptr[i] *= scale;
         //wdataptr[i] =0; // for testing...
      }
   }
}





static int BHWDWeightOps_clipWeights(lua_State *L)
{
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  float normbound = luaL_optnumber(L, 2, 1);

  int op = weight->size[0];
  int str0 = weight->stride[0];

  float* wdata=THCudaTensor_data(weight);

  dim3 blocks(op);
  dim3 threads(32);
  
  SCUclipWeightsKernel <<<blocks, threads>>>(wdata, normbound, str0);

  return 1;
}


__global__ void SCUcenterWeightsKernel(float* wdataptr, int str0)
{
   /* blockIdx.x  = [ 0, op    ] ()
      threadIdx.x = [ 0, 31    ] ()
   */

   wdataptr += blockIdx.x*str0;

   volatile __shared__ float sqrsums[32];
   int i;
   float sum=0;
   float current;
   const int numel=str0;
   for(i=threadIdx.x; i<numel; i+=blockDim.x)
   {
      current=wdataptr[i];
      sum+=current;
   }

   sqrsums[threadIdx.x]=sum;
   
   // NVCC : Y U NO __SHFL ?
   if (threadIdx.x < 16)
   {
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 16];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 8];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 4];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 2];
      sqrsums[threadIdx.x] += sqrsums[threadIdx.x + 1];
      sqrsums[threadIdx.x + 1] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 2] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 4] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 8] = sqrsums[threadIdx.x];
      sqrsums[threadIdx.x + 16] = sqrsums[threadIdx.x];
   }

   sum=sqrsums[threadIdx.x]/float(numel);   

   for(i=threadIdx.x; i<numel; i+=blockDim.x)
   {
      wdataptr[i] -= sum;
      //wdataptr[i] =0; // for testing...
   }

}





static int BHWDWeightOps_centerWeights(lua_State *L)
{
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");

  int op = weight->size[0];
  int str0 = weight->stride[0];

  float* wdata=THCudaTensor_data(weight);

  dim3 blocks(op);
  dim3 threads(32);
  
  SCUcenterWeightsKernel <<<blocks, threads>>>(wdata, str0);

  return 1;
}



__global__ void SCUcenterWeightMapsKernel(float* wdataptr, int str0, int str2)
{
   /* each thread does one input map */
   /* blockIdx.x  = [ 0, op    ] ()
      threadIdx.x = [ 0, 31    ] ()
   */

   wdataptr += blockIdx.x*str0;

   volatile __shared__ float sqrsums[32];
   int i, pixidx;
   float sum=0;
   float current;
   const int numel=str0;
   for(i=threadIdx.x; i<str2; i+=blockDim.x)
   {
      sum=0;
      for(pixidx=0; pixidx<numel; pixidx++)
      {
         current=wdataptr[i+pixidx*str2];
         sum+=current;
      }
   
      sum /= float(numel);

      for(pixidx=0; pixidx<numel; pixidx++)
      {
         wdataptr[i+pixidx*str2] -= sum;
      }
   }

}

static int BHWDWeightOps_centerWeightMaps(lua_State *L)
{
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");

  int op = weight->size[0];
  int str0 = weight->stride[0];
  int str2 = weight->stride[2];

  float* wdata=THCudaTensor_data(weight);

  dim3 blocks(op);
  dim3 threads(32);
  
  SCUcenterWeightMapsKernel <<<blocks, threads>>>(wdata, str0, str2);

  return 1;
}




static const struct luaL_Reg BHWDWeightOps__ [] = {
	{"BHWDWeightOps_clipWeights", BHWDWeightOps_clipWeights},
	{"BHWDWeightOps_centerWeights", BHWDWeightOps_centerWeights},
	{"BHWDWeightOps_centerWeightMaps", BHWDWeightOps_centerWeightMaps},
	{NULL, NULL}
};

static void BHWDWeightOps_init(lua_State *L)
{
	luaT_pushmetatable(L, "torch.CudaTensor");
	luaT_registeratname(L, BHWDWeightOps__, "nn");
	lua_pop(L,1);
}

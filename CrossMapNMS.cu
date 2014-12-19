__global__ void CrossMapNMS_kernel(float *ptrinput, float *ptroutput, const int str0, const int str2, const int nPlanes)
{
   // blockIdx.x  = [0, isize1*isize2]
   // blockIdx.y  = [0, bs]
   // threadIdx.x = [0, 31] 
	// each block does NMS on a pixel from an image in the batch
	ptrinput    += blockIdx.y * str0 + blockIdx.x * str2;
	ptroutput   += blockIdx.y * str0 + blockIdx.x * str2;

	// nPlanes better be a multiple of 32 for coalesced reads
   int i;
   float currentValue = -2e38;
   float valueBuf;
   int currentIndex=-1;
   int indexBuf=-1;

   // max within thread :
   for(i=threadIdx.x; i<nPlanes; i+=blockDim.x)
   {
      valueBuf = ptrinput[i];
      if(valueBuf>currentValue)
      {
         currentValue=valueBuf;
         currentIndex=i;
      }
   }

   // reduce across warp :
   for (int offset = 16; offset > 0; offset /= 2) 
   {
      valueBuf = __shfl_down(currentValue, offset);
      indexBuf = __shfl_down(currentIndex, offset);
      if(valueBuf>currentValue)
      {
         currentValue=valueBuf;
         currentIndex=indexBuf;
      }
   }

   currentValue = __shfl(currentValue, 0);
   currentIndex = __shfl(currentIndex, 0);

   // write output :
   for(i=threadIdx.x; i<nPlanes; i+=blockDim.x)
   {
      valueBuf=0;
      if(currentIndex==i)
      {
         valueBuf=currentValue;
      }
      ptroutput[i]=valueBuf;
   }
}

__global__ void CrossMapNMS_backwardKernel(float *ptrinput, float *ptrgradoutput, float *ptrgradinput, const int str0, const int str2, const int gstr0, const int gstr2, const int nPlanes)
{
   // blockIdx.x  = [0, isize1*isize2]
   // blockIdx.y  = [0, bs]
   // threadIdx.x = [0, 31] 
	// each block does NMS on a pixel from an image in the batch
	ptrinput    += blockIdx.y * str0 + blockIdx.x * str2;
	ptrgradinput   += blockIdx.y * gstr0 + blockIdx.x * gstr2;
	ptrgradoutput   += blockIdx.y * gstr0 + blockIdx.x * gstr2;

	// nPlanes better be a multiple of 32 for coalesced reads
   int i;
   float currentValue = -2e38;
   float valueBuf;
   int currentIndex=-1;
   int indexBuf=-1;

   // max within thread :
   for(i=threadIdx.x; i<nPlanes; i+=blockDim.x)
   {
      valueBuf = ptrinput[i];
      if(valueBuf>currentValue)
      {
         currentValue=valueBuf;
         currentIndex=i;
      }
   }

   // reduce across warp :
   for (int offset = 16; offset > 0; offset /= 2) 
   {
      valueBuf = __shfl_down(currentValue, offset);
      indexBuf = __shfl_down(currentIndex, offset);
      if(valueBuf>currentValue)
      {
         currentValue=valueBuf;
         currentIndex=indexBuf;
      }
   }

   currentValue = __shfl(currentValue, 0);
   currentIndex = __shfl(currentIndex, 0);

   // write output :
   for(i=threadIdx.x; i<nPlanes; i+=blockDim.x)
   {
      valueBuf=0;
      if(currentIndex==i)
      {
         valueBuf=ptrgradoutput[i];
      }
      ptrgradinput[i]=valueBuf;
   }
}


static int cunxn_CrossMapNMS_updateOutput(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

	//luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

	// input should be contiguous already but... well.
	input = THCudaTensor_newContiguous(input);

	long bs     = input->size[0];
	long str0   = input->stride[0];
	long isize1 = input->size[1];
	long isize2 = input->size[2];
	long str2   = input->stride[2];
	long nPlanes = input->size[3];

	THCudaTensor_resizeAs(output, input);


	float* ptroutput  = THCudaTensor_data(output);
	float* ptrinput   = THCudaTensor_data(input);


	// cuda blocks & threads:
	dim3 blocks (isize1*isize2, bs);
	dim3 threads (32);

	CrossMapNMS_kernel <<<blocks, threads>>> (ptrinput, ptroutput, str0, str2, nPlanes);



	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in CrossMapNMS_kernel: %s\n", cudaGetErrorString(err));
		THError("aborting");
	}


	// final cut:
	THCudaTensor_free(input); 
	//THCudaTensor_select(output, NULL, dimension, 0);

	return 1;
}





static int cunxn_CrossMapNMS_updateGradInput(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

	long bs     = input->size[0];
	long str0   = input->stride[0];
	long gstr0   = gradOutput->stride[0];
	long isize1 = input->size[1];
	long isize2 = input->size[2];
	long str2   = input->stride[2];
	long gstr2   = gradOutput->stride[2];
	long nPlanes = input->size[3];

	THCudaTensor_resizeAs(gradInput, gradOutput);

	float* ptrgradoutput  = THCudaTensor_data(gradOutput);
	float* ptrgradinput   = THCudaTensor_data(gradInput);
	float* ptrinput   = THCudaTensor_data(input);

	// cuda blocks & threads:
	dim3 blocks (isize1*isize2, bs);
	dim3 threads (32);

	CrossMapNMS_backwardKernel <<<blocks, threads>>> (ptrinput, ptrgradoutput, ptrgradinput, str0, str2, gstr0, gstr2, nPlanes);


	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in CrossMapNMS_backwardKernel: %s\n", cudaGetErrorString(err));
		THError("aborting");
	}
	return 1;
}



static const struct luaL_Reg cunxn_CrossMapNMS__ [] = {
	{"CrossMapNMS_updateOutput", cunxn_CrossMapNMS_updateOutput},
	{"CrossMapNMS_updateGradInput", cunxn_CrossMapNMS_updateGradInput},
	{NULL, NULL}
};

static void cunxn_CrossMapNMS_init(lua_State *L)
{
	luaT_pushmetatable(L, "torch.CudaTensor");
	luaT_registeratname(L, cunxn_CrossMapNMS__, "nn");
	lua_pop(L,1);
}

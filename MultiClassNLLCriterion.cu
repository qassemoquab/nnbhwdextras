
__global__ void MultiClassNLLCriterion_forwardKernel(float* input, float* target, float* tmp, int n)
{
   /* self.output=self.tmp:cmul(target):mul(-1):exp():add(1):log():sum() */
   /* sum() is kept for later */

   for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
         tmp[i]=log(1+exp(input[i]*target[i]*(-1.0f)));
      }
}

__global__ void MultiClassNLLCriterion_backwardKernel(float* input, float* target, float* gradInput, int n)
{
   /* self.gradInput:map2(input, target, function(x, inp, tgt)  return 1/(1+math.exp(-1*inp*tgt))*(-1*tgt)*math.exp(-1*inp*tgt) end) */
   float inp;
   float tgt;

   for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
         inp = input[i];
         tgt = target[i];
         gradInput[i]=1.0f/(1.0f+exp(-1.0f*inp*tgt))*(-1.0f*tgt)*exp(-1.0f*inp*tgt);
      }
}


static int cunxn_MultiClassNLLCriterion_updateOutput(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *target = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
   THCudaTensor *tmp = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "tmp", "torch.CudaTensor");
   long size = THCudaTensor_nElement(input);

   THCudaTensor_resizeAs(tmp, input);


   long nthreads = 32;
   long nblocks = (size+nthreads-1) / nthreads;
   dim3 blocks(nblocks);
   dim3 threads(nthreads);

   MultiClassNLLCriterion_forwardKernel<<<blocks, threads>>>(THCudaTensor_data(input), THCudaTensor_data(target), THCudaTensor_data(tmp), size);

   float output = THCudaTensor_sumall(tmp);

   lua_pushnumber(L, output);

   return 1;
}

static int cunxn_MultiClassNLLCriterion_updateGradInput(lua_State *L)
{
   THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *target = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

   long size = THCudaTensor_nElement(input);
   THCudaTensor_resizeAs(gradInput, input);

   long nthreads = 32;
   long nblocks = (size+nthreads-1) / nthreads;
   dim3 blocks(nblocks);
   dim3 threads(nthreads);

   MultiClassNLLCriterion_backwardKernel<<<blocks, threads>>>(THCudaTensor_data(input), THCudaTensor_data(target), THCudaTensor_data(gradInput), size);

   return 0;
}

static const struct luaL_Reg cunxn_MultiClassNLLCriterion__ [] = {
  {"MultiClassNLLCriterion_updateOutput", cunxn_MultiClassNLLCriterion_updateOutput},
  {"MultiClassNLLCriterion_updateGradInput", cunxn_MultiClassNLLCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunxn_MultiClassNLLCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_MultiClassNLLCriterion__, "nn");
  lua_pop(L,1);
}

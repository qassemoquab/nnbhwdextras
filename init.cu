#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "SpatialGlobalMaxPooling.cu"
#include "CrossMapNormalization.cu"
#include "Dropmap.cu"
#include "MultiClassNLLCriterion.cu"
#include "CrossMapNMS.cu"
#include "BHWDWeightOps.cu"


LUA_EXTERNC DLL_EXPORT int luaopen_libnnbhwdextras(lua_State *L);

int luaopen_libnnbhwdextras(lua_State *L)
{
  lua_newtable(L);

  cunxn_SpatialGlobalMaxPooling_init(L);
  cunxn_CrossMapNormalization_init(L);
  cunxn_Dropmap_init(L);
  cunxn_MultiClassNLLCriterion_init(L);
  cunxn_CrossMapNMS_init(L);
  BHWDWeightOps_init(L);

  return 1;
}

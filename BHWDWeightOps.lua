function nn.SpatialConvolutionBHWD:clipWeights(normbound)
   self.weight.nn.BHWDWeightOps_clipWeights(self, normbound)
end

function nn.SpatialConvolutionBHWD:centerWeights()
   self.weight.nn.BHWDWeightOps_centerWeights(self)
end

function nn.SpatialConvolutionBHWD:centerWeightMaps()
   self.weight.nn.BHWDWeightOps_centerWeightMaps(self)
end



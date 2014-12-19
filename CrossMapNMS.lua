local CrossMapNMS, parent = torch.class('nn.CrossMapNMS', 'nn.Module')

local help_str = 
[[This is the Cross-map non-max suppression module.
It performs a non-max suppression on each input position and returns an output of the same size.

Usage : m = nxn.CrossMapNMS()

It only works in BATCH MODE (4D) :
- with the following input layout : (batch, y, x, channels).
- channels are the contiguous dimension.
- a single image must be a (1, y, x, channels) tensor.

The module doesn't require fixed-size inputs.]]

function CrossMapNMS:__init()
   parent.__init(self)
   self.gpucompatible = true
end


function CrossMapNMS:updateOutput(input)
   input.nn.CrossMapNMS_updateOutput(self, input)
   return self.output
end

function CrossMapNMS:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.CrossMapNMS_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end


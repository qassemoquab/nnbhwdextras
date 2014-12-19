local MultiClassNLLCriterion, parent = torch.class('nn.MultiClassNLLCriterion', 'nn.Criterion')

function MultiClassNLLCriterion:__init()
   parent.__init(self)
   self.tmp=torch.Tensor()
   print('CAREFUL ! ALL TARGETS MUST BE EITHER -1 OR 1 !')
end

function MultiClassNLLCriterion:updateOutput(input, target)
   if not input:isSameSizeAs(target) then error('CHECK THEM SIZES !') end
   if input:type()=='torch.CudaTensor' then
      self.output=input.nn.MultiClassNLLCriterion_updateOutput(self, input, target)
   else
      self.tmp:resizeAs(input)
      self.tmp:copy(input)
      
      self.output=self.tmp:cmul(target):mul(-1):exp():add(1):log():sum()
   end
   return self.output
end


function MultiClassNLLCriterion:updateGradInput(input, target)
   if not input:isSameSizeAs(target) then error('CHECK THEM SIZES !') end
   if input:type()=='torch.CudaTensor' then
      input.nn.MultiClassNLLCriterion_updateGradInput(self, input, target)
   else
      self.gradInput:resizeAs(input)
      self.gradInput:zero()

      self.gradInput:map2(input, target, function(x, inp, tgt)  return 1/(1+math.exp(-1*inp*tgt))*(-1*tgt)*math.exp(-1*inp*tgt) end)
   end
   return self.gradInput
end

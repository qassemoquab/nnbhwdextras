require "torch"
require 'cunn'
require "libnnbhwdextras"


local nnbhwdextras = {}

nnbhwdextras.SpatialGlobalMaxPooling = require 'nnbhwdextras.SpatialGlobalMaxPooling'
nnbhwdextras.CrossMapNormalization = require 'nnbhwdextras.CrossMapNormalization'
nnbhwdextras.MultiClassNLLCriterion = require 'nnbhwdextras.MultiClassNLLCriterion'
nnbhwdextras.Dropmap = require 'nnbhwdextras.Dropmap'
nnbhwdextras.CrossMapNMS = require 'nnbhwdextras.CrossMapNMS'
nnbhwdextras.BHWDWeightOps = require 'nnbhwdextras.BHWDWeightOps'

return nnbhwdextras

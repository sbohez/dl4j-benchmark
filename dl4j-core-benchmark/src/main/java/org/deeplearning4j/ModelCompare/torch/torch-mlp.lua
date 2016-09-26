-- Torch7 MLP
--
-- Reference Code:
--      https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
--      https://github.com/eladhoffer/ImageNet-Training/blob/master/Main.lua
--      https://github.com/soumith/imagenet-multiGPU.torch
-- Reference Xaviar: https://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua#L19

-- Note kept global variables to make it easeir to copy and debug in interactive shell


require 'nn'

mlp = {}

------------------------------------------------------------
function w_init_xavier(fan_in, fan_out)
    return math.sqrt(2/(fan_in + fan_out))
end

function w_init_xavier_caffe(fan_in, fan_out)
    return math.sqrt(1/fan_in)
end

function w_init_xavier_dl4j(fan_in, fan_out)
    return math.sqrt(1/(fan_in + fan_out))
end

function updateParams(model)
    for i=1, #model.modules do
        method = w_init_xavier_dl4j
        local m = model.modules[i]
        if m.__typename == 'nn.SpatialConvolutionMM' then
            m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
            m.bias = nil
            m.gradBias = nil
        elseif m.__typename == 'nn.Linear' then
            m:reset(method(m.weight:size(2), m.weight:size(1)))
            m.bias:zero()
        end
    end
    return model
end

function mlp.build_model(numInputs, numClasses)
    local numHidden = 1000

    local model = nn.Sequential()
    model:add(nn.Reshape(numInputs))
    model:add(nn.Linear(numInputs,numHidden))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(numHidden,numClasses))
    return updateParams(model)
end

function mlp.define_loss()
    return nn.CrossEntropyCriterion()
end


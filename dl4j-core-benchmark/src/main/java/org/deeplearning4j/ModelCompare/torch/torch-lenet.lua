-- Torch7 Lenet
--
-- Reference: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua

-- Lessons learned:
--    requires batch and numExamples to be divisable without remainder
--    harder to debug and research than python
--    More steps to apply gpu vs caffe and dl4j

-- Note kept global variables to make it easeir to copy and debug in interactive shell

require 'nn'

lenet = {}

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

function lenet.build_model(numInputs, numClasses)
    local ccn1depth = 20
    local ccn2depth = 50
    local ffn1depth = 500
    local model = nn.Sequential()
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(numInputs, ccn1depth, 5, 5))
    model:add(nn.Identity())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(ccn1depth, ccn2depth, 5, 5))
    model:add(nn.Identity())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 3 : standard 2-layer MLP:
    model:add(nn.Reshape(ccn2depth*4*4))
    model:add(nn.Linear(ccn2depth*4*4, ffn1depth))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(ffn1depth, numClasses))
    return updateParams(model)
end

function lenet.define_loss()
    return nn.CrossEntropyCriterion()
end



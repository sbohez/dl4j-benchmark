--
-- Util for org.org
--
require 'torch'
require 'nn'
require 'dl4j-core-benchmark/src/main/resources/torch-data/dataset-mnist'
require 'optim'

util = {}

local opt = {
    cudnn_fastest = true,
    cudnn_deterministic = false,
    cudnn_benchmark = true,
    flatten = true,
    useNccl = true, -- Nvidia's library bindings for parallel table
    save = "src/main/resources/torch-data/",
}

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

function util.loadData(numExamples, numTestExamples, geometry)
    trainData = mnist.loadTrainSet(numExamples, geometry)
    trainData:normalizeGlobal()

    testData = mnist.loadTestSet(numTestExamples, geometry)
    testData:normalizeGlobal()
    return trainData, testData

end

function util.applyCuda(flag, module) if flag then require 'cunn' return module:cuda() else return module end end

function util.w_init_xavier(fan_in, fan_out)
    return math.sqrt(2/(fan_in + fan_out))
end

function util.w_init_xavier_caffe(fan_in, fan_out)
    return math.sqrt(1/fan_in)
end

function util.w_init_xavier_dl4j(fan_in, fan_out)
    return math.sqrt(1/(fan_in + fan_out))
end

function util.updateParams(model)
    for i=1, #model.modules do
        method = util.w_init_xavier_dl4j
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

function util.makeDataParallelTable(model, use_cudnn, nGPU)
    local net = model
    local dpt = nn.DataParallelTable(1, opt.flatten, opt.useNccl)
    for i = 1, nGPU do
        cutorch.setDevice(i)
        dpt:add(net:clone(), i)
        if use_cudnn then
            dpt:threads(function()
                local cudnn = require 'cudnn'
                cudnn.verbose = false
                cudnn.fastest,
                cudnn.benchmark = opt.cudnn_fastest,
                opt.cudnn_benchmark
            end)
--            dpt:add(model:get(opt.nGPU), opt.gpus)
        end
        dpt.gradInput = nil
        model = dpt:cuda()
    end
    return model
end

function util.convertCuda(model, use_cudnn, nGPU)
    require 'cunn'
    --    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    if use_cudnn then
        local cudnn = require 'cudnn'
        cudnn.convert(model:get(nGPU), cudnn)
        cudnn.verbose = false
        cudnn.benchmark = true
        if opt.cudnn_fastest then
            for _,v in ipairs(model:findModules'cudnn.SpatialConvolution') do v:fastest() end
        end
        if opt.cudnn_deterministic then
            model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
        end
    end
    if nGPU > 1 then
        model = util.makeDataParallelTable(model, use_cudnn, nGPU)
    else
        model = util.applyCuda(true, model)
    end
    return model
end


function util.printTime(time_type, time)
    local min = math.floor(time/60)
    local partialSec = min - time/60
    local sec = 0
    if partialSec > 0 then
        sec = math.floor(partialSec * 60)
    end
    local milli = time * 1000
    print(time_type .. ' time:' .. min .. ' min ' .. sec .. 'sec | ' .. milli .. ' millisec')
end

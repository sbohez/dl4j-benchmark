--
-- Torch Utils
--

require 'cutorch'

util = {}

opt = {
    cudnn_fastest = true,
    cudnn_deterministic = false,
    optnet_optimize = true,
}

function util.applyCuda(flag, module) if flag then require 'cunn' return module:cuda() else return module end end

function util.printTime(time_type, time)
    local min = math.floor(time/60)
    local partialSec = min - time/60
    local sec = 0
    if partialSec > 0 then
        sec = math.floor(partialSec * 60)
    end
    local milli = time * 1000
    print(string.format(time_type .. ' time: %0.2f min %0.2f sec %0.2f millisec', min, sec,  milli))
end


function util.convertCuda(model, nGPU)
    require 'cunn'
    --    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    if model_config.usecuDNN then
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
        model = util.makeDataParallelTable(model, nGPU)
    else
        model = util.applyCuda(true, model)
    end
    return model
end

function util.makeDataParallelTable(model, nGPU)
    local net = model
    local dpt = nn.DataParallelTable(1, opt.flatten, opt.useNccl)
    for i = 1, nGPU do
        cutorch.withDevice(i, function()
            dpt:add(net:clone(), i)
        end)
        dpt.gradInput = nil
        model = dpt:cuda()
    end
    return model
end

--
-- Util for benchmarks
--
require 'torch'
require 'nn'

util = {}

opt = {
    cudnn_fastest = true,
    cudnn_deterministic = false,
    multiply_input_factor = 1,
}

function util.makeDataParallelTable(model, nGPU)
    if nGPU > 1 then
        local gpus = torch.range(1, nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end
    return model
end

function util.convertCuda(model, use_cudnn)
    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())
    if(use_cudnn) then
        require 'cudnn'
        cudnn.convert(model, cudnn)
        cudnn.verbose = false
        cudnn.benchmark = true
        if opt.cudnn_fastest then
            for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
        end
        if opt.cudnn_deterministic then
            model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
        end
    end
    model:add(util.makeDataParallelTable(model))

    return model
end


function util.printTime(time_type, time)
    local min = time/60000 - time/1000
    local sec = time/1000 - min
    local milli = time
    print(time_type .. ' load time:' .. min .. ' min ' .. sec .. 'sec | ' .. milli .. ' millisec')
end

function util.cast(t, gpu)
    if gpu  then
        require 'cunn'
        return t:cuda()
    else
        return t:float()
    end
end

function util.w_init_xavier(fan_in, fan_out)
    return math.sqrt(2/(fan_in + fan_out))
end

function util.w_init_xavier_caffe(fan_in, fan_out)
    return math.sqrt(1/fan_in)
end

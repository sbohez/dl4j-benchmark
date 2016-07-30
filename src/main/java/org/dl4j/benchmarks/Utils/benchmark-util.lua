--
-- Util for benchmarks
--
require 'torch'
require 'nn'

util = {}

opt = {
    cudnn_fastest = true,
    cudnn_deterministic = false,
    cudnn_benchmark = true,
    flatten = true,
    useNccl = true -- Nvidia's library bindings for parallel table
}

function util.makeDataParallelTable(model, use_cudnn)
        local dpt
        if use_cudnn then
            print("CUDNN")
            dpt = nn.DataParallelTable(1, opt.flatten, opt.useNccl)
            :add(model, gpus)
            :threads(function()
                local cudnn = require 'cudnn'
                cudnn.verbose = false
                cudnn.fastest, cudnn.benchmark = opt.cudnn_fastest, opt.cudnn_benchmark
            end)

        else
            dpt = nn.DataParallelTable(1, true, true)
            :add(model, gpus)
        end
        dpt.gradInput = nil
        model = dpt:cuda()
    return model
end

function util.convertCuda(model, use_cudnn, nGPU)
--    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    if use_cudnn then
        require 'cudnn'
        cudnn.convert(model, cudnn)
        cudnn.verbose = false
        cudnn.benchmark = true
        if opt.cudnn_fastest then
            for _,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
        end
        if opt.cudnn_deterministic then
            model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
        end
    end
    if nGPU > 1 then
        model:add(util.makeDataParallelTable(model, use_cudnn))
    end
    return model
end


function util.printTime(time_type, time)
    local min = math.floor(time/60)
    local sec = math.floor((time/60 - min) * 100)
    local milli = time * 1000
    print(time_type .. ' time:' .. min .. ' min ' .. sec .. 'sec | ' .. milli .. ' millisec')
end

function util.cast(t, gpu)
    if gpu then
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

function util.updateParams(model)
    for i=1, #model.modules do
        method = util.w_init_xavier_caffe
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

function util.applyCuda(flag, module) if flag then require 'cunn' return module:cuda() else return module end end

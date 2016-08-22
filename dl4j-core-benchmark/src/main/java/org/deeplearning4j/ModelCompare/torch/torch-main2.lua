-- Torch7 Lenet
--
-- timing issues led to creating single script. Retaining till all timing issues confirmed resolved.


require 'torch'
require 'nn'
require 'optim'
require 'logroll'
require 'dl4j-core-benchmark/src/main/resources/torch-data/dataset-mnist'

log = logroll.print_logger()
log.level = logroll.INFO

cmd = torch.CmdLine()
cmd:option('-gpu', false, 'boolean flag to use gpu for training')
cmd:option('-multi', false, 'boolean flag to use multi-gpu for training')
cmd:option('-threads', 8, 'Number of threads to use on the computer')
cmd:option('-model_type', 'mlp', 'Which model to run')
config = cmd:parse(arg)

model_config = { lenet = {
        usecuDNN = config.cudnn,
        learningRate = 1e-2,
        weightDecay = 5e-4,
        nesterov = true,
        momentum =  0.9,
        dampening = 0,
    }, mlp = {
        usecuDNN = false,
        learningRate = 0.006,
        weightDecay = 6e-3,
        nesterov = true,
        momentum =  0.9,
        dampening = 0,
    }
}

opt = {
    gpu = config.gpu,
    multi = config.multi,
    max_epoch = 15,
    nExamples = 60000,
    nTestExamples = 10000,
    batchSize = 100,
    numClasses = 10,
    channels = 1,
    height = 28,
    width = 28,
    numInputs = 28*28,
    nGPU = 1,
    threads = config.threads,
    logger = log.level == logroll.DEBUG,
    plot = false,--log.level == logroll.DEBUG,
    seed = 42,
    devid = 1,
    optimization = 'sgd',
    cudnn_fastest = true,
    cudnn_deterministic = false,
    cudnn_benchmark = true,
    flatten = true,
    useNccl = true, -- Nvidia's library bindings for parallel table
    save = "src/main/resources/torch-data/",
}

if config.model_type == 'mlp' then
    model_config = model_config.lenet
else
    model_config = model_config.lenet
end

total_time = sys.clock()
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu then
    require 'cutorch'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    if opt.multi then opt.nGPU = cutorch.getDeviceCount() end
    print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
end

optimState = {
    learningRate = model_config.learningRate,
    weightDecay = model_config.weightDecay,
    nesterov = model_config.nesterov,
    momentum =  model_config.momentum,
    dampening = model_config.dampening
}

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {opt.height,opt.width}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

------------------------------------------------------------
-- Support Functions

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

function makeDataParallelTable(model, nGPU)
    local net = model
    local dpt = nn.DataParallelTable(1, opt.flatten, opt.useNccl)
    for i = 1, nGPU do
        cutorch.withDevice(i, function()
            dpt:add(net:clone(), i)
        end)
        --        if use_cudnn then
        --            dpt:threads(function()
        --                local cudnn = require 'cudnn'
        --                cudnn.verbose = false
        --                cudnn.fastest,
        --                cudnn.benchmark = opt.cudnn_fastest,
        --                opt.cudnn_benchmark
        --            end)
        --        end
        dpt.gradInput = nil
        model = dpt:cuda()
    end
    return model
end

function applyCuda(flag, module) if flag then require 'cunn' return module:cuda() else return module end end

function convertCuda(model, nGPU)
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
        model = makeDataParallelTable(model, nGPU)
    else
        model = applyCuda(true, model)
    end
    return model
end


function printTime(time_type, time)
    local min = math.floor(time/60)
    local partialSec = min - time/60
    local sec = 0
    if partialSec > 0 then
        sec = math.floor(partialSec * 60)
    end
    local milli = time * 1000
    print(string.format(time_type .. ' time: %0.2f min %0.2f sec %0.2f millisec', min, sec,  milli))
end

------------------------------------------------------------
log.debug('Load data...')

function loadData(numExamples, numTestExamples, geometry)
    trainData = mnist.loadTrainSet(numExamples, geometry)
    trainData:normalizeGlobal()
    testData = mnist.loadTestSet(numTestExamples, geometry)
    testData:normalizeGlobal()
    return trainData, testData
end

------------------------------------------------------------
log.debug('Build model...')

function w_init_xavier(fan_in, fan_out)
    return math.sqrt(2/(fan_in + fan_out))
end

function w_init_xavier_caffe(fan_in, fan_out)
    return math.sqrt(1/fan_in)
end

function w_init_xavier_dl4j(fan_in, fan_out)
    return math.sqrt(1/(fan_in + fan_out))
end

model = nn.Sequential()
if config.model_type == 'mlp' then
    nhidden = 1000
    model:add(nn.Reshape(opt.numInputs))
    model:add(nn.Linear(opt.numInputs,nhidden))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(nhidden,opt.numClasses))
else
    ccn1depth = 20
    ccn2depth = 50
    ffn1depth = 500
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(opt.channels, ccn1depth, 5, 5))
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
    model:add(nn.Linear(ffn1depth, opt.numClasses))
end
model = updateParams(model)

if opt.gpu then model = convertCuda(model, opt.nGPU) end

parameters,gradParameters = model:getParameters()
criterion = applyCuda(opt.gpu, nn.CrossEntropyCriterion())

if opt.logger then
    print("GPUS", opt.nGPU)
    print("MODEL", model)
end

------------------------------------------------------------
--Train model

function train(dataset)
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    if opt.multi then
        cutorch.synchronize()
    end
    local loss
    local lossVal = 0
    for t=1,dataset.size(),opt.batchSize do
        -- disp moving progress for data load
        if opt.logger then xlua.progress(t, dataset:size()) end
        --create a minibatch
        local inputs = applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = applyCuda(opt.gpu, torch.zeros(opt.batchSize))
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone():resize(opt.channels,opt.height,opt.width)
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        local feval =  function(x)
            --get new parameters
            if x ~= parameters then parameters:copy(x) end
            --reset gradients
            model:zeroGradParameters()
            local output = model:forward(inputs)
            --average error of criterion
            loss =  criterion:forward(output,targets)
            --estimate df/dW
            local df_do = criterion:backward(output,targets)
            model:backward(inputs,df_do)
            return loss, gradParameters
        end
        if opt.nGPU > 1 then
            model:syncParameters()
        end
        --        if opt.multi then cutorch.syncronize() end
        optim.sgd(feval,parameters,optimState)
        lossVal = loss + lossVal
    end
    confusion:updateValids()
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100 }
    -- plot errors
    if opt.plot then
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        trainLogger:plot()
    end
    confusion:zero()
    if opt.logger then
        print(confusion)
        print(string.format('Loss: [%.2f]', lossVal))
    end
    collectgarbage()
end

------------------------------------------------------------
--Evaluate model

function test(dataset)
    log.debug('Evaluate model...')
    --     test over given dataset
    model:evaluate()
    for t = 1,dataset:size(),opt.batchSize do
        -- disp moving progress for data load
        if opt.logger then xlua.progress(t, dataset:size()) end
        -- create mini batch
        local inputs = applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = applyCuda(opt.gpu, torch.zeros(opt.batchSize))
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone():resize(opt.channels,opt.height,opt.width)
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        local preds = model:forward(inputs)
        confusion:batchAdd(preds, targets)
        -- plot errors
        if opt.plot then
            testLogger:style{['% mean class accuracy (test set)'] = '-'}
            testLogger:plot()
        end
    end
    -- print confusion matrix
    confusion:updateValids()
    if opt.logger then print(confusion) end
    print(string.format('Accuracy: [%0.2f]', confusion.totalValid * 100))
end

------------------------------------------------------------
-- Run

data_load_time = sys.clock()
trainData, testData =  loadData(opt.nExamples, opt.nTestExamples, geometry)
data_load_time = sys.clock() - data_load_time

parameters,gradParameters = model:getParameters()

if opt.logger then
    print("GPUS", opt.nGPU)
    print("MODEL", model)
end

log.debug('Train model...')
model:training()
train_time = sys.clock()
for epoch = 1,opt.max_epoch do
    log.debug('<trainer> on training set:')
    log.debug("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    train(trainData)
end
train_time = sys.clock() - train_time

test_time = sys.clock()
test(testData)
test_time = sys.clock() - test_time
total_time = sys.clock() - total_time

print("****************Example finished********************")
printTime('Data load', data_load_time)
printTime('Train', train_time)
printTime('Test', test_time)
printTime('Total', total_time)
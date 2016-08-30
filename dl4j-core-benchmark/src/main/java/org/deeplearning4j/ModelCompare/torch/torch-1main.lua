-- Torch7 Main Class
-- Run Lenet or MLP
--
require 'torch'
require 'nn'
require 'dl4j-core-benchmark/src/main/resources/torch-data/dataset-mnist'
require 'optim'
require 'logroll'
require 'dl4j-core-benchmark/src/main/java/org/deeplearning4j/ModelCompare/torch/torch-lenet'
require 'dl4j-core-benchmark/src/main/java/org/deeplearning4j/ModelCompare/torch/torch-mlp'
require 'dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/torch-utils'

------------------------------------------------------------
-- Setup Variables
log = logroll.print_logger()
log.level = logroll.INFO

cmd = torch.CmdLine()
cmd:option('-gpu', false, 'boolean flag to use gpu for training')
cmd:option('-cudnn', true, 'boolean flag to use cudnn for training')
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
    nExamples = 60000 ,
    nTestExamples = 10000,
    batchSize = 100,
    testBatchSize = 100,
    numOutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    numInputs = 28*28,
    nGPU = 1,
    cudnn_fastest = true,
    cudnn_deterministic = false,
    cudnn_benchmark = true,
    flatten = true,
    useNccl = true, -- Nvidia's library bindings for parallel table
    save = "src/main/resources/torch-data/",
    threads = config.threads,
    logger = log.level == logroll.DEBUG,
    plot = false,
    seed = 42,
    devid = 1,
    optimization = 'sgd'

}

if config.model_type == 'mlp' then
    model_config = model_config.lenet
else
    model_config = model_config.lenet
end

optimState = {
    learningRate = model_config.learningRate,
    weightDecay = model_config.weightDecay,
    nesterov = model_config.nesterov,
    momentum =  model_config.momentum,
    dampening = model_config.dampening
}


total_time = sys.clock()
torch.manualSeed(42)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu then
    require 'cutorch'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    if opt.multi then opt.nGPU = cutorch.getDeviceCount() end
    print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
end

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {opt.height, opt.width}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

------------------------------------------------------------
-- Support Functions

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


------------------------------------------------------------
-- Load Data

function loadData(numExamples, numTestExamples, geometry)
    log.debug('Load data...')
    trainData = mnist.loadTrainSet(numExamples, geometry)
    trainData:normalizeGlobal()
    testData = mnist.loadTestSet(numTestExamples, geometry)
    testData:normalizeGlobal()
    return trainData, testData
end


------------------------------------------------------------
-- Train

function train(dataset, model, criterion)
    if opt.multi then cutorch.synchronize() end
    local loss
    local lossVal = 0
    --    loops from 1 to full dataset size by batchsize
    for t = 1,dataset.size(), opt.batchSize do
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
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            model:zeroGradParameters()
            local outputs = model:forward(inputs)
            loss = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            -- update confusion
            return loss, gradParameters
        end
        if opt.nGPU > 1 then model:syncParameters() end
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
-- Evaluate

function test(dataset)
    log.debug('Evaluate model...')
    model:evaluate()
    for t = 1,dataset:size(),opt.testBatchSize do
        -- create mini batch
        local inputs = applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = applyCuda(opt.gpu, torch.zeros(opt.batchSize))
        local k = 1
        for i = t,math.min(t+opt.testBatchSize-1,opt.nTestExamples) do
            -- disp moving progress for data load
            if opt.logger then xlua.progress(t, dataset:size()) end
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
    print('Accuracy: ', confusion.totalValid * 100)
end


------------------------------------------------------------
-- Run

data_load_time = sys.clock()
trainData, testData =  loadData(opt.nExamples, opt.nTestExamples, geometry)
data_load_time = sys.clock() - data_load_time

log.debug('Build model...')
if config.model_type == 'lenet' then
    model = lenet.build_model(opt.channels, opt.numOutputs)
    criterion = applyCuda(opt.gpu, lenet.define_loss())
else
    model = mlp.build_model(opt.numInputs, opt.numOutputs)
    criterion = applyCuda(opt.gpu, mlp.define_loss())
end

if opt.gpu then model = convertCuda(model, opt.nGPU) end
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
    train(trainData, model, criterion)
end
train_time = sys.clock() - train_time

test_time = sys.clock()
test(testData)
test_time = sys.clock() - test_time
total_time = sys.clock() - total_time

print("****************Example finished********************")
util.printTime('Data load', data_load_time)
util.printTime('Train', train_time)
util.printTime('Test', test_time)
util.printTime('Total', total_time)

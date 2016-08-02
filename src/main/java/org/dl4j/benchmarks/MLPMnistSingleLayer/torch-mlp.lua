-- Torch7 MLP
--
-- Reference Code: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
-- Reference Xaviar: https://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua#L19

require 'torch'
require 'nn'
require 'optim'
require 'src/main/java/org/dl4j/benchmarks/Utils/benchmark-util'

total_time = sys.clock()
torch.manualSeed(42)
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-gpu', false, 'boolean flag to use gpu for training')
cmd:option('-multi', false, 'boolean flag to use multi-gpu for training')
config = cmd:parse(arg)
if config.multi then print("Multi-GPU Not Implemented Yet") end

opt = {
    gpu = config.gpu,
    usecuDNN = false,
    max_epoch = 1,
    numExamples = 60000, -- after it throws errors in target and doesn't properly load
    numTestExamples = 10000,
    batchSize = 100,
    noutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    ninputs = 28*28,
    nhidden = 1000,
    multiply_input_factor = 1,
    nGPU = 1,
    learningRate = 0.006,
    weightDecay = 1e-4,
    nesterov = true,
    momentum =  0.9,
    dampening = 0,
}

if opt.multi then opt.nGPU = 4 end

if opt.gpu then
    require 'cutorch'
    print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
end

optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    nesterov = opt.nesterov,
    momentum =  opt.momentum,
    dampening = opt.dampening
}

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {opt.height,opt.width}

------------------------------------------------------------
-- print('Load data')
data_load_time = sys.clock()

trainData, testData = util.loadData()

data_load_time = sys.clock() - data_load_time
------------------------------------------------------------
-- print('Build model')

model = nn.Sequential()
model:add(nn.Reshape(opt.ninputs))
model:add(nn.Linear(opt.ninputs,opt.nhidden))
model:add(nn.ReLU())
model:add(nn.Linear(opt.nhidden,opt.noutputs))
model = util.updateParams(model)


if(opt.gpu) then model = util.convertCuda(model, false, opt.nGPU) end

local parameters,gradParameters = model:getParameters()
criterion = util.applyCuda(opt.gpu, nn.CrossEntropyCriterion())

------------------------------------------------------------
--print('Train model')
function train(dataset)
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()
    for t=1,dataset.size(),opt.batchSize do
        --create a minibatch
        local inputs = util.applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = util.applyCuda(opt.gpu, torch.zeros(opt.batchSize))
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
        -- create a closure to evaluate f(x) and df(x)/dW i.e. dZ/dW
        local feval =  function(x)
            -- just in case:
            collectgarbage()
            --get new parameters
            if x ~= parameters then parameters:copy(x) end
            --reset gradients
            gradParameters:zero()
            local output = model:forward(inputs)
            --average error of criterion
            local loss =  criterion:forward(output,targets)
            --estimate df/dW
            local df_do = criterion:backward(output,targets)
            model:backward(inputs,df_do)
            return loss, gradParameters
        end
        optim.sgd(feval,parameters,optimState)
    end
end
------------------------------------------------------------
--print('Evaluate model')

confusion = optim.ConfusionMatrix(classes)

function test(dataset)
    print('Evaluate')
--     test over given dataset
    model:evaluate()

    for t = 1,dataset:size(),opt.batchSize do

        -- create mini batch
        local inputs = util.applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = util.applyCuda(opt.gpu, torch.zeros(opt.batchSize))
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
    end
    -- print confusion matrix
    confusion:updateValids()
    print(confusion)
    print('Accuracy: ', confusion.totalValid * 100)
    confusion:zero()
end

-- Run program
train_time = sys.clock()
for _ = 1,opt.max_epoch do
    train(trainData)
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



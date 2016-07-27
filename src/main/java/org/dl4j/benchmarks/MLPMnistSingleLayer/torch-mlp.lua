-- Torch7 MLP
--
-- Reference Code: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
-- Reference Xaviar: https://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua#L19

--require 'sys'
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'src/main/resources/torch-data/dataset-mnist'
--mnist = require 'mnist' -- alternative but was giving bad results
require 'src/main/java/org/dl4j/benchmarks/Utils/benchmark-util'

total_time = sys.clock()
torch.manualSeed(42)

--print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

opt = {
    gpu = false,
    usecuDNN = false,
    max_epoch = 15,
    numExamples = 59904, -- after it throws errors in target and doesn't properly load
    numTestExamples = 10000,
    batchSize = 128,
    noutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    ninputs = 28*28,
    nhidden = 1000,
    multiply_input_factor = 1,
    nGPU = 4,
    learningRate = 0.006,
    weightDecay = 1e-4,
    nesterov = true,
    momentum =  0.9,
    dampening = 0
}

--opt = lapp[[
--   -s,--save                  (default "logs")      subdirectory to save logs
--   -b,--batchSize             (default 128)          batch size
--   -r,--learningRate          (default 1)        learning rate
--   --learningRateDecay        (default 1e-7)      learning rate decay
--   --weightDecay              (default 0.0005)      weightDecay
--   -m,--momentum              (default 0.9)         momentum
--   --epoch_step               (default 25)          epoch step
--   --model                    (default vgg_bn_drop)     model name
--   --max_epoch                (default 300)           maximum number of iterations
--   --backend                  (default nn)            backend
--   --type                     (default cuda)          cuda/float/cl
--]]

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

trainData = mnist.loadTrainSet(opt.numExamples, geometry)
trainData:normalizeGlobal()

testData = mnist.loadTestSet(opt.numTestExamples, geometry)
testData:normalizeGlobal()

--trainData = mnist.traindataset()
--testData = mnist.testdataset()
data_load_time = sys.clock() - data_load_time
------------------------------------------------------------
-- print('Build model')

model = nn.Sequential()
model:add(nn.Reshape(opt.ninputs))
model:add(nn.Linear(opt.ninputs,opt.nhidden))
model:add(nn.ReLU())
model:add(nn.Linear(opt.nhidden,opt.noutputs))
model:add(nn.LogSoftMax())
--model:add(util.cast(nn.Copy('torch.FloatTensor', torch.type(util.cast(torch.Tensor(), opt.gpu)))), opt.gpu)
--model:add(util.cast(model, opt.gpu))

if(opt.gpu) then
    require 'cunn'
    model:cuda()
    model = util.convertCuda(model, opt.usecuDNN)
end

for i=1, #model.modules do
    method = util.w_init_xavier_caffe
    local m = model.modules[i]
    if m.__typename == 'nn.Linear' then
        m:reset(method(m.weight:size(2), m.weight:size(1)))
        m.bias:zero()
    end
end

parameters,gradParameters = model:getParameters()

criterion = opt.gpu and nn.ClassNLLCriterion():cuda() or nn.ClassNLLCriterion()

------------------------------------------------------------
-- print('Train model')
function train(dataset)

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    for t=1,dataset.size(),opt.batchSize do

        --create a minibatch
--        local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
        local inputs = opt.gpu and torch.CudaTensor(opt.batchSize,1,28,2) or torch.Tensor(opt.batchSize,1,28,28)
        local targets = opt.gpu and torch.CudaTensor(opt.batchSize):zero() or torch.zeros(opt.batchSize)
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
--        for i = t,math.min(t+opt.batchSize-1,dataset.size) do
--            local sample = dataset[i]
--            local input = sample.x:clone()
--            local target = sample.y+1
--            inputs[k] = input
--            if target <= 0 then
--                print(target) -- 59905 stops converting to int?
--            end
--            targets[k] = target
--            k = k + 1
--        end

        inputs = opt.gpu and inputs:cuda() or inputs
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
print('Evaluate')
confusion = optim.ConfusionMatrix(classes)

function test(dataset)
--     test over given dataset
    model:evaluate()

    for t = 1,dataset:size(),opt.batchSize do

        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width)
        local targets = torch.Tensor(opt.batchSize)
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
--    for t=1,dataset.size,opt.batchSize do
--
--        -- create mini batch
--        local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
--        local targets = torch.Tensor(opt.batchSize)
--        local k = 1
--        for i = t,math.min(t+opt.batchSize-1,dataset.size) do
--            local sample = dataset[i]
--            local input = sample.x:clone()
--            local target = sample.y+1
--            inputs[k] = input
--            targets[k] = target
--            k = k + 1
--        end

        -- test samples

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
test_time = sys.clock()
test(testData)
test_time = sys.clock() - test_time
total_time = sys.clock() - total_time

print("****************Example finished********************")
util.printTime('Data load', data_load_time)
util.printTime('Train', train_time)
util.printTime('Test', test_time)
util.printTime('Total', total_time)



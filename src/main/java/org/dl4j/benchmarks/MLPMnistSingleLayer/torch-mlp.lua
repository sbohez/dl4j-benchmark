--require 'sys'
--require 'cunn'
--require 'ccn2'
--require 'cudnn'

require 'torch'
require 'nn'
require 'optim'
require 'src/main/resources/torch-data/dataset-mnist'
--mnist = require 'mnist' -- alternative but was giving bad results

--cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
--cudnn.verbose = false
torch.manualSeed(42)
torch.setdefaulttensortype('torch.FloatTensor')

--print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

opt = {
    gpu = true,
    max_epoch = 15,
    numExamples = 59904, -- after it throws errors in target and doesn't properly load
    numTestExamples = 10000,
    batchSize = 128,
    noutputs = 10,
    channels = 1,
    height = 32,
    width = 32,
    ninputs = 32*32,
--    height = 28,
--    width = 28,
--    ninputs = 28*28,
    nhidden = 1000,
    coefL2 = 1e-4
}

optimState = {
    learningRate = 0.006,
    weightDecay = 1e-4,
    nesterov = true,
    momentum =  0.9,
    dampening = 0
}

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {opt.height,opt.width }


------------------------------------------------------------
-- print('Load data')
trainData = mnist.loadTrainSet(opt.numExamples, geometry)
mean = trainData.data:mean()
std =  trainData.data:std()
trainData:normalizeGlobal(mean, std)

testData = mnist.loadTestSet(opt.numTestExamples, geometry)
mean = testData.data:mean()
std =  testData.data:std()
testData:normalizeGlobal(mean, std)

--trainData = mnist.traindataset()
--testData = mnist.testdataset()


------------------------------------------------------------
-- print('Build model')
model = nn.Sequential()
model:add(nn.Reshape(opt.ninputs))
model:add(nn.Linear(opt.ninputs,opt.nhidden))
model:add(nn.ReLU(true))
model:add(nn.Linear(opt.nhidden,opt.noutputs))

parameters,gradParameters = model:getParameters()

model:add(nn.LogSoftMax())
criterion =  nn.ClassNLLCriterion()

-- TODO add Xavier

------------------------------------------------------------
-- print('Train model')
function train(dataset)

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    for t=1,dataset.size(),opt.batchSize do

        --create a minibatch
        local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone()
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
--                target = 1 -- 59905 stops converting to int?
--            end
--            targets[k] = target
--            k = k + 1
--        end

        -- create a closure to evaluate f(x) and df(x)/dW i.e. dZ/dW
        local feval =  function(x)
            -- just in case:
            collectgarbage()

            --get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            --reset gradients
            gradParameters:zero()

            local output = model:forward(inputs)

            --f is the average error of criterion
            local f =  criterion:forward(output,targets)

            --estimate df/dW
            local df_do = criterion:backward(output,targets)
            model:backward(inputs,df_do)

            -- penalties (L1 and L2):
            local norm= torch.norm
            -- Loss:
            f = f + opt.coefL2 * norm(parameters,2)^2/2
            -- Gradients:
            gradParameters:add(parameters:clone():mul(opt.coefL2))

            return f, gradParameters
        end
        optim.sgd(feval,parameters,optimState)
    end

end
------------------------------------------------------------
-- print('Evaluate')
confusion = optim.ConfusionMatrix(classes)

function test(dataset)
    -- test over given dataset
    for t = 1,dataset:size(),opt.batchSize do

        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone()
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
--            if target <=0 then
--                target = 1
--            end
--            inputs[k] = input
--            targets[k] = target
--            k = k + 1
--        end

        -- test samples
        local preds = model:forward(inputs)

        -- confusion:
        for i = 1,opt.batchSize do
            confusion:add(preds[i], targets[i])
        end
    end

    -- print confusion matrix
--    print(confusion)
    print('Accuracy: ', confusion.totalValid * 100)
    confusion:zero()
end

-- Run program
time = sys.clock()
for _ = 1,opt.max_epoch do
    train(trainData)
end
test(testData)
print("Total time: ", (sys.clock() - time))






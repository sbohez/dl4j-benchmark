--
--
-- Reference: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
--

require 'torch'
require 'nn'
require 'optim'
require 'src/main/resources/torch-data/dataset-mnist'

torch.manualSeed(42)
torch.setdefaulttensortype('torch.FloatTensor')

-- epoch tracker
opt = {
    gpu = true,
    max_epoch = 11,
    numExamples = 59904 , -- numExamples
    numTestExamples = 10000,
    batchSize = 66,
    testBatchSize = 100,
    noutputs = 10,
    channels = 1,
    height = 32,
    width = 32,
    ninputs = 32*32,
    coefL2 = 5e-4
}

optimState = {
    learningRate = 1e-2,
    weightDecay = opt.coefL2,
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

------------------------------------------------------------
-- print('Build model')
model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(1, 20, 5, 5))
model:add(nn.Identity())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(20, 50, 5, 5))
model:add(nn.Identity())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(50*4*4))
model:add(nn.Linear(50*4*4, 500))
model:add(nn.ReLU(true))
model:add(nn.Linear(500, #classes))

-- reset weights TODO figure out if this works
method = 'xavier'
model_new = require('weight-init')(model, method)

--flattens & creates views for optim to process param and gradients
parameters,gradParameters = model_new:getParameters()

model_new:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

--print(model)

------------------------------------------------------------
-- print('Train model')
function train(dataset)

    model_new:training()

--    loops from 1 to full dataset size by batchsize
    for t = 1,opt.numExamples,opt.batchSize do
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

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- evaluate function for complete mini batch
            local outputs = model_new:forward(inputs)
            local f = criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model_new:backward(inputs, df_do)

            -- penalties (L1 and L2):
            local norm= torch.norm
            -- Loss:
            f = f + opt.coefL2 * norm(parameters,2)^2/2
            -- Gradients:
            gradParameters:add(parameters:clone():mul(opt.coefL2))

            -- return f and df/dX
            return f, gradParameters
        end

        optim.sgd(feval,parameters,optimState)
    end

end

------------------------------------------------------------
-- print('Evaluate')
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

function test(dataset)

    -- test over given dataset
    for t = 1,dataset:size(),opt.testBatchSize do
        -- disp progress
        xlua.progress(t, dataset:size())

        -- create mini batch
        local inputs = torch.Tensor(opt.testBatchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(opt.testBatchSize)
        local k = 1
        for i = t,math.min(t+opt.testBatchSize-1,opt.numTestExamples) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

        -- test samples
        local preds = model_new:forward(inputs)

        -- confusion:
        for i = 1,opt.batchSize do
            confusion:add(preds[i], targets[i])
        end
    end

    -- print confusion matrix
    print(confusion)
    print('Accuracy: ', confusion.totalValid * 100)
    confusion:zero()
end

time = sys.clock()
for _ = 1,opt.max_epoch do
    train(trainData)
end
test(testData)
print("Total time: ", (sys.clock() - time))

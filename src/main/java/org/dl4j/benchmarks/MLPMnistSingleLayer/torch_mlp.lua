--require 'sys'
--require 'cunn'
--require 'ccn2'
--require 'cudnn'

require 'torch'
require 'nn'
require 'optim'
require 'mnist'

--cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
--cudnn.verbose = false

torch.setdefaulttensortype('torch.FloatTensor')

--print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

opt = {
    gpu = true,
    max_epoch = 15,
    numExamples = 60000 , -- numExamples
    numTestExamples = 10000,
    batchSize = 128,
    noutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    ninputs = 784,
    nhidden = 1000
}

optimState = {
    learningRate = 0.006,
    weightDecay = 1e-4,
    momentum =  0.9
}

classes = {'1','2','3','4','5','6','7','8','9','10'}

------------------------------------------------------------
-- print('Load data')
--train  = '/train.t7'
--test =  '/test.t7'
--proj_path = '/resources/'
--
--loaded_train = torch.load(train,'ascii')
--loaded_test = torch.load(test,'ascii')
--trsize = math.min(loaded_train.labels:size()[1],opt.trsize_custom)
--tesize = loaded_test.labels:size()[1]

--trainData = {
--    --Conversion to double was necessary coz the model:forward() doesnot work on ByteTensor
--    data = loaded_train.data[{{1,trsize},{},{},{}}]:double(),
--    labels = loaded_train.labels[{{1,trsize}}],
--    size = function() return trsize end
--}
--
--testData = {
--    data = loaded_test.data:double(),
--    labels = loaded_test.labels,
--    size = function () return tesize end
--}


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


------------------------------------------------------------
-- print('Train model')
function train(dataset)

    -- epoch tracker
    epoch = epoch or 1 -- global variable (local keyword is not present)

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    for t=1,opt.numExamples,opt.batchSize do

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
        optim.nesterov(feval,parameters,optimState)
    end

    epoch =  epoch + 1

end
------------------------------------------------------------
-- print('Evaluate')
confusion = optim.ConfusionMatrix(classes)

function test(dataset)
    --print('Evaluate')
    local time = sys.clock()

    -- test over given dataset
    print('<trainer> on testing Set:')
    for t = 1,opt.numTestExamples,opt.batchSize do

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
        local preds = model:forward(inputs)

        -- confusion:
        for i = 1,opt.batchSize do
            confusion:add(preds[i], targets[i])
        end
    end

    -- print confusion matrix
    print(confusion)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero()
end

for _ = 1,opt.max_epoch do
    local time = sys.clock()
    train(trainData)
    test(testData)
    time = sys.clock() - time
    print("Total time: " .. sys.clock() - time)

end







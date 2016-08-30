--
-- Torch Utils
--


util = {}

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


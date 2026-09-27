-- KEYS[1]: speedinfer:balance:<api_key_id>
-- ARGV[1]: cost_to_deduct (number)
-- ARGV[2]: initial_trial (optional number for cache-miss seeding)
-- ARGV[3]: initial_paid (optional number for cache-miss seeding)

-- Handle legacy string keys if present by migrating to Hash
local key_type = redis.call('TYPE', KEYS[1]).ok
if key_type == 'string' then
    local old_val = tonumber(redis.call('GET', KEYS[1]) or 0)
    redis.call('DEL', KEYS[1])
    redis.call('HSET', KEYS[1], 'trial', '0', 'paid', tostring(old_val), 'total', tostring(old_val))
end

-- Check existence or seed from initial values
local exists = redis.call('EXISTS', KEYS[1])
if exists == 0 then
    if not ARGV[2] or not ARGV[3] then
        return {0, -1, -1, -1}
    end
    local st = tonumber(ARGV[2]) or 0
    local sp = tonumber(ARGV[3]) or 0
    redis.call('HSET', KEYS[1], 'trial', tostring(st), 'paid', tostring(sp), 'total', tostring(st + sp))
end

local trial = tonumber(redis.call('HGET', KEYS[1], 'trial') or 0)
local paid = tonumber(redis.call('HGET', KEYS[1], 'paid') or 0)
local cost = tonumber(ARGV[1])
local total = trial + paid

if total < cost then
    return {0, tostring(total), tostring(trial), tostring(paid)}
end

-- Priority deduction: trial balance first, then paid balance
local new_trial = trial
local new_paid = paid

if trial >= cost then
    new_trial = trial - cost
else
    local rem = cost - trial
    new_trial = 0
    new_paid = paid - rem
end

local new_total = new_trial + new_paid
redis.call('HSET', KEYS[1], 'trial', tostring(new_trial), 'paid', tostring(new_paid), 'total', tostring(new_total))

return {1, tostring(new_total), tostring(new_trial), tostring(new_paid)}

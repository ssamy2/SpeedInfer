-- KEYS[1]: bucket key
-- ARGV[1]: capacity
-- ARGV[2]: refill_rate (tokens per second)
-- ARGV[3]: requested_tokens
-- ARGV[4]: current_timestamp (seconds)
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

if not tokens or not last_updated then
    tokens = capacity
    last_updated = now
else
    local delta = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + delta * refill_rate)
    last_updated = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_updated', tostring(last_updated))
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 60)
    return {1, math.floor(tokens), 0}
else
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_updated', tostring(last_updated))
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 60)
    local retry_after = math.ceil((requested - tokens) / refill_rate)
    return {0, math.floor(tokens), retry_after}
end

-- KEYS[1]: speedinfer:balance:<api_key_id>
-- ARGV[1]: cost_to_deduct (number)
local current = redis.call('GET', KEYS[1])
if not current then
    return {0, -1}
end
local balance = tonumber(current)
local cost = tonumber(ARGV[1])
if balance < cost then
    return {0, tostring(balance)}
end
local new_balance = balance - cost
redis.call('SET', KEYS[1], tostring(new_balance))
return {1, tostring(new_balance)}

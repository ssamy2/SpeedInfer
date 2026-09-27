import assert from 'node:assert/strict';
import fs from 'node:fs';
globalThis.window = {location: {origin: 'http://test'}};
const source = fs.readFileSync(new URL('../speedinfer/frontend/js/api.js', import.meta.url));
const {api} = await import('data:text/javascript;base64,' + source.toString('base64'));
async function check(events) {
  let result, error;
  globalThis.fetch = async () => new Response(new ReadableStream({start(controller) {
    const bytes = new TextEncoder().encode(events);
    controller.enqueue(bytes.slice(0, 17));
    controller.enqueue(bytes.slice(17));
    controller.close();
  }}), {status: 200});
  await api.streamChatCompletion({model: 'test', messages: [{role:'user',content:'Hi'}],
    apiKey: 'test', onChunk() {}, onDone(value) {result=value;}, onError(value) {error=value;}});
  return {result, error};
}
const content = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n';
const usage = 'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5}}\n\n';
const success = await check(content + usage + 'data: [DONE]\n\n');
assert.equal(success.error, undefined);
assert.equal(success.result.promptTokens, 12);
assert.equal(success.result.completionTokens, 5);
for (const events of [content, content + 'data: [DONE]\n\n', content + 'data: {"error":{"message":"offline"}}\n\n']) {
  const failure=await check(events);
  assert.equal(failure.result, undefined);
  assert.ok(failure.error);
}
console.log('PASS: client uses authoritative usage and rejects SSE errors/truncated streams');

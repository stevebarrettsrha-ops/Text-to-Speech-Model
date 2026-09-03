// Hermetic smoke test for the Script Builder app.
// Serves the repo over HTTP and opens it in headless Chromium with EVERY
// third-party request (fonts, CDN, TTS APIs) blocked, then checks that the app
// still boots and its core behaviour works. Run with `npm test`.
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startServer } from '../scripts/serve.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const server = await startServer(root, 0);
const base = `http://127.0.0.1:${server.port}/`;
const browser = await chromium.launch();
let failed = 0;

const APP_TITLE = 'Script Builder — Natural Voices';
// Console noise produced by the deliberate request blocking.
const IGNORED = [/ERR_BLOCKED_BY_CLIENT/, /Failed to load resource/];

async function openPage({ viewport = { width: 1280, height: 800 }, init, url = base } = {}) {
  const ctx = await browser.newContext({ viewport });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => {
    if (m.type() === 'error' && !IGNORED.some(r => r.test(m.text()))) errors.push('console: ' + m.text());
  });
  await page.route('**/*', route =>
    route.request().url().startsWith(base) ? route.continue() : route.abort('blockedbyclient'));
  if (init) await page.addInitScript(init);
  const response = await page.goto(url);
  return { ctx, page, errors, response };
}

async function test(name, fn) {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (err) {
    failed++;
    console.error(`FAIL ${name}\n     ${(err && err.stack) || err}`);
  }
}

await test('app boots with all third-party resources blocked', async () => {
  const { ctx, page, errors } = await openPage();
  assert.equal(await page.title(), APP_TITLE);
  await page.waitForSelector('.dialog-block');
  assert.equal(await page.locator('.dialog-block').count(), 4, 'podcast preset renders 4 lines');
  assert.ok((await page.locator('#vsel-s1 option').count()) >= 14, 'speaker 1 has cloud voices');
  assert.ok((await page.locator('#vsel-s2 option').count()) >= 14, 'speaker 2 has cloud voices');
  assert.ok(await page.locator('#run-btn').isVisible(), 'Run button visible');
  assert.ok(await page.locator('#dl-btn').isVisible(), 'Download button visible');
  assert.ok((await page.textContent('#raw-text')).includes('Speaker 1:'), 'raw preview populated');
  assert.deepEqual(errors, []);
  await ctx.close();
});

await test('edits survive a reload (autosave)', async () => {
  const { ctx, page, errors } = await openPage();
  await page.waitForSelector('.dialog-block');
  await page.click('.add-btn');
  assert.equal(await page.locator('.dialog-block').count(), 5);
  await page.fill('#t-4', 'Autosave check line');
  await page.fill('#name-s1', 'Host');
  await page.waitForTimeout(400); // autosave debounce
  await page.reload();
  await page.waitForSelector('.dialog-block');
  assert.equal(await page.locator('.dialog-block').count(), 5);
  assert.equal(await page.inputValue('#t-4'), 'Autosave check line');
  assert.equal(await page.inputValue('#name-s1'), 'Host');
  assert.ok((await page.textContent('#raw-text')).includes('Host: '));
  assert.deepEqual(errors, []);
  await ctx.close();
});

await test('boots without the Web Speech API (cloud voices only)', async () => {
  const { ctx, page, errors } = await openPage({
    init: () => { Object.defineProperty(window, 'speechSynthesis', { value: undefined, configurable: true }); },
  });
  await page.waitForSelector('.dialog-block');
  assert.equal(await page.locator('.dialog-block').count(), 4);
  assert.ok(await page.locator('#nosup').isVisible(), 'unsupported-browser notice shown');
  assert.ok((await page.locator('#vsel-s1 option').count()) >= 14, 'cloud voices still offered');
  assert.deepEqual(errors, []);
  await ctx.close();
});

await test('speaker names and dialog render as text, never as HTML', async () => {
  const { ctx, page, errors } = await openPage();
  await page.waitForSelector('.dialog-block');
  const payload = '<img src=x onerror="window.__pwned=1">';
  const dialog = payload + '</textarea><b id="broken">x</b>';
  await page.fill('#t-0', dialog);
  await page.fill('#name-s1', payload); // triggers a full re-render
  await page.waitForTimeout(100);
  assert.equal(await page.evaluate(() => window.__pwned), undefined, 'injected handler must not run');
  assert.equal(await page.locator('#broken').count(), 0, 'textarea must not be broken out of');
  assert.ok((await page.locator('#block-0 .dialog-speaker').textContent()).includes(payload));
  assert.equal(await page.inputValue('#t-0'), dialog);
  assert.deepEqual(errors, []);
  await ctx.close();
});

await test('unknown paths get the custom 404 page, which sends visitors back to the app', async () => {
  const { ctx, page, response } = await openPage({ url: base + 'no-such-page' });
  assert.equal(response.status(), 404);
  assert.match(await response.text(), /Page not found/);
  await page.waitForURL(base, { timeout: 10000 });
  assert.equal(await page.title(), APP_TITLE);
  await ctx.close();
});

await test('phone-sized layout has no horizontal overflow and keeps the controls reachable', async () => {
  const { ctx, page, errors } = await openPage({ viewport: { width: 390, height: 844 } });
  await page.waitForSelector('.dialog-block');
  const { scrollWidth, innerWidth } = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth,
  }));
  assert.ok(scrollWidth <= innerWidth, `page scrolls horizontally (${scrollWidth} > ${innerWidth})`);
  assert.ok(await page.locator('#run-btn').isVisible(), 'Run button visible');
  assert.ok(await page.locator('#dl-btn').isVisible(), 'Download button visible');
  assert.deepEqual(errors, []);
  await ctx.close();
});

await test('download falls back to WAV when the MP3 encoder cannot load', async () => {
  const { ctx, page, errors } = await openPage();
  await page.waitForSelector('.dialog-block');
  const result = await page.evaluate(() => {
    const captured = [];
    window.triggerDownload = (blob, name) => captured.push({ name, size: blob.size, type: blob.type });
    const actx = new OfflineAudioContext(1, 4410, 44100);
    const buf = actx.createBuffer(1, 4410, 44100);
    buf.getChannelData(0).fill(0.25);
    encodeAndDownload(buf);
    return { lame: typeof lamejs, captured };
  });
  assert.equal(result.lame, 'undefined', 'CDN encoder is blocked in this test');
  assert.deepEqual(result.captured, [{ name: 'script-audio.wav', size: 44 + 4410 * 2, type: 'audio/wav' }]);
  assert.deepEqual(errors, []);
  await ctx.close();
});

await browser.close();
await server.close();
if (failed) {
  console.error(`\n${failed} check(s) failed`);
  process.exit(1);
}
console.log('\nAll smoke checks passed');

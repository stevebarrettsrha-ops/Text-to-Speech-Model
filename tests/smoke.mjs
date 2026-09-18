// Drives the real interface in headless Chromium against the stand-in ComfyUI.
//
// Every check here stands for something that broke at some point. A page error
// or a failed request to the app is a failure, not a warning — a thrown error
// in the inline script kills all interactivity silently, which is the whole
// reason the validation gate in CLAUDE.md exists.
//
// Run with `npm test`. Needs no GPU, no model downloads and no network.
import { chromium } from 'playwright';
import { boot } from './harness.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
let passed = 0;
const failures = [];

function ok(name, detail = '') {
  passed++;
  console.log(`ok   ${name}${detail ? ` — ${detail}` : ''}`);
}
function bad(name, detail = '') {
  failures.push(`${name}${detail ? ` — ${detail}` : ''}`);
  console.error(`FAIL ${name}${detail ? ` — ${detail}` : ''}`);
}
const is = (cond, name, detail) => cond ? ok(name, detail) : bad(name, detail);

const app = await boot();
// CI runs `npx playwright install chromium` and Playwright finds its own.
// CHROMIUM_PATH is for environments that ship a browser already.
const browser = await chromium.launch(
  process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const noise = [];

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', e => noise.push(`page error: ${e.message}`));
  page.on('console', m => {
    if (m.type() === 'error') noise.push(`console: ${m.text()}`);
  });
  page.on('response', r => {
    if (r.url().startsWith(app.base) && r.status() >= 400) {
      noise.push(`HTTP ${r.status()} ${r.url().replace(app.base, '')}`);
    }
  });

  const takes = () => app.api('/api/takes');

  await page.goto(app.base, { waitUntil: 'networkidle' });
  await sleep(1500);

  /* ---------------------------------------------------------------- shell */
  is(await page.title() === 'Script Builder', 'page title', await page.title());

  const navs = await page.$$eval('.nav[data-view]', e => e.map(x => x.dataset.view));
  const pages = ['home', 'create', 'library', 'models', 'engine'];
  is(pages.every(v => navs.includes(v)), 'five pages in the rail', navs.join(', '));

  is(await page.$eval('#veil-setup', e => e.hidden),
     'setup panel stays shut when setup is complete');

  const pill = (await page.textContent('#enginePill')).trim();
  is(pill.includes('ready'), 'engine reports ready', pill);

  for (const view of pages) {
    await page.click(`.nav[data-view="${view}"]`);
    await sleep(900);
    const shown = await page.$eval(`[data-page="${view}"]`, e => !e.hidden);
    const chars = await page.$eval(`[data-page="${view}"]`, e => e.innerText.trim().length);
    is(shown && chars > 20, `page "${view}" renders`, `${chars} chars`);
  }

  /* --------------------------------------------------------- the builder */
  await page.click('.nav[data-view="create"]');
  await sleep(700);

  // scrollHeight reads 0 while a page is hidden, so blocks rendered
  // off-screen come back clipped unless resizeAll() runs when it is shown.
  let heights = await page.$$eval('#blocks textarea',
    e => e.map(x => x.getBoundingClientRect().height));
  is(heights.length > 0 && heights.every(h => h > 10),
     'textareas are sized once the page is on screen',
     heights.map(Math.round).join('/'));

  const voices = await page.$$eval('[data-voice="1"] option', o => o.map(x => x.value));
  is(voices.includes('Serena') && voices.length === 9,
     'preset voices come from the node', `${voices.length} voices`);

  const models = await page.$$eval('#model-sel option', o => o.map(x => x.value));
  is(JSON.stringify(models) === JSON.stringify(['0.6B', '1.7B']),
     'model list comes from the node', models.join(', '));

  const atts = await page.$$eval('#attn-sel option', o => o.map(x => x.value));
  is(atts.includes('sdpa'), 'attention modes come from the node', atts.join(', '));

  await page.click('#btnClearScript');
  await sleep(300);
  await page.fill('#style-input', 'Read briskly, like a trailer');
  for (const [i, text] of ['The browser is driving this.',
                           'And every line is its own graph.'].entries()) {
    await page.click('#btnAdd');
    await sleep(250);
    (await page.$$('#blocks textarea'))[i].fill(text);
    await sleep(200);
  }
  is((await page.$$('#blocks .blk')).length === 2, 'lines can be added');

  const raw = (await page.textContent('#raw-text')).trim();
  is(raw.includes('Read briskly') && raw.includes('Speaker 1:')
     && raw.includes('Speaker 2:'), 'the raw structure mirrors the script');

  await page.click('#blocks .blk:nth-child(2) [data-flip]');
  await sleep(300);
  is((await page.textContent('#raw-text')).split('\n')[2].startsWith('Speaker 1:'),
     'the speaker chip flips a line');
  await page.click('#blocks .blk:nth-child(2) [data-flip]');
  await sleep(300);

  await page.click('#blocks .blk:nth-child(1) [data-move][data-dir="1"]');
  await sleep(400);
  is((await page.$$eval('#blocks textarea', e => e.map(x => x.value)))[0]
     === 'And every line is its own graph.', 'move reorders a line');
  await page.click('#blocks .blk:nth-child(1) [data-move][data-dir="1"]');
  await sleep(400);

  await page.click('#segMode button[data-v="single"]');
  await sleep(700);
  is(await page.$$eval('#blocks .chip', e => e.every(x => x.className.includes('s1')))
     && await page.$eval('#spk-2', e => e.hidden),
     'one speaker collapses the script and hides the second');
  await page.click('#blocks .blk:nth-child(1) [data-flip]');
  await sleep(400);
  is(await page.$$eval('#blocks .chip', e => e.every(x => x.className.includes('s1'))),
     'flipping is refused in one-speaker mode');
  await page.click('#segMode button[data-v="multi"]');
  await sleep(600);

  is(await page.$eval('#cardVoices', e => e.open), 'cards start open');
  await page.click('#cardVoices > summary');
  await sleep(400);
  is(await page.$eval('#cardVoices', e => !e.open), 'cards collapse');
  await page.click('#cardVoices > summary');
  await sleep(400);

  /* ------------------------------------------------------------ generate */
  const before = (await takes()).length;
  await page.click('#btnRun');
  let made = false;
  for (let i = 0; i < 60; i++) {
    await sleep(1000);
    if ((await takes()).length > before) { made = true; break; }
  }
  is(made, 'pressing Read produces a take');
  await sleep(1500);

  const npTitle = (await page.textContent('#npTitle')).trim();
  is(npTitle && npTitle !== '—', 'the player bar picks the take up', npTitle);

  const audio = await page.evaluate(() => {
    const a = document.getElementById('audio');
    return { src: a.currentSrc || a.src, dur: a.duration, t: a.currentTime };
  });
  is(audio.src.includes('/api/take/'), 'the player streams a clip',
     audio.src.replace(app.base, ''));
  is(audio.dur > 0 || audio.t > 0, 'the clip decodes',
     `dur=${audio.dur} t=${audio.t.toFixed(2)}`);

  /* ----------------------------------------------------------- transport */
  const longTake = (await takes()).find(t => t.title === 'Transport test');
  if (!longTake) {
    bad('transport fixture missing');
  } else {
    await page.evaluate(t => { playTake(t, false); }, longTake);
    await sleep(1200);
    const read = () => page.evaluate(() => ({
      flag: document.getElementById('lineFlag').textContent,
      lines: S.take.lines.length,
    }));
    const seen = [await read()];
    for (const btn of ['#btnNext', '#btnNext', '#btnPrev', '#btnPrev']) {
      await page.click(btn);
      await sleep(1400);
      seen.push(await read());
    }
    const want = ['line 1 of 3', 'line 2 of 3', 'line 3 of 3',
                  'line 2 of 3', 'line 1 of 3'];
    is(JSON.stringify(seen.map(s => s.flag)) === JSON.stringify(want),
       'skip walks forward and back', seen.map(s => s.flag).join(' -> '));
    // stepLine used to hand playTake a sliced copy, which became S.take: the
    // counter renumbered itself and the lines the back button needed were gone.
    is(seen.every(s => s.lines === longTake.lines.length),
       'skipping does not cut the take down',
       `${longTake.lines.length} lines throughout`);
  }

  await page.click('#btnRepeat');
  await sleep(300);
  is(await page.evaluate(() => S.repeat), 'repeat toggles');
  await page.click('#btnRepeat');
  await sleep(200);
  await page.click('#btnPlay');
  await sleep(600);
  is(await page.evaluate(() => document.getElementById('audio').paused),
     'play/pause pauses');
  await page.evaluate(() => stopAll());
  await sleep(600);
  const cleared = await page.evaluate(() => ({
    playing: S.playing, flag: document.getElementById('lineFlag').textContent }));
  is(!cleared.playing && !cleared.flag, 'stop clears the player');

  /* ------------------------------------------------------------- library */
  await page.click('.nav[data-view="library"]');
  await sleep(900);
  const cards = (await page.$$('#libGrid .gcard')).length;
  is(cards > 0, 'the library lists takes', `${cards} cards`);

  await page.fill('#libSearch', 'Transport test');
  await sleep(600);
  const filtered = (await page.$$('#libGrid .gcard')).length;
  is(filtered > 0 && filtered < cards, 'search filters', `${cards} -> ${filtered}`);
  await page.fill('#libSearch', 'zzz-nothing-matches');
  await sleep(600);
  is(await page.$eval('#libGrid', e => e.innerText.includes('Nothing here yet')),
     'an empty search shows the empty state');
  await page.fill('#libSearch', '');
  await sleep(500);

  await page.evaluate(t => loadTakeScript(t), longTake);
  await sleep(900);
  const loaded = await page.$$eval('#blocks textarea', e => e.map(x => x.value));
  is(loaded.length === longTake.lines.length && loaded[0] === longTake.lines[0].text,
     'a take loads back into the builder', `${loaded.length} lines`);
  heights = await page.$$eval('#blocks textarea',
    e => e.map(x => x.getBoundingClientRect().height));
  is(heights.every(h => h > 10), 'the loaded-back lines are sized',
     heights.map(Math.round).join('/'));

  const countBefore = (await takes()).length;
  page.once('dialog', d => d.accept());
  await page.click('#takeList .trow:first-child [data-act="del"]');
  await sleep(2000);
  is((await takes()).length === countBefore - 1, 'delete removes a take',
     `${countBefore} -> ${(await takes()).length}`);
  const countAfter = (await takes()).length;
  page.once('dialog', d => d.dismiss());
  await page.click('#takeList .trow:first-child [data-act="del"]');
  await sleep(1200);
  is((await takes()).length === countAfter, 'cancelling the confirm keeps it');

  /* -------------------------------------------------------------- models */
  await page.click('.nav[data-view="models"]');
  await sleep(1400);
  const settings = await app.api('/api/hf/settings');
  is(settings.vram_mb === 8191,
     'the models page knows what the card holds', String(settings.vram_mb));
  const tooBig = settings.curated.filter(c => c.fits === false).map(c => c.repo);
  is(tooBig.length === 1 && tooBig[0] === 'OpenMOSS-Team/MOSS-TTS',
     'exactly the 8B is marked as more than this card can hold',
     tooBig.join(', ') || 'none');
  // Nothing a first run downloads may be something the card cannot then load.
  const defaults = settings.curated.filter(c => c.role !== 'optional');
  is(defaults.every(c => c.fits !== false),
     'everything first launch fetches will actually run',
     defaults.map(c => `${c.repo.split('/')[1]}:${c.fits}`).join(' '));

  // The self-test is the answer to "does this engine actually work", so it
  // has to run the whole way and it has to fail when the engine is broken.
  const st = async engine => {
    const r = await app.post(`/api/selftest/${engine}`, {});
    for (let i = 0; i < 90; i++) {
      await sleep(700);
      const t = await app.api(`/api/tasks?id=${r.task.id}&since=99999`);
      if (t.state !== 'running') return t;
    }
    return null;
  };
  const pass = await st('moss');
  const names = (pass.meta.steps || []).map(x => `${x.id}:${x.state}`);
  is(pass.state === 'done' && (pass.meta.steps || []).every(x => x.state !== 'fail'),
     'the MOSS self-test runs the whole way', names.join(' '));
  is((pass.meta.steps || []).some(x => x.id === 'audio' && /of audio/.test(x.detail)),
     'and proves audio came back',
     (pass.meta.steps || []).find(x => x.id === 'audio')?.detail);

  await app.comfyApi('/mock/hide/moss', { method: 'POST' }, 'moss');
  const broken = await st('moss');
  const nodeStep = (broken.meta.steps || []).find(x => x.id === 'nodes');
  is(broken.state === 'error' && nodeStep && nodeStep.state === 'fail',
     'and fails at the right step when ComfyUI has not loaded the nodes',
     nodeStep && nodeStep.state);
  // The schema is cached for two minutes; a self-test reading that cache
  // reported nodes as loaded straight after they were hidden.
  is(!(broken.meta.steps || []).some(x => x.id === 'audio'),
     'stopping early rather than carrying on to generate');

  // Put the nodes back and run it again. This is not tidying up: the app
  // caches ComfyUI's schema for two minutes, so the run that proves recovery
  // is also the thing that clears the stale "they are gone" answer for
  // everything after it.
  await app.comfyApi('/mock/hide/none', { method: 'POST' }, 'moss');
  const again = await st('moss');
  is(again.state === 'done',
     'and passes again once the nodes are back, without waiting out the cache',
     (again.meta.steps || []).map(x => `${x.id}:${x.state}`).join(' '));

  const eightB = await app.api('/api/moss/8b');
  is(eightB.through_comfyui.fits === false
       && /AutoModel\.from_pretrained/.test(eightB.through_comfyui.why),
     'the 8B report names the node as the limit, not the model');
  const notFetchable = eightB.through_llama_cpp.steps.filter(s => !s.obtainable);
  is(notFetchable.length === 2,
     'and admits two prerequisites cannot be downloaded at all',
     notFetchable.map(s => s.id).join(', '));

  const curated = await page.$$eval('#curated-list .fitem',
    e => e.map(x => x.querySelector('span').textContent.split(' · ')[0]));
  // Six Qwen folders and four MOSS ones, each row saying which engine it is
  // for — the two sets live in different folder shapes and are not
  // interchangeable.
  is(curated.filter(x => x === 'Qwen3-TTS').length === 6
       && curated.filter(x => x === 'MOSS-TTS').length === 4,
     'both engines\' model folders are listed', curated.join(', '));
  const installed = await page.$$eval('#curated-list .state.ok', e => e.length);
  is(installed >= 7, 'installed folders are marked', `${installed} installed`);

  /* ------------------------------------------------------- the two engines */
  await page.click('.nav[data-view="create"]');
  await sleep(600);
  const engines = await page.$$eval('#engine-sel option',
    e => e.map(x => ({ id: x.value, label: x.textContent })));
  is(engines.length === 2 && engines.some(e => e.id === 'qwen')
       && engines.some(e => e.id === 'moss'),
     'both engines are offered', engines.map(e => e.label).join(', '));

  // MOSS has no speaker enum on any node, so "Preset" would open an empty
  // dropdown. The same slot has to become the model's own voice instead.
  await page.selectOption('#engine-sel', 'moss');
  await sleep(2500);
  const mossSrc = await page.$$eval('[data-src="1"] button',
    e => e.map(x => x.textContent.trim()));
  is(mossSrc[0] === 'Own voice', 'MOSS drops the preset speaker list',
     mossSrc.join('/'));
  is(/no preset speakers/i.test(await page.textContent('[data-body="1"]')),
     'and says why rather than showing an empty picker');
  is(await page.$eval('#rowAttn', e => e.hidden),
     'the attention picker is hidden on MOSS, which has none');
  const mossModels = await page.$$eval('#model-sel option',
    e => e.map(x => ({ v: x.value, t: x.textContent, off: x.disabled })));
  is(mossModels.some(m => m.v.startsWith('OpenMOSS-Team/')),
     'the model picker carries MOSS repo ids',
     mossModels.map(m => m.v).join(', '));
  // The stand-in reports an 8 GB card. The 8B wants ~18 GB through these
  // nodes, so it has to be visible and unselectable rather than quietly
  // offered and then failing inside ComfyUI.
  const big = mossModels.find(m => m.v === 'OpenMOSS-Team/MOSS-TTS');
  is(big && big.off && /will not fit/i.test(big.t),
     'a model too big for the card is shown but cannot be chosen',
     big && big.t);
  is(!(await page.$eval('#model-sel', e => e.selectedOptions[0].disabled)),
     'and is never what the picker lands on');
  const mossStatus = await app.api('/api/status?engine=moss');
  is(mossStatus.ready === true && mossStatus.capabilities.preset === false,
     'MOSS reports ready with no presets',
     JSON.stringify(mossStatus.capabilities));

  // A take on MOSS, through the real page, on the two-node graph.
  await page.click('.nav[data-view="create"]');
  await sleep(400);
  const beforeMoss = (await takes()).length;
  await page.click('#btnRun');
  for (let i = 0; i < 90; i++) {
    await sleep(700);
    if ((await takes()).length > beforeMoss) break;
  }
  const mossTake = (await takes()).find(t => t.engine === 'moss');
  is(!!mossTake, 'MOSS produces a take', mossTake && mossTake.title);
  // MOSS's own ComfyUI, on its own port — the Qwen one should never have seen
  // a MOSS graph, and that is half of what this proves.
  const mossLog = await app.comfyApi('/mock/log', undefined, 'moss');
  const mossQueued = (mossLog.log || []).filter(l => l.includes('MossTTS'));
  const qwenSawMoss = ((await app.comfyApi('/mock/log')).log || [])
    .filter(l => l.includes('MossTTS'));
  is(qwenSawMoss.length === 0,
     "and Qwen's ComfyUI never saw a MOSS graph", String(qwenSawMoss.length));
  is(mossQueued.some(l => l.includes('MossTTSModelLoader')
                       && l.includes('MossTTSGenerate') && l.includes('|local')),
     'the MOSS graph is loader + generator, pointed at a local folder',
     mossQueued[mossQueued.length - 1] || 'none');

  await page.selectOption('#engine-sel', 'qwen');
  await sleep(2000);
  is((await page.$$eval('[data-src="1"] button',
       e => e.map(x => x.textContent.trim())))[0] === 'Preset',
     'switching back restores the preset speakers');

  /* -------------------------------------------------------------- engine */
  await page.click('.nav[data-view="engine"]');
  await sleep(2500);
  const deps = (await page.$$('#dep-list .fitem')).length;
  is(deps >= 6, 'the dependency list renders', `${deps} rows`);
  const depIds = (await app.api('/api/deps')).items.map(i => i.id);
  // Nothing below ComfyUI is shared any more, so nothing below ComfyUI gets
  // one row for both engines.
  const perEngine = ['comfyui', 'node', 'torch', 'node_reqs', 'models', 'engine'];
  is(perEngine.every(b => depIds.includes(`${b}_qwen`) && depIds.includes(`${b}_moss`)),
     'every engine has its own row for everything', depIds.join(', '));

  // torch 2.14.0+cpu landed on a machine with an RTX 4060 in it, because the
  // only test for a GPU was shutil.which("nvidia-smi"). The panel now says
  // out loud what Automatic resolved to, so a wrong answer is visible.
  const depPayload = await app.api('/api/deps');
  is(depPayload.gpu && typeof depPayload.gpu.name === 'string'
       && typeof depPayload.gpu.driver === 'boolean',
     'the engine report says what GPU it found',
     JSON.stringify(depPayload.gpu));
  is(/\/whl\/(cpu|cu\d+)$/.test(depPayload.torch_auto || ''),
     'Automatic resolves to a real wheel index', depPayload.torch_auto);
  const autoLabel = await page.$eval('#torch-index option', e => e.textContent);
  is(autoLabel.startsWith('Automatic —')
       && (depPayload.gpu.name
             ? autoLabel.includes(depPayload.gpu.name)
             : /no NVIDIA GPU found/.test(autoLabel)),
     'the PyTorch picker says what Automatic chose', autoLabel);

  // Restarting an engine that was never set up is a sentence, not a stack
  // trace — and never a 500.
  const restart = await app.post('/api/comfy/restart', {});
  is(typeof restart.error === 'string' && restart.error.length > 0,
     'restarting an engine with no install says so', restart.error);
  for (let i = noise.length - 1; i >= 0; i--) {
    if (noise[i].includes('/api/comfy/restart')) noise.splice(i, 1);
  }

  // The primary button used to read "Set up the engine" for every one of
  // these, and open a setup dialog that cannot restart an engine.
  const labels2 = await page.evaluate(() => ({
    ready: blocker({ ready: true }).label,
    fresh: blocker({ ready: false, setup_complete: false }).label,
    during: blocker({ ready: false, setup_running: true }).label,
    offline: blocker({ ready: false, setup_complete: true, comfy_online: false }).label,
    nodes: blocker({ ready: false, setup_complete: true, comfy_online: true,
                     nodes_ready: false }).label,
    models: blocker({ ready: false, setup_complete: true, comfy_online: true,
                      nodes_ready: true, missing_models: ['Qwen/x'] }).label,
  }));
  is(labels2.ready === 'Read the script' && labels2.fresh === 'Set up the engine'
       && labels2.during === 'Setting up…'
       && labels2.offline === 'Start the engine'
       && labels2.nodes === 'Restart the engine'
       && labels2.models === 'Download the models',
     'the primary button names the actual blocker', JSON.stringify(labels2));

  /* --------------------------------------------------------------- setup */
  await page.click('#btnReRun');
  await sleep(800);
  is((await page.$$('#pick-list label')).length >= 2, 'setup offers its routes');
  const labels = await page.$$('#pick-list label');
  await labels[labels.length - 1].click();      // connect to a ComfyUI I run
  await page.click('#btnRunSetup');
  let setupDone = false;
  for (let i = 0; i < 90; i++) {
    await sleep(700);
    if (await page.$eval('#setup-done', e => !e.hidden).catch(() => false)) {
      setupDone = true; break;
    }
    if (await page.$eval('#setup-retry', e => !e.hidden).catch(() => false)) break;
  }
  is(setupDone, 'setup runs to completion');
  const steps = await page.$$eval('#step-list .step',
    e => e.map(x => x.querySelector('.lbl').innerText.trim()));
  // Flask sorts the keys of a dict, which listed Check Python last — after the
  // step that starts the engine — on the one screen where order is the point.
  is(steps[0].startsWith('Check Python') && steps[steps.length - 1].startsWith('Start ComfyUI'),
     'setup steps are in the order they run', steps.join(' | '));
  is(steps.some(s => /speech nodes/i.test(s)),
     'the node step covers both engines, not just Qwen', steps[2]);
  // Escape deliberately will not dismiss a setup that has run — the panel is
  // closed with its own button, which is what a person would press.
  await page.click(setupDone ? '#setup-done' : '#setup-hide');
  await sleep(600);
  is(await page.$eval('#veil-setup', e => e.hidden), 'the setup panel closes');

  /* ------------------------------------------------------------ settings */
  await page.click('#navSettings');
  await sleep(600);
  await page.fill('#cfg-url', `  ${app.comfy.replace('http://', '')}/  `);
  await page.click('#btnSaveCfg');
  await sleep(1200);
  await page.click('#btnCloseCfg');
  await sleep(400);
  is((await app.api('/api/status')).config.comfy_url === app.comfy,
     'a typed address is normalised',
     (await app.api('/api/status')).config.comfy_url);

  /* ----------------------------------------------------------- shortcuts */
  await page.click('.nav[data-view="create"]');
  await sleep(600);
  await page.keyboard.press('Control+Enter');
  await sleep(1500);
  is(await page.$eval('#btnStop', e => !e.hidden), 'Ctrl+Enter starts a run');
  await page.click('#btnStop');
  await sleep(1500);
  is(await page.$eval('#btnStop', e => e.hidden), 'Stop resets the controls');

  await page.click('#btnResetOpts');
  await sleep(500);
  is(await page.inputValue('#pause-sl') === '0.5', 'reset restores the defaults');

  /* -------------------------------------------------------------- reload */
  // Compared against what is on screen now, not a literal: earlier checks
  // load a take back into the builder, which replaces the style text.
  const draft = {
    style: await page.inputValue('#style-input'),
    lines: await page.$$eval('#blocks textarea', e => e.map(x => x.value)),
  };
  await page.reload({ waitUntil: 'networkidle' });
  await sleep(1500);
  await page.click('.nav[data-view="create"]');
  await sleep(700);
  const after = {
    style: await page.inputValue('#style-input'),
    lines: await page.$$eval('#blocks textarea', e => e.map(x => x.value)),
  };
  is(after.style === draft.style
     && JSON.stringify(after.lines) === JSON.stringify(draft.lines),
     'the draft survives a reload', `${after.lines.length} lines restored`);
  heights = await page.$$eval('#blocks textarea',
    e => e.map(x => x.getBoundingClientRect().height));
  is(heights.every(h => h > 10), 'the restored lines are sized',
     heights.map(Math.round).join('/'));

  /* ---------------------------------------------------------- narrow view */
  await page.setViewportSize({ width: 420, height: 880 });
  await sleep(900);
  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  // A grid item is min-width:auto, so the rail refused to shrink and widened
  // the page instead of scrolling its own chips.
  is(overflow <= 2, 'no horizontal overflow at 420px', `${overflow}px`);
  await page.setViewportSize({ width: 1440, height: 900 });
  await sleep(500);

  /* ------------------------------------------------- faults are surfaced */
  await page.evaluate(() => { setTimeout(() => { throw new Error('planted'); }, 0); });
  await sleep(700);
  const toast = await page.$eval('#toast', e => ({ hidden: e.hidden, text: e.textContent }));
  is(!toast.hidden && /planted|went wrong/i.test(toast.text),
     'an unexpected script error reaches the user', toast.text.slice(0, 60));
  // That deliberate throw lands in the console too; not a real failure.
  for (let i = noise.length - 1; i >= 0; i--) {
    if (noise[i].includes('planted')) noise.splice(i, 1);
  }
} finally {
  await browser.close();
  await app.stop();
}

if (noise.length) {
  for (const n of [...new Set(noise)]) bad('unexpected browser output', n);
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.error('\nfailures:');
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}

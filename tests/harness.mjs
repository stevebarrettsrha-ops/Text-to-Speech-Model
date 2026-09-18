// Brings up everything the smoke test needs and takes it all down again.
//
// Nothing here touches the network, a GPU, or the data folder a real install
// uses: the two stand-ins under tests/ answer for ComfyUI and HuggingFace, and
// SCRIPT_BUILDER_DATA points the app at a throwaway directory.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';

export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sleep = ms => new Promise(r => setTimeout(r, ms));

function pythonCandidates() {
  return process.env.PYTHON ? [process.env.PYTHON]
    : ['python3', 'python', 'py'];
}

async function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function waitFor(url, label, tries = 120, procs = []) {
  for (let i = 0; i < tries; i++) {
    for (const { name, p, tail } of procs) {
      // A process that has already exited is never going to answer, and its
      // own output says why far better than a timeout does — a missing Flask
      // otherwise reads as "never came up" after a minute of polling.
      if (p.exitCode !== null && p.exitCode !== 0) {
        throw new Error(`${name} exited ${p.exitCode} before ${label} was `
                        + `ready:\n${tail.join('')}\n`
                        + 'If that names a missing module, the suite needs the '
                        + "app's own dependencies: pip install -r requirements.txt");
      }
    }
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch { /* not up yet */ }
    await sleep(500);
  }
  throw new Error(`${label} never came up at ${url}`);
}

export async function boot({ log = console.log } = {}) {
  const python = pythonCandidates()[0];
  const scratch = mkdtempSync(path.join(tmpdir(), 'sb-smoke-'));
  const dataDir = path.join(scratch, 'data');
  const modelsDir = path.join(scratch, 'models');
  const comfyRoot = path.join(scratch, 'comfy');
  for (const d of [dataDir, modelsDir, comfyRoot]) mkdirSync(d, { recursive: true });

  const comfyPort = await freePort();
  const hfPort = await freePort();
  const appPort = await freePort();
  const procs = [];

  function start(name, args, env) {
    const p = spawn(python, args, {
      cwd: REPO,
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const tail = [];
    const keep = chunk => {
      tail.push(String(chunk));
      if (tail.length > 40) tail.shift();
    };
    p.stdout.on('data', keep);
    p.stderr.on('data', keep);
    p.on('exit', code => {
      if (code !== 0 && code !== null) {
        log(`  [${name}] exited ${code}\n${tail.join('')}`);
      }
    });
    procs.push({ name, p, tail });
    return p;
  }

  start('mock-comfy', ['tests/mock_comfy.py', comfyRoot],
        { MOCK_COMFY_PORT: String(comfyPort) });
  start('mock-hf', ['tests/mock_hf.py'], { MOCK_HF_PORT: String(hfPort) });
  await waitFor(`http://127.0.0.1:${comfyPort}/system_stats`, 'mock ComfyUI',
                120, procs);
  await waitFor(`http://127.0.0.1:${hfPort}/api/models/Qwen/Qwen3-TTS-Tokenizer-12Hz/tree/main`,
                'mock HuggingFace', 120, procs);

  // Point the app at the stand-ins. setup_complete is true because the setup
  // run itself is covered by its own check below.
  writeFileSync(path.join(dataDir, 'config.json'), JSON.stringify({
    comfy_url: `http://127.0.0.1:${comfyPort}`,
    comfy_dir: '', models_dir: modelsDir, python: '', managed: false,
    auto_start_comfy: false, torch_index: '', hf_token: '',
    hf_endpoint: `http://127.0.0.1:${hfPort}`,
    hf_repo: 'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
    want_clone: true, want_17b: false, want_voicedesign: true,
    setup_complete: true,
  }, null, 2));

  start('app', ['server.py'], {
    SCRIPT_BUILDER_DATA: dataDir,
    SCRIPT_BUILDER_PORT: String(appPort),
    SCRIPT_BUILDER_NO_BROWSER: '1',
  });
  const base = `http://127.0.0.1:${appPort}`;
  await waitFor(base + '/', 'Script Builder', 120, procs);

  const api = async (p, opts) => {
    const r = await fetch(base + p, opts);
    return r.json();
  };
  const post = (p, body) => api(p, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  // Seed the voices the app needs before it will call itself ready.
  for (const repo of ['Qwen/Qwen3-TTS-Tokenizer-12Hz',
                      'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
                      'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                      'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign']) {
    await post('/api/hf/download', { repo });
    for (let i = 0; i < 40; i++) {
      const tasks = await api('/api/tasks');
      if (!tasks.some(t => t.state === 'running')) break;
      await sleep(250);
    }
  }
  for (let i = 0; i < 40; i++) {
    if ((await api('/api/status')).ready) break;
    await sleep(500);
  }

  async function runTake(payload) {
    const { job } = await post('/api/speak', payload);
    for (let i = 0; i < 120; i++) {
      const j = (await api('/api/jobs')).find(x => x.id === job);
      if (!j) break;
      if (j.status !== 'running') return j;
      await sleep(500);
    }
    return null;
  }

  // A take whose clips outlast a button press, for the transport checks.
  const LONG = [
    'This is a deliberately long first line so that the clip lasts several '
      + 'seconds and the transport buttons can be pressed while it plays.',
    'And here is an equally long second line, written at length purely so '
      + 'that playback does not finish before the test can press skip.',
    'Finally a third long line, so the take has a last position to skip to '
      + 'and then step back away from again.',
  ];
  await runTake({
    mode: 'multi', style: 'steady', title: 'Transport test',
    lines: LONG.map((text, i) => ({ speaker: (i % 2) + 1, text })),
    speakers: { 1: { name: 'A', kind: 'preset', speaker: 'Eric' },
                2: { name: 'B', kind: 'preset', speaker: 'Serena' } },
    model: '0.6B', attention: 'auto', unload: false, pause: 0.2,
    temperature: 0.9,
  });

  return {
    base, api, post, runTake, dataDir, modelsDir,
    comfy: `http://127.0.0.1:${comfyPort}`,
    hf: `http://127.0.0.1:${hfPort}`,
    async stop() {
      for (const { p } of procs) { try { p.kill('SIGTERM'); } catch { /* gone */ } }
      await sleep(400);
      for (const { p } of procs) { try { p.kill('SIGKILL'); } catch { /* gone */ } }
      rmSync(scratch, { recursive: true, force: true });
    },
  };
}

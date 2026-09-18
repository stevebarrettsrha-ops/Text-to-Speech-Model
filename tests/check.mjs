// The validation gate from CLAUDE.md, made runnable: `npm run check`.
//
// A missing function declaration in the inline script kills all interactivity
// silently, and py_compile catches the Python equivalent. Neither needs the app
// running, so this is the cheap check to run after every edit.
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.PYTHON || 'python3';
let failed = 0;

function step(name, fn) {
  const problem = fn();
  if (problem) {
    failed++;
    console.error(`FAIL ${name}\n${String(problem).trimEnd().replace(/^/gm, '     ')}`);
  } else {
    console.log(`ok   ${name}`);
  }
}

step('python modules compile', () => {
  const r = spawnSync(python, ['-m', 'py_compile', 'server.py', 'comfy.py',
                               'bootstrap.py', 'manager.py'],
                      { cwd: REPO, encoding: 'utf8' });
  return r.status === 0 ? null : (r.stderr || r.stdout || `exit ${r.status}`);
});

step('test helpers compile', () => {
  const r = spawnSync(python, ['-m', 'py_compile', 'tests/mock_comfy.py',
                               'tests/mock_hf.py', 'tests/test_units.py'],
                      { cwd: REPO, encoding: 'utf8' });
  return r.status === 0 ? null : (r.stderr || r.stdout || `exit ${r.status}`);
});

step('the inline script parses', () => {
  const html = readFileSync(path.join(REPO, 'web', 'index.html'), 'utf8');
  const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (!blocks.length) return 'no inline <script> found in web/index.html';
  const file = path.join(mkdtempSync(path.join(tmpdir(), 'sb-check-')), 'inline.js');
  writeFileSync(file, blocks.join('\n'));
  const r = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  return r.status === 0 ? null : (r.stderr || `exit ${r.status}`);
});

step('the page still has one script and no build step', () => {
  const html = readFileSync(path.join(REPO, 'web', 'index.html'), 'utf8');
  if (/<script[^>]+src=/.test(html)) {
    return 'web/index.html pulls in an external script — rule 1 says one file, '
         + 'no build step';
  }
  return null;
});

process.exit(failed ? 1 : 0);

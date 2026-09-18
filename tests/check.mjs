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

// The reason branch-protection.json is in the repo at all: GitHub accepts a
// required check whose name matches no job and then gates nothing, silently.
// Renaming a job in the workflow without editing the JSON would do exactly
// that, so the two are compared here rather than trusted.
//
// Parsed by hand, deliberately: this is the cheap gate and `node
// tests/check.mjs` has to keep working with nothing installed, so it cannot
// pull in a YAML library. It fails loudly if the file stops looking as
// expected rather than quietly finding nothing.
function workflowJobNames(yaml) {
  const lines = yaml.split('\n');
  const jobsAt = lines.findIndex(l => /^jobs:\s*$/.test(l));
  if (jobsAt < 0) return { error: 'no top-level `jobs:` in the workflow' };

  const names = [];
  let current = null;          // { name, matrix: {key: [values]} }
  const flush = () => {
    if (!current || !current.name) return;
    const slot = current.name.match(/\$\{\{\s*matrix\.(\w+)\s*\}\}/);
    if (!slot) { names.push(current.name); return; }
    const values = current.matrix[slot[1]];
    if (!values) {
      names.push({ error: `job name uses matrix.${slot[1]}, which the job `
                        + 'does not define' });
      return;
    }
    for (const v of values) names.push(current.name.replace(slot[0], v));
  };

  for (const line of lines.slice(jobsAt + 1)) {
    if (/^\S/.test(line) && line.trim()) break;          // left the jobs block
    if (/^ {2}[\w-]+:\s*$/.test(line)) { flush(); current = { name: null, matrix: {} }; continue; }
    if (!current) continue;
    const named = line.match(/^ {4}name:\s*(.+?)\s*$/);
    if (named) { current.name = named[1].replace(/^['"]|['"]$/g, ''); continue; }
    const matrixed = line.match(/^ {8}([\w-]+):\s*\[(.+)\]\s*$/);
    if (matrixed) {
      current.matrix[matrixed[1]] = matrixed[2].split(',')
        .map(v => v.trim().replace(/^['"]|['"]$/g, ''));
    }
  }
  flush();
  const broken = names.find(n => typeof n !== 'string');
  if (broken) return { error: broken.error };
  if (!names.length) return { error: 'found no job names — has the workflow changed shape?' };
  return { names };
}

step('required checks name jobs that exist', () => {
  const wf = path.join(REPO, '.github', 'workflows', 'test.yml');
  const rules = path.join(REPO, '.github', 'branch-protection.json');
  const { names, error } = workflowJobNames(readFileSync(wf, 'utf8'));
  if (error) return error;

  const wanted = JSON.parse(readFileSync(rules, 'utf8'))
    .required_status_checks.contexts;
  const missing = wanted.filter(c => !names.includes(c));
  const unguarded = names.filter(n => !wanted.includes(n));
  const problems = [];
  if (missing.length) {
    problems.push(`branch-protection.json requires checks no job produces: `
                  + `${missing.join(', ')}\n     jobs the workflow defines: `
                  + names.join(', '));
  }
  if (unguarded.length) {
    problems.push('jobs run but nothing requires them, so a failure would not '
                  + `block a merge: ${unguarded.join(', ')}`);
  }
  return problems.length ? problems.join('\n     ') : null;
});

process.exit(failed ? 1 : 0);

// Minimal static file server for local development and the smoke test.
// Mirrors GitHub Pages behaviour: directories resolve to index.html and
// unknown paths return 404.html with a 404 status.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.webmanifest': 'application/manifest+json',
};

export function startServer(root, port = 8080) {
  root = path.resolve(root);
  return new Promise((resolve, reject) => {
    const server = http.createServer(async (req, res) => {
      try {
        let pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
        if (pathname.endsWith('/')) pathname += 'index.html';
        const file = path.join(root, path.normalize(pathname));
        if (!file.startsWith(root + path.sep) && file !== root) {
          res.writeHead(403); res.end('Forbidden'); return;
        }
        let data;
        try {
          data = await readFile(file);
        } catch {
          res.writeHead(404, { 'Content-Type': TYPES['.html'], 'Cache-Control': 'no-store' });
          try { res.end(await readFile(path.join(root, '404.html'))); }
          catch { res.end('Not found'); }
          return;
        }
        res.writeHead(200, {
          'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream',
          'Cache-Control': 'no-store',
        });
        res.end(data);
      } catch (err) {
        res.writeHead(500); res.end(String(err));
      }
    });
    server.on('error', reject);
    server.listen(port, '127.0.0.1', () => {
      resolve({
        server,
        port: server.address().port,
        close: () => new Promise(done => server.close(done)),
      });
    });
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const { port } = await startServer(root, Number(process.env.PORT) || 8080);
  console.log(`Script Builder: http://127.0.0.1:${port}/`);
}

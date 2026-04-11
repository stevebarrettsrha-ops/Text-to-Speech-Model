const express = require('express');
const http = require('http');
const https = require('https');
const path = require('path');
const { URL } = require('url');

const app = express();
const PORT = process.env.PORT || 3000;

/* ── Serve static files ── */
app.use(express.static(path.join(__dirname)));

/* ── TTS proxy endpoint ── */
app.get('/api/tts', (req, res) => {
  const voice = req.query.voice || '';
  const text  = req.query.text  || '';

  if (!text) return res.status(400).send('Missing text parameter');

  /* Build the upstream URL */
  let upstream;
  if (voice && voice !== '_google') {
    upstream = 'https://api.streamelements.com/kappa/v2/speech?voice='
      + encodeURIComponent(voice) + '&text=' + encodeURIComponent(text);
  } else {
    upstream = 'https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q='
      + encodeURIComponent(text);
  }

  const parsed = new URL(upstream);
  const transport = parsed.protocol === 'https:' ? https : http;

  const proxyReq = transport.get(upstream, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      'Referer': parsed.origin + '/'
    }
  }, (proxyRes) => {
    if (proxyRes.statusCode < 200 || proxyRes.statusCode >= 300) {
      res.status(proxyRes.statusCode).send('Upstream TTS error');
      proxyRes.resume();
      return;
    }
    res.set('Content-Type', proxyRes.headers['content-type'] || 'audio/mpeg');
    res.set('Cache-Control', 'public, max-age=86400');
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    console.error('TTS proxy error:', err.message);
    if (!res.headersSent) res.status(502).send('TTS proxy failed');
  });

  proxyReq.setTimeout(15000, () => {
    proxyReq.destroy();
    if (!res.headersSent) res.status(504).send('TTS proxy timeout');
  });
});

app.listen(PORT, () => {
  console.log('Server running at http://localhost:' + PORT);
});

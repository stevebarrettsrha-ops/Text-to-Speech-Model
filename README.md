# Script Builder — Natural Voices

A free, browser-based text-to-speech script builder. Write single- or multi-speaker dialog, hear it read aloud with natural neural voices, and download the result as an MP3.

**Live app:** https://stevebarrettsrha-ops.github.io/Text-to-Speech-Model/

## Features

- Two speakers with their own names and voices, and a per-line speaker toggle
- 60+ natural voices: built-in neural voices in Chrome, Edge and Safari, plus cloud voices that work in any browser
- Speed, volume and pause-between-lines controls
- One-click example scripts (podcast, assistant, movie) and **Ctrl/Cmd + Enter** to run
- MP3 download, with an automatic WAV fallback if the MP3 encoder can't be loaded
- Autosaves your script in the browser, so a refresh never loses your work
- Works on phones and tablets

## Deployment

The site is published to GitHub Pages by [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml):

1. Every push and pull request runs the smoke test in headless Chromium.
2. On `main`, once the test passes, the site (`index.html`, `404.html`, `.nojekyll`) is published with the official `actions/deploy-pages` action.

The workflow enables Pages on its own. To check or set it by hand: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Getting a 404?

- Use the exact URL above. The account root, `https://stevebarrettsrha-ops.github.io/`, has no site and always returns 404. The app lives under `/Text-to-Speech-Model/`.
- Open the **Actions** tab and confirm the latest "Test & Deploy" run on `main` is green.
- Under **Settings → Pages**, confirm the Source is "GitHub Actions" (or "Deploy from a branch" with `main` and `/ (root)`).
- Wait a minute after a deploy, then hard-refresh (Ctrl + Shift + R).
- Any other path under the site shows a "Page not found" page and sends you back to the app automatically.

## Local development

```bash
npm install
npx playwright install chromium   # once, for the test
npm start                          # serves http://127.0.0.1:8080/
npm test                           # smoke test in headless Chromium
```

There is no build step: `index.html` is the whole app.

## Browser support and privacy

- Built-in neural voices need Chrome, Edge or Safari. Any modern browser can use the cloud voices.
- Cloud voices and the MP3 download send the script text to third-party text-to-speech services (StreamElements, and Google Translate as a fallback) to fetch the audio. Scripts are otherwise stored only in your browser's local storage.
- MP3 encoding uses [lamejs](https://github.com/zhuker/lamejs), loaded from jsDelivr.

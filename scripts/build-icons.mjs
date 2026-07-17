import fs from 'fs';
import path from 'path';

const needed = new Set([
  'ri-sun-line',
  'ri-moon-line',
  'ri-github-fill',
  'ri-telegram-fill',
  'ri-bilibili-fill',
  'ri-mail-fill',
  'ri-stack-line',
  'ri-vuejs-line',
  'ri-database-2-line',
  'ri-git-branch-line',
  'ri-terminal-line',
  'ri-code-s-slash-line',
  'ri-quill-pen-line',
  'ri-arrow-right-line',
  'ri-loader-4-line',
  'ri-music-2-line',
  'ri-album-line',
  'ri-skip-back-line',
  'ri-play-fill',
  'ri-pause-fill',
  'ri-skip-forward-line',
  'ri-arrow-right-up-line',
  'ri-calendar-line',
  'ri-volume-up-line',
  'ri-volume-down-line',
  'ri-volume-mute-line',
  'ri-repeat-line',
  'ri-shuffle-line',
  'ri-play-list-line',
  'ri-arrow-down-s-line',
]);

const src = path.resolve('node_modules/remixicon/fonts/remixicon.symbol.svg');
const outDir = path.resolve('images');
const outFile = path.join(outDir, 'icons.svg');

if (!fs.existsSync(src)) {
  console.error('remixicon.symbol.svg not found. Run npm install first.');
  process.exit(1);
}

const raw = fs.readFileSync(src, 'utf8');
const symbols = [];
const re = /<symbol\s+([^>]*?)id="([^"]+)"([^>]*)>([\s\S]*?)<\/symbol>/g;
let m;
while ((m = re.exec(raw)) !== null) {
  const id = m[2];
  if (needed.has(id)) {
    symbols.push(`<symbol ${m[1]}id="${id}"${m[3]}>${m[4]}</symbol>`);
  }
}

const missing = [...needed].filter((id) => !symbols.some((s) => s.includes(`id="${id}"`)));
if (missing.length) {
  console.error('Missing icons:', missing);
  process.exit(1);
}

if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

const sprite = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="0" height="0" style="display:none;">\n${symbols.join('\n')}\n</svg>`;
fs.writeFileSync(outFile, sprite, 'utf8');
console.log(`Built ${symbols.length} icons -> ${outFile}`);

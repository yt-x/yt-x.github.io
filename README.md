# HomePage

个人主页静态站点，面向 GitHub Pages 部署。页面包含个人资料、音乐播放器、技能展示、博客入口和项目展示，使用原生 HTML/CSS/JavaScript 组织组件，不依赖前端构建工具。

## 功能概览

- **个人主页**：顶部双栏布局，左侧为个人资料卡片，右侧为浅粉色音乐播放器。
- **音乐播放器**：读取 `data/playlist.json` 静态曲库，使用浏览器原生 `<audio>` 播放本地或 CDN 音频。
- **网易云辅助匹配**：GitHub Actions 定时更新 `data/now-playing.json`，前端用网易云记录匹配曲库默认选中歌曲。
- **动态组件加载**：`js/main.js` 通过 `fetch` 加载 `components/` 下的 HTML 片段。
- **主题切换**：支持浅色/深色模式，并用 `localStorage` 记住用户选择。
- **博客入口**：从外部 Hugo 博客 JSON/RSS 端点读取最新文章。

## 项目结构

```text
.
├── .github/workflows/
│   ├── deploy.yml              # GitHub Pages 部署
│   └── fetch-music.yml         # 定时更新网易云播放记录
├── components/                 # 页面组件片段
├── css/                        # 全局样式和组件样式
├── data/
│   ├── playlist.json           # 静态可播放曲库
│   └── now-playing.json        # 网易云记录生成的数据
├── images/                     # 图片资源
├── js/
│   ├── main.js                 # 组件加载、主题、博客渲染
│   ├── music-card.js           # 音乐播放器逻辑
│   └── theme.js                # 主题相关脚本
├── music/                      # 本地音频文件目录
├── scripts/
│   └── fetch-netease-music.js  # 网易云记录抓取脚本
├── index.html
├── MUSIC_CARD.md               # 音乐播放器详细配置
└── package.json
```

## 本地运行

项目无需安装依赖，直接启动静态服务器：

```powershell
python -m http.server 8000
```

浏览器打开：

```text
http://localhost:8000
```

也可以使用 `package.json` 中的脚本：

```powershell
npm run dev
```

## 音乐播放器配置

播放器只会播放 `data/playlist.json` 中带 `audioUrl` 的歌曲。示例：

```json
[
  {
    "id": "song-1",
    "neteaseId": "548885986",
    "title": "歌曲名",
    "artist": "歌手名",
    "cover": "images/cat.jpg",
    "audioUrl": "music/song-1.mp3",
    "url": "https://music.163.com/#/song?id=548885986",
    "lyrics": ""
  }
]
```

字段说明：

- `audioUrl`：必填，可填写仓库内音频路径，也可填写 CDN 或对象存储直链。
- `cover`：封面图路径，建议使用本地图片或稳定 CDN。
- `neteaseId`：可选，用于和 `data/now-playing.json` 的网易云歌曲记录匹配。
- `url`：可选，播放器中的“查看来源”链接。

更多说明见 `MUSIC_CARD.md`。

## 网易云记录更新

`.github/workflows/fetch-music.yml` 每 6 小时运行一次，调用 `scripts/fetch-netease-music.js` 更新 `data/now-playing.json`。

在 GitHub 仓库 Settings → Secrets and variables → Actions 中配置：

| Secret | 必填 | 说明 |
| --- | --- | --- |
| `NETEASE_UID` | 是 | 网易云用户 ID |
| `NETEASE_API_BASE` | 否 | 自建或托管的 NeteaseCloudMusicApi 地址 |
| `NETEASE_COOKIE` | 否 | 网易云登录 Cookie，用于后续更准确的最近播放 |

本地测试抓取脚本：

```powershell
$env:NETEASE_UID="你的网易云UID"
node scripts/fetch-netease-music.js
Get-Content -LiteralPath "data/now-playing.json"
```

## 部署

站点通过 `.github/workflows/deploy.yml` 部署到 GitHub Pages，发布内容为仓库根目录。

推送到 `main` 分支后，部署工作流会上传当前静态文件并发布。

## 验证命令

```powershell
node --check js/music-card.js
node --check scripts/fetch-netease-music.js
node -e "const fs=require('fs'); JSON.parse(fs.readFileSync('data/playlist.json','utf8')); JSON.parse(fs.readFileSync('data/now-playing.json','utf8')); console.log('json ok')"
git diff --check
```

## 待开发 TODO

- **接入网易云真实音频直链**：在已有 `NETEASE_API_BASE` 和 `NETEASE_COOKIE` 预留基础上，调用 NeteaseCloudMusicApi 的音频 URL 接口，自动为 `now-playing.json` 或曲库补充 `audioUrl`。
- **网易云最近播放升级**：优先使用 `/record/recent/song?limit=1` 获取真正最近播放；Cookie 不可用时回退到公开一周播放记录。
- **曲库自动匹配增强**：从网易云记录提取歌曲 ID、歌名、歌手，优先按 `neteaseId` 匹配，失败时再做歌名/歌手模糊匹配。
- **歌词展示**：利用 `playlist.json` 的 `lyrics` 字段，增加可展开歌词或同步歌词区域。
- **播放体验优化**：增加音量控制、播放模式、错误提示和不可播放歌曲自动跳过。
- **音频资源治理**：评估将大体积 FLAC 转为 MP3/M4A，或迁移到 CDN/对象存储，减少 GitHub Pages 仓库体积和本地开发连接重置日志。


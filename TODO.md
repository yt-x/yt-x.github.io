# TODO.md — 可追踪任务清单（跨会话交接唯一事实源）

> **目的**：让 AI 在会话结束、新开会话、切换模型后都能无缝续作。
>
> **AI 使用规则**：
> 1. 会话开始：先读「当前状态快照」，再从「待办」领取任务。
> 2. 完成任务：从「待办」移除，追加到「已完成」（日期 + 一句话结果 + 落点文件）。
> 3. 新增需求：写入「待办」，必须含【背景】【落点】【验收标准】三要素。
> 4. 踩坑不写这里，写 `CONSTRAINTS.md`；架构与机制不抄这里，读 `AGENTS.md`。

## 当前状态快照（2026-07-17）

- 页面结构：名片卡（全宽）→ 音乐 slim 横条（66px）→ EXHIBIT 01 技术栈 → EXHIBIT 02 博客 → EXHIBIT 03 项目 → Colophon 页脚。
- 动效齐备：入场交错 reveal（delay 1-6）、悬停微交互、播放均衡器、theme-spin、雨滴涟漪背景、身份词轮播（3 词/4s）、顶部阅读进度发线。
- 概念闭环：开馆（入场）→ 看展（EXHIBIT 编号）→ 导览（进度发线）→ 页码（Colophon）。
- 音乐播放器：slim bar 形态，歌名/进度/状态 + 7 图标按钮，音量=图标静音切换，播放列表可展开，网易云匹配逻辑在线。
- 已知未决：博客 feed 404（见 P1-1）；无 favicon（见 P3-2）。

## 待办

### P1 — 影响线上功能

- [ ] **P1-1 修复博客 feed 404**
  - 【背景】`js/main.js` 的 `loadLatestBlog()` 请求 `https://sgldbhxs.top/hugoblog/posts/index.json` 返回 404，博客卡长期显示错误态。Console 报错见 CONSTRAINTS.md C13。
  - 【落点】先确认 Hugo 博客侧 index.json 是否生成/路径是否变更；前端在 `js/main.js`。
  - 【验收】线上环境博客卡显示真实最新文章；或端点确认废弃后，前端改为展示静态兜底文案且不报 Console 错误。

### P2 — 网易云与音乐增强（承接 README 原 TODO）

- [ ] **P2-1 接入网易云真实音频直链**
  - 【背景】`NETEASE_API_BASE` / `NETEASE_COOKIE` 已在 workflow secrets 预留；曲库 `audioUrl` 目前依赖仓库内音频文件，体积大。
  - 【落点】`scripts/fetch-netease-music.js` 调 NeteaseCloudMusicApi 音频 URL 接口，为 `data/now-playing.json` 或 playlist 补充 `audioUrl`。
  - 【验收】`node scripts/fetch-netease-music.js` 本地跑通；至少一首歌无本地文件也可播放。
- [ ] **P2-2 最近播放升级**
  - 【背景】现用公开一周记录近似"正在听"。
  - 【落点】优先 `/record/recent/song?limit=1`；Cookie 不可用时回退现逻辑。
  - 【验收】now-playing.json 反映真实最近一首；无 Cookie 环境不报错回退。
- [ ] **P2-3 曲库自动匹配增强**
  - 【落点】`js/music-card.js` `findInitialSongIndex`：neteaseId 优先匹配，失败再歌名/歌手模糊匹配。
  - 【验收】构造 neteaseId 不一致但歌名一致的用例能选中；都不匹配时回退第一首。
- [ ] **P2-4 歌词展示**
  - 【落点】`data/playlist.json` 的 `lyrics` 字段 + 音乐条展开区（参考播放列表的展开模式）。
  - 【验收】有 lyrics 的歌曲可展开查看；无 lyrics 不显示入口；reduced-motion 与暗色模式达标。
- [ ] **P2-5 音频资源治理**
  - 【背景】FLAC 文件大，拖慢 Pages 仓库与本地克隆。
  - 【落点】评估转 MP3/M4A 或迁移 CDN/对象存储；改动涉及 `music/` 与 `data/playlist.json`。
  - 【验收】仓库体积明显下降且全部歌曲可播。

### P3 — 锦上添花

- [ ] **P3-1 播放体验补齐**：错误提示文案化（音频加载失败时在状态栏说明）、不可播放歌曲自动跳过。验收：坏链歌曲不卡死、状态栏有可读提示。
- [ ] **P3-2 添加 favicon**：消除既有 404（CONSTRAINTS.md C13），用水墨风字母标或小印章 SVG。

## 已评估未采纳（备查，重启需先讨论）

- **开馆帷幕（预加载幕布）**：源自 glukhovsky.com 借鉴评估。静态站无真实加载需求，若做须 <1s 且 localStorage 每日仅一次。2026-07-16 评估暂缓。
- **大标题负字距**：glukhovsky 手法（Noto Serif 300 + -2.1px）。当前中文标题用 tracking-widest 更合适；仅在未来新增英文展示大字时启用。
- **全屏 Burger 菜单**：单页站点无导航需求，明确不做。

## 已完成

| 日期 | 事项 | 落点 |
| --- | --- | --- |
| 2026-07-16 | 全站动效系统：入场交错 reveal、悬停微交互、均衡器、theme-spin、reduced-motion 守卫 | css/styles.css、css/components.css、index.html、js/main.js、components/music.html |
| 2026-07-16 | 雨滴涟漪背景特效（含修复 Canvas 负半径 IndexSizeError，见 C5） | js/rain.js（新建）、index.html |
| 2026-07-16 | 雨滴增强：蔟发 2-4 滴、涟漪加深加粗、入水水花点 | js/rain.js |
| 2026-07-16 | glukhovsky 借鉴三项：身份词轮播、阅读进度发线、EXHIBIT 编号 + Colophon | js/main.js、css/styles.css、index.html |
| 2026-07-17 | 音乐卡 slim bar 重构：垂直排布、去封面、控制图标化、音量改静音切换；清理约 250 行死 CSS | components/music.html、css/components.css、js/music-card.js、index.html |
| 2026-07-17 | 建立 AGENTS.md / CONSTRAINTS.md / TODO.md 协作文件体系 | 仓库根目录 |

## 交接要点

- 读文件顺序：`AGENTS.md`（机制）→ `CONSTRAINTS.md`（坑）→ 本文件。
- 改 JS 后必跑 `node --check`；改样式后必跑 `npm run build`；交付前浏览器实测双主题。
- 排查"行为没变化"先清浏览器缓存（C4）；本地预览必须起服务器（C1）。
- 本地验证服务器：`npm start` → http://localhost:8000。

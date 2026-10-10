# TODO.md — HomePage 当前状态与待办

> 读序：本文件（状态）→ `AGENTS.md`（机制）→ `CONSTRAINTS.md`（坑）→ `decisions/`（取舍因果）。
> 本文件只放状态、下一步与阻塞。

## 功能进度

| 功能 | 状态 | 证据 |
| --- | --- | --- |
| 水墨博物馆风页面与动效系统 | 完成 | `css/styles.css` |
| 音乐 slim 播放器 | 完成 | `js/music-card.js` |
| 博客卡（JSON 优先、RSS 兜底） | 完成 | `js/main.js` |
| 雨滴涟漪背景 | 完成 | `js/rain.js` |
| 浅色/暗色主题切换 | 完成 | `js/main.js` |
| 站点图标（favicon + apple-touch-icon） | 完成 | `favicon.ico` |
| 网易云最近播放同步 | 进行中 | `data/now-playing.json` |
| 歌词展示 | 未开始 | |
| 播放错误提示与坏链跳过 | 未开始 | |
| 音频体积治理 | 未开始 | |

## 当前状态

- 结构：名片卡 → 音乐横条（66px）→ EXHIBIT 01 技术栈 → 02 博客 → 03 项目 → Colophon。
- 入场 reveal、悬停微交互、均衡器、雨滴背景、身份词轮播齐备；暗色与 reduced-motion 均有守卫。
- 博客卡读 `https://sgldbhxs.top/hugoblog/posts/index.json`，失败回退 `.../posts/index.xml`。
- 站点图标为白文印 S（洋红底纸白字），由 `scripts/make-favicon.py` 生成，可重跑复现。
- `data/now-playing.json` 的内容是 `null`，网易云匹配拿不到真实输入。

## 下一步

1. 网易云最近播放同步产出有效数据——`data/now-playing.json` 内容非 null，且播放器能据此选中默认曲
2. 网易云音频直链接入——至少一首歌在没有本地音频文件的情况下可播放
3. 曲库匹配增强——neteaseId 不一致但歌名相同的用例能选中，全不匹配时回退第一首
4. 歌词展示——有 `lyrics` 的歌曲可展开查看，无 `lyrics` 不显示入口
5. 播放错误提示与坏链跳过——坏链歌曲不卡死，状态栏给出可读提示
6. 音频体积治理——`music/` 转码后仓库跟踪体积明显下降且全部歌曲可播放

## 阻塞

- 网易云同步的定位依赖 GitHub 仓库 secrets（`NETEASE_UID` 等）的配置状态确认。

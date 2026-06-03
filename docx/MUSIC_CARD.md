# 音乐播放器配置说明

## 当前架构

音乐区域是一个可播放组件，前端读取 `data/playlist.json` 作为静态曲库，并使用浏览器原生 `<audio>` 播放音频。

`data/now-playing.json` 仍由网易云低维护版脚本更新，用于决定默认选中哪首歌；如果网易云记录无法匹配本地曲库，会默认播放曲库第一首。

## 配置本地曲库

在 `data/playlist.json` 中配置歌曲：

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

`audioUrl` 是必填项，可以是仓库内的本地 MP3，也可以是 CDN 或对象存储上的直链。

## 推荐目录

如果歌曲文件直接放在仓库中，建议使用：

```text
music/
├── song-1.mp3
└── song-2.mp3
```

然后在 `audioUrl` 中填写 `music/song-1.mp3`。

## 网易云记录辅助匹配

在 GitHub 仓库的 Settings → Secrets and variables → Actions 中新增：

| Secret | 说明 |
| --- | --- |
| `NETEASE_UID` | 网易云用户 ID |

工作流会更新 `data/now-playing.json`。前端会优先用 `neteaseId` 匹配曲库；没有 `neteaseId` 时，会用歌曲名匹配。

## 后续升级空间

如果以后要自动获取真实网易云播放音频，可以新增：

| Secret | 说明 |
| --- | --- |
| `NETEASE_API_BASE` | 自建或托管的 NeteaseCloudMusicApi 地址 |
| `NETEASE_COOKIE` | 网易云登录 Cookie |

当前播放器已经保留 `audioUrl` 字段，后续只需要让抓取脚本写入可播放音频直链即可。

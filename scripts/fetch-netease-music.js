const fs = require('fs/promises');
const path = require('path');

const OUTPUT_PATH = process.env.MUSIC_OUTPUT_PATH || path.join(process.cwd(), 'data', 'now-playing.json');
const NETEASE_UID = process.env.NETEASE_UID || '';
const NETEASE_COOKIE = process.env.NETEASE_COOKIE || '';
const NETEASE_API_BASE = (process.env.NETEASE_API_BASE || '').replace(/\/$/, '');

async function main() {
    await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });

    if (!NETEASE_UID) {
        await writeSong(null, '未配置 NETEASE_UID');
        return;
    }

    const song = await fetchNeteaseSong();
    await writeSong(song, song ? '更新成功' : '暂无可用歌曲');
}

async function fetchNeteaseSong() {
    const attempts = buildFetchAttempts();

    for (const attempt of attempts) {
        try {
            const response = await fetch(attempt.url, {
                headers: buildHeaders(attempt.needsCookie),
            });

            if (!response.ok) {
                console.warn(`网易云请求失败：${attempt.name} ${response.status}`);
                continue;
            }

            const payload = await response.json();
            const song = normalizeNeteasePayload(payload, attempt.source);
            if (song) return song;
        } catch (error) {
            console.warn(`网易云请求异常：${attempt.name}`, error.message);
        }
    }

    return null;
}

function buildFetchAttempts() {
    const attempts = [];

    if (NETEASE_API_BASE && NETEASE_COOKIE) {
        attempts.push({
            name: '代理最近播放',
            source: 'netease-recent',
            needsCookie: true,
            url: `${NETEASE_API_BASE}/record/recent/song?limit=1`,
        });
    }

    if (NETEASE_API_BASE) {
        attempts.push({
            name: '代理一周记录',
            source: 'netease-weekly',
            needsCookie: false,
            url: `${NETEASE_API_BASE}/user/record?uid=${encodeURIComponent(NETEASE_UID)}&type=1`,
        });
    }

    attempts.push({
        name: '公开一周记录',
        source: 'netease-weekly',
        needsCookie: false,
        url: `https://music.163.com/api/v1/play/record?uid=${encodeURIComponent(NETEASE_UID)}&type=1`,
    });

    return attempts;
}

function buildHeaders(needsCookie) {
    const headers = {
        Accept: 'application/json,text/plain,*/*',
        Referer: `https://music.163.com/user/home?id=${NETEASE_UID}`,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36',
    };

    if (needsCookie && NETEASE_COOKIE) {
        headers.Cookie = NETEASE_COOKIE;
    }

    return headers;
}

function normalizeNeteasePayload(payload, source) {
    const record = pickFirstRecord(payload);
    const rawSong = record?.song || record?.data || record;
    if (!rawSong || typeof rawSong !== 'object') return null;

    const album = rawSong.al || rawSong.album || {};
    const artists = rawSong.ar || rawSong.artists || [];
    const artistText = Array.isArray(artists)
        ? artists.map((artist) => artist.name).filter(Boolean).join(' / ')
        : artists.name || '';
    const songId = rawSong.id || rawSong.songId;

    return {
        source,
        provider: 'netease',
        title: rawSong.name || '未知歌曲',
        artist: artistText || '未知歌手',
        cover: album.picUrl || album.blurPicUrl || '',
        url: songId ? `https://music.163.com/#/song?id=${songId}` : `https://music.163.com/user/home?id=${NETEASE_UID}`,
        playing: false,
        statusText: source === 'netease-recent' ? '最近播放' : '一周常听',
        updatedAt: new Date().toISOString(),
    };
}

function pickFirstRecord(payload) {
    if (Array.isArray(payload?.data?.list)) return payload.data.list[0];
    if (Array.isArray(payload?.data)) return payload.data[0];
    if (Array.isArray(payload?.weekData)) return payload.weekData[0];
    if (Array.isArray(payload?.allData)) return payload.allData[0];
    if (Array.isArray(payload?.list)) return payload.list[0];
    return null;
}

async function writeSong(song, message) {
    await fs.writeFile(OUTPUT_PATH, `${JSON.stringify(song, null, 2)}\n`, 'utf8');
    console.log(`网易云音乐数据：${message}`);
}

main().catch(async (error) => {
    console.error('网易云音乐数据更新失败:', error);
    await writeSong(null, '更新失败');
    process.exitCode = 1;
});

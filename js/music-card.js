const MUSIC_CONFIG = {
    playlistUrl: 'data/playlist.json',
    neteaseDataUrl: 'data/now-playing.json',
};

const musicPlayerState = {
    playlist: [],
    currentIndex: 0,
    isSeeking: false,
    pendingAudioUrl: '',
    playMode: 'loop', // 'loop' | 'shuffle' | 'single'
    volume: 0.8,
    preMuteVolume: 0.8,
};

function initMusicCard() {
    const card = document.querySelector('.music-card');
    if (!card) return;

    setupMusicEvents();
    loadMusicPlayer();
}

async function loadMusicPlayer() {
    renderMusicLoading();

    try {
        const [playlist, neteaseSong] = await Promise.all([
            fetchPlaylist(),
            fetchNeteaseStaticSong(),
        ]);

        musicPlayerState.playlist = playlist.filter((song) => song.audioUrl);

        if (musicPlayerState.playlist.length === 0) {
            renderMusicEmpty('待配置');
            return;
        }

        musicPlayerState.currentIndex = findInitialSongIndex(musicPlayerState.playlist, neteaseSong);
        renderPlaylist();
        renderCurrentSong();
        applyVolume();
    } catch (error) {
        console.error('音乐播放器加载失败:', error);
        renderMusicEmpty('加载失败');
    }
}

async function fetchPlaylist() {
    const response = await fetch(MUSIC_CONFIG.playlistUrl);
    if (!response.ok) return [];

    const data = await response.json();
    if (!Array.isArray(data)) return [];

    return data.map(normalizePlaylistSong).filter(Boolean);
}

async function fetchNeteaseStaticSong() {
    try {
        const response = await fetch(MUSIC_CONFIG.neteaseDataUrl);
        if (!response.ok) return null;

        const song = await response.json();
        if (!song || typeof song !== 'object') return null;
        return song;
    } catch (error) {
        return null;
    }
}

function normalizePlaylistSong(song) {
    if (!song || typeof song !== 'object') return null;

    return {
        id: song.id || song.neteaseId || song.title,
        neteaseId: song.neteaseId || '',
        title: song.title || '未知歌曲',
        artist: song.artist || '未知歌手',
        cover: song.cover || 'images/cat.jpg',
        audioUrl: song.audioUrl || '',
        url: song.url || '',
        lyrics: song.lyrics || '',
    };
}

function findInitialSongIndex(playlist, neteaseSong) {
    if (!neteaseSong) return 0;

    const neteaseId = extractNeteaseId(neteaseSong.url);
    const matchedIndex = playlist.findIndex((song) => {
        return (neteaseId && song.neteaseId === neteaseId) ||
            normalizeText(song.title) === normalizeText(neteaseSong.title);
    });

    return matchedIndex >= 0 ? matchedIndex : 0;
}

function extractNeteaseId(url) {
    if (!url) return '';
    const match = String(url).match(/[?&]id=(\d+)/);
    return match?.[1] || '';
}

function normalizeText(text) {
    return String(text || '').trim().toLowerCase();
}

function setupMusicEvents() {
    const audio = getMusicElement('music-audio');
    const playButton = getMusicElement('music-play');
    const prevButton = getMusicElement('music-prev');
    const nextButton = getMusicElement('music-next');
    const progress = getMusicElement('music-progress');
    const modeButton = getMusicElement('music-mode');
    const volumeButton = getMusicElement('music-volume-icon');
    const playlistToggle = getMusicElement('music-playlist-toggle');

    if (!audio || !playButton || !prevButton || !nextButton || !progress) return;

    playButton.addEventListener('click', toggleMusicPlayback);
    prevButton.addEventListener('click', playPreviousSong);
    nextButton.addEventListener('click', playNextSong);
    modeButton?.addEventListener('click', togglePlayMode);
    playlistToggle?.addEventListener('click', togglePlaylist);

    // 音量图标：点击切换静音
    volumeButton?.addEventListener('click', toggleMute);

    audio.addEventListener('loadedmetadata', updateMusicDuration);
    audio.addEventListener('timeupdate', updateMusicProgress);
    audio.addEventListener('play', () => setMusicPlaying(true));
    audio.addEventListener('pause', () => setMusicPlaying(false));
    audio.addEventListener('ended', handleSongEnded);
    audio.addEventListener('error', handleAudioError);

    audio.addEventListener('volumechange', () => {
        musicPlayerState.volume = audio.volume;
        updateVolumeIcon();
    });

    progress.addEventListener('input', () => {
        musicPlayerState.isSeeking = true;
        updateDisplayedCurrentTime(progress.value);
    });

    progress.addEventListener('change', () => {
        seekMusic(Number(progress.value));
        musicPlayerState.isSeeking = false;
    });
}

function handleAudioError() {
    const audio = getMusicElement('music-audio');
    console.error('音频加载失败:', audio?.error);
    // 自动跳到下一首，避免卡死
    playNextSong();
}

function handleSongEnded() {
    const { playMode } = musicPlayerState;
    if (playMode === 'single') {
        const audio = getMusicElement('music-audio');
        if (audio) {
            audio.currentTime = 0;
            audio.play().catch(() => renderMusicEmpty('播放失败'));
        }
    } else if (playMode === 'shuffle') {
        playRandomSong();
    } else {
        playNextSong();
    }
}

function playRandomSong() {
    const { playlist, currentIndex } = musicPlayerState;
    if (playlist.length <= 1) {
        playNextSong();
        return;
    }
    let nextIndex;
    do {
        nextIndex = Math.floor(Math.random() * playlist.length);
    } while (nextIndex === currentIndex);
    musicPlayerState.currentIndex = nextIndex;
    renderCurrentSong();
}

function togglePlayMode() {
    const modes = ['loop', 'shuffle', 'single'];
    const current = musicPlayerState.playMode;
    const next = modes[(modes.indexOf(current) + 1) % modes.length];
    musicPlayerState.playMode = next;
    renderPlayMode();
}

function renderPlayMode() {
    const button = getMusicElement('music-mode');
    const icon = button?.querySelector('use');
    if (!button || !icon) return;

    const map = {
        loop: { icon: 'ri-repeat-line', label: '列表循环' },
        shuffle: { icon: 'ri-shuffle-line', label: '随机播放' },
        single: { icon: 'ri-repeat-line', label: '单曲循环' },
    };

    // 单曲循环图标用 repeat-line + 数字标记不够直观，仍用 repeat-line
    const config = map[musicPlayerState.playMode];
    icon.setAttribute('href', `images/icons.svg#${config.icon}`);
    button.setAttribute('aria-label', config.label);
    button.setAttribute('title', config.label);
}

function togglePlaylist() {
    const panel = getMusicElement('music-playlist');
    const icon = getMusicElement('music-playlist-toggle')?.querySelector('use');
    if (!panel) return;

    const isOpen = panel.classList.toggle('hidden');
    if (icon) {
        icon.setAttribute('href', isOpen ? 'images/icons.svg#ri-playlist-line' : 'images/icons.svg#ri-arrow-down-s-line');
    }
}

function setVolume(value) {
    const audio = getMusicElement('music-audio');
    if (!audio) return;
    const clamped = Math.max(0, Math.min(1, value));
    audio.volume = clamped;
    musicPlayerState.volume = clamped;
    updateVolumeIcon();
}

function applyVolume() {
    const audio = getMusicElement('music-audio');
    if (audio) audio.volume = musicPlayerState.volume;
    updateVolumeIcon();
}

// 点击音量图标：切换静音 / 恢复
function toggleMute() {
    const audio = getMusicElement('music-audio');
    if (!audio) return;

    if (audio.volume > 0) {
        musicPlayerState.preMuteVolume = audio.volume;
        audio.volume = 0;
    } else {
        audio.volume = musicPlayerState.preMuteVolume || 0.8;
    }
    musicPlayerState.volume = audio.volume;
    updateVolumeIcon();
}

function updateVolumeIcon() {
    const icon = getMusicElement('music-volume-icon')?.querySelector('use');
    if (!icon) return;

    const value = musicPlayerState.volume;
    let name = 'ri-volume-up-line';
    if (value === 0) name = 'ri-volume-mute-line';
    else if (value < 0.4) name = 'ri-volume-down-line';
    icon.setAttribute('href', `images/icons.svg#${name}`);
}

function renderCurrentSong() {
    const song = musicPlayerState.playlist[musicPlayerState.currentIndex];
    const audio = getMusicElement('music-audio');
    if (!song || !audio) {
        renderMusicEmpty('暂无歌曲');
        return;
    }

    switchMusicState('music-player');
    setMusicStatus('准备播放');

    const titleEl = getMusicElement('music-title');
    if (titleEl) {
        titleEl.textContent = song.title;
        titleEl.setAttribute('title', song.title + ' - ' + song.artist);
    }
    setSourceLink(song.url);
    highlightPlaylistItem();

    audio.pause();
    audio.removeAttribute('src');
    musicPlayerState.pendingAudioUrl = song.audioUrl;
    resetMusicProgress();
    setMusicPlaying(false);
}

async function toggleMusicPlayback() {
    const audio = getMusicElement('music-audio');
    if (!audio) return;

    if (!audio.src && musicPlayerState.pendingAudioUrl) {
        audio.src = musicPlayerState.pendingAudioUrl;
    }

    if (!audio.src) return;

    if (audio.paused) {
        try {
            await audio.play();
        } catch (error) {
            console.error('音乐播放失败:', error);
            renderMusicEmpty('播放失败');
        }
    } else {
        audio.pause();
    }
}

function playPreviousSong() {
    if (musicPlayerState.playlist.length === 0) return;
    musicPlayerState.currentIndex = (musicPlayerState.currentIndex - 1 + musicPlayerState.playlist.length) % musicPlayerState.playlist.length;
    renderCurrentSong();
}

function playNextSong() {
    if (musicPlayerState.playlist.length === 0) return;
    musicPlayerState.currentIndex = (musicPlayerState.currentIndex + 1) % musicPlayerState.playlist.length;
    renderCurrentSong();
}

function updateMusicDuration() {
    const audio = getMusicElement('music-audio');
    if (!audio) return;

    setText('music-duration', formatMusicTime(audio.duration));
}

function updateMusicProgress() {
    const audio = getMusicElement('music-audio');
    const progress = getMusicElement('music-progress');
    if (!audio || !progress || musicPlayerState.isSeeking) return;

    const percent = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
    progress.value = String(percent);
    updateDisplayedCurrentTime(percent);
}

function updateDisplayedCurrentTime(percent) {
    const audio = getMusicElement('music-audio');
    if (!audio?.duration) {
        setText('music-current-time', '0:00');
        return;
    }

    setText('music-current-time', formatMusicTime((Number(percent) / 100) * audio.duration));
}

function seekMusic(percent) {
    const audio = getMusicElement('music-audio');
    if (!audio?.duration) return;

    audio.currentTime = (percent / 100) * audio.duration;
}

function resetMusicProgress() {
    const progress = getMusicElement('music-progress');
    if (progress) progress.value = '0';
    setText('music-current-time', '0:00');
    setText('music-duration', '0:00');
}

function setMusicPlaying(isPlaying) {
    const player = getMusicElement('music-player');
    const playButton = getMusicElement('music-play');
    const icon = playButton?.querySelector('use');

    player?.classList.toggle('is-playing', isPlaying);
    setMusicStatus(isPlaying ? '正在播放' : '准备播放');

    if (playButton) playButton.setAttribute('aria-label', isPlaying ? '暂停' : '播放');
    if (icon) icon.setAttribute('href', isPlaying ? 'images/icons.svg#ri-pause-fill' : 'images/icons.svg#ri-play-fill');
}

function setSourceLink(url) {
    const link = getMusicElement('music-source-link');
    if (!link) return;

    if (url) {
        link.href = url;
        link.classList.remove('hidden');
    } else {
        link.href = '#';
        link.classList.add('hidden');
    }
}

function renderMusicLoading() {
    setMusicStatus('加载中');
    switchMusicState('music-loading');
}

function renderMusicEmpty(statusText) {
    setMusicStatus(statusText);
    switchMusicState('music-empty');
}

function setMusicStatus(text) {
    setText('music-status', text);
}

function switchMusicState(activeId) {
    ['music-loading', 'music-empty', 'music-player'].forEach((id) => {
        const element = getMusicElement(id);
        if (!element) return;
        element.classList.toggle('hidden', id !== activeId);
    });
}

function renderPlaylist() {
    const panel = getMusicElement('music-playlist');
    if (!panel) return;

    const list = panel.querySelector('ol');
    if (!list) return;

    list.innerHTML = musicPlayerState.playlist.map((song, index) => `
        <li data-index="${index}" class="music-playlist__item">
            <span class="music-playlist__number">${String(index + 1).padStart(2, '0')}</span>
            <span class="music-playlist__info">
                <span class="music-playlist__title">${escapeHtml(song.title)}</span>
                <span class="music-playlist__artist">${escapeHtml(song.artist)}</span>
            </span>
        </li>
    `).join('');

    list.querySelectorAll('.music-playlist__item').forEach((item) => {
        item.addEventListener('click', () => {
            const index = Number(item.dataset.index);
            if (Number.isNaN(index)) return;
            musicPlayerState.currentIndex = index;
            renderCurrentSong();
            if (musicPlayerState.playMode === 'shuffle') {
                musicPlayerState.playMode = 'loop';
                renderPlayMode();
            }
        });
    });

    highlightPlaylistItem();
}

function highlightPlaylistItem() {
    const panel = getMusicElement('music-playlist');
    if (!panel) return;

    panel.querySelectorAll('.music-playlist__item').forEach((item, index) => {
        item.classList.toggle('is-active', index === musicPlayerState.currentIndex);
    });
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function setText(id, text) {
    const element = getMusicElement(id);
    if (element) element.textContent = text;
}

function setImage(id, src, alt) {
    const image = getMusicElement(id);
    if (!image) return;

    image.src = src;
    image.alt = alt;
    image.hidden = !src;
}

function formatMusicTime(seconds) {
    if (!Number.isFinite(seconds)) return '0:00';

    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${String(remainingSeconds).padStart(2, '0')}`;
}

function getMusicElement(id) {
    return document.getElementById(id);
}

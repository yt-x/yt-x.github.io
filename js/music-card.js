const MUSIC_CONFIG = {
    playlistUrl: 'data/playlist.json',
    neteaseDataUrl: 'data/now-playing.json',
};

const musicPlayerState = {
    playlist: [],
    currentIndex: 0,
    isSeeking: false,
    pendingAudioUrl: '',
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
        renderCurrentSong();
    } catch (error) {
        console.error('音乐播放器加载失败:', error);
        renderMusicEmpty('加载失败');
    }
}

async function fetchPlaylist() {
    const response = await fetch(`${MUSIC_CONFIG.playlistUrl}?v=${Date.now()}`);
    if (!response.ok) return [];

    const data = await response.json();
    if (!Array.isArray(data)) return [];

    return data.map(normalizePlaylistSong).filter(Boolean);
}

async function fetchNeteaseStaticSong() {
    try {
        const response = await fetch(`${MUSIC_CONFIG.neteaseDataUrl}?v=${Date.now()}`);
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

    if (!audio || !playButton || !prevButton || !nextButton || !progress) return;

    playButton.addEventListener('click', toggleMusicPlayback);
    prevButton.addEventListener('click', playPreviousSong);
    nextButton.addEventListener('click', playNextSong);

    audio.addEventListener('loadedmetadata', updateMusicDuration);
    audio.addEventListener('timeupdate', updateMusicProgress);
    audio.addEventListener('play', () => setMusicPlaying(true));
    audio.addEventListener('pause', () => setMusicPlaying(false));
    audio.addEventListener('ended', playNextSong);
    audio.addEventListener('error', () => renderMusicEmpty('音频不可用'));

    progress.addEventListener('input', () => {
        musicPlayerState.isSeeking = true;
        updateDisplayedCurrentTime(progress.value);
    });

    progress.addEventListener('change', () => {
        seekMusic(Number(progress.value));
        musicPlayerState.isSeeking = false;
    });
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

    setText('music-title', song.title);
    setText('music-artist', song.artist);
    setImage('music-cover', song.cover, `${song.title} 的歌曲封面`);
    setSourceLink(song.url);

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
    const icon = playButton?.querySelector('i');

    player?.classList.toggle('is-playing', isPlaying);
    setMusicStatus(isPlaying ? '正在播放' : '准备播放');

    if (playButton) playButton.setAttribute('aria-label', isPlaying ? '暂停' : '播放');
    if (icon) icon.className = isPlaying ? 'ri-pause-fill' : 'ri-play-fill';
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

// Synced, gain-matched A/B/C player. Runs as the js_on_load body of a gr.HTML component:
// `element`, `props`, `watch` and `server` come from Gradio.
//
// Every source plays at once from one clock, each through its own gain node, and only the
// selected one is audible. Switching is a short gain crossfade, so the playhead never moves
// and, with loudness matching on, neither does the level.

const FADE_S = 0.015;
const LETTERS = "ABCDEFGHI";

class ABPlayer {
  constructor(root) {
    this.root = root;
    this.ctx = null;
    this.sources = [];
    this.nodes = [];
    this.active = 0;
    this.playing = false;
    this.offset = 0;
    this.startedAt = 0;
    this.duration = 0;
    this.matched = true;
    this.loop = null;
    this.blind = null;
    this.loadId = 0;
    this.attach(root);
    document.addEventListener("keydown", (e) => this.onKey(e));
    const tick = () => {
      this.render();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // Gradio re-renders the template when the value changes, so the controls are re-bound then.
  attach(root) {
    this.root = root;
    this.q = (selector) => root.querySelector(selector);
    this.q(".mb-play").onclick = () => this.toggle();
    this.q(".mb-seek").oninput = (e) => this.seek((e.target.value / 1000) * this.duration);
    this.q(".mb-match").onchange = (e) => this.setMatched(e.target.checked);
    this.q(".mb-match").checked = this.matched;
    this.q(".mb-loop-in").onclick = () => this.setLoop(this.position(), this.loop ? this.loop[1] : null);
    this.q(".mb-loop-out").onclick = () => this.setLoop(this.loop ? this.loop[0] : 0, this.position());
    this.q(".mb-loop-clear").onclick = () => this.setLoop(null, null);
    this.q(".mb-blind").onclick = () => (this.blind ? this.endBlind() : this.startBlind());
    this.q(".mb-save").onclick = () => this.reveal();
  }

  onKey(e) {
    // Keys go to the player unless the user is typing somewhere. The player's own checkbox
    // and seek slider are inputs too, and must not swallow the shortcuts once clicked.
    const tag = (e.target.tagName || "").toLowerCase();
    const typing =
      ["textarea", "select"].includes(tag) ||
      (tag === "input" && !["checkbox", "range", "button"].includes(e.target.type)) ||
      e.target.isContentEditable;
    if (!this.root.offsetParent || typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === "Space") {
      e.preventDefault();
      this.toggle();
    } else if (/^[1-9]$/.test(e.key) && Number(e.key) <= this.sources.length) {
      e.preventDefault();
      this.select(this.order()[Number(e.key) - 1]);
    }
  }

  async load(config) {
    const loadId = ++this.loadId;
    this.stop();
    this.blind = null;
    this.loop = null;
    this.offset = 0;
    this.active = 0;
    this.track = config.track || null;
    this.sources = [];
    this.duration = 0;
    this.status(config.sources && config.sources.length ? "Loading audio…" : config.message || "");
    this.renderSources();
    if (!config.sources || !config.sources.length) return;
    this.ctx = this.ctx || new AudioContext();
    try {
      const loaded = await Promise.all(
        config.sources.map(async (source) => {
          const response = await fetch(source.url);
          if (!response.ok) throw new Error(`${source.label}: HTTP ${response.status}`);
          const buffer = await this.ctx.decodeAudioData(await response.arrayBuffer());
          return { ...source, buffer };
        })
      );
      if (loadId !== this.loadId) return;
      this.sources = loaded;
      this.duration = Math.max(...loaded.map((s) => s.buffer.duration));
      this.active = Math.max(0, loaded.findIndex((s) => s.id === config.selected));
      this.status("");
    } catch (error) {
      if (loadId === this.loadId) this.status(`Could not load audio: ${error.message}`);
    }
    this.renderSources();
  }

  order() {
    return this.blind ? this.blind.order : this.sources.map((_, i) => i);
  }

  gainFor(index) {
    const source = this.sources[index];
    let db = source.gain_db || 0;
    if (this.matched || this.blind) {
      db += Math.min(...this.sources.map((s) => s.lufs)) - source.lufs;
    }
    return Math.pow(10, db / 20);
  }

  applyGains(immediate) {
    if (!this.nodes.length) return;
    const now = this.ctx.currentTime;
    this.nodes.forEach((node, index) => {
      const target = index === this.active ? this.gainFor(index) : 0;
      node.gain.gain.cancelScheduledValues(now);
      if (immediate) {
        node.gain.gain.setValueAtTime(target, now);
      } else {
        node.gain.gain.setValueAtTime(node.gain.gain.value, now);
        node.gain.gain.linearRampToValueAtTime(target, now + FADE_S);
      }
    });
  }

  position() {
    if (!this.playing) return this.offset;
    let pos = this.offset + (this.ctx.currentTime - this.startedAt);
    if (this.loop && pos >= this.loop[1]) {
      pos = this.loop[0] + ((pos - this.loop[0]) % (this.loop[1] - this.loop[0]));
    }
    return Math.min(pos, this.duration);
  }

  start() {
    this.nodes = this.sources.map((source) => {
      const node = this.ctx.createBufferSource();
      const gain = this.ctx.createGain();
      node.buffer = source.buffer;
      if (this.loop) {
        node.loop = true;
        node.loopStart = this.loop[0];
        node.loopEnd = this.loop[1];
      }
      node.connect(gain).connect(this.ctx.destination);
      return { node, gain };
    });
    this.applyGains(true);
    const when = this.ctx.currentTime + 0.02;
    this.nodes.forEach(({ node }) => node.start(when, Math.min(this.offset, node.buffer.duration)));
    this.startedAt = when;
    this.playing = true;
  }

  halt() {
    this.nodes.forEach(({ node, gain }) => {
      try {
        node.stop();
      } catch (_) {}
      node.disconnect();
      gain.disconnect();
    });
    this.nodes = [];
    this.playing = false;
  }

  async play() {
    if (this.playing || !this.sources.length) return;
    await this.ctx.resume();
    if (this.offset >= this.duration - 0.01) this.offset = this.loop ? this.loop[0] : 0;
    this.start();
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.position();
    this.halt();
  }

  stop() {
    this.halt();
    this.offset = 0;
  }

  toggle() {
    return this.playing ? this.pause() : this.play();
  }

  seek(seconds) {
    const target = Math.max(0, Math.min(seconds, this.duration));
    const wasPlaying = this.playing;
    this.halt();
    this.offset = target;
    if (wasPlaying) this.start();
  }

  select(index) {
    if (index < 0 || index >= this.sources.length) return;
    this.active = index;
    this.applyGains(false);
    this.renderSources();
  }

  setMatched(on) {
    this.matched = on;
    this.applyGains(false);
  }

  setLoop(start, end) {
    const valid = start !== null && end !== null && end - start > 0.05;
    const position = this.position();
    const wasPlaying = this.playing;
    this.halt();
    this.loop = valid ? [start, end] : start !== null && end === null ? [start, this.duration] : null;
    this.offset = this.loop ? Math.max(this.loop[0], Math.min(position, this.loop[1] - 0.01)) : position;
    if (wasPlaying) this.start();
  }

  startBlind() {
    if (this.sources.length < 2) return;
    const order = this.sources.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    this.blind = { order, ranking: [], revealed: false };
    this.active = order[0];
    this.applyGains(false);
    this.renderSources();
  }

  endBlind() {
    this.blind = null;
    this.applyGains(false);
    this.renderSources();
  }

  rank(index) {
    if (!this.blind || this.blind.revealed) return;
    const at = this.blind.ranking.indexOf(index);
    if (at >= 0) this.blind.ranking.splice(at, 1);
    else this.blind.ranking.push(index);
    this.renderSources();
  }

  async reveal() {
    const blind = this.blind;
    if (!blind || blind.ranking.length !== this.sources.length) return;
    const letter = (index) => LETTERS[blind.order.indexOf(index)];
    const payload = {
      track: this.track,
      candidates: this.sources.map((s) => ({ id: s.id, label: s.label })),
      ranking: blind.ranking.map(letter),
      revealed: Object.fromEntries(this.sources.map((s, i) => [letter(i), s.id])),
    };
    blind.revealed = true;
    this.renderSources();
    try {
      const saved = await server.save_blind_test(payload);
      this.status(saved && saved.error ? `Not saved: ${saved.error}` : "Blind test saved.");
    } catch (error) {
      this.status(`Not saved: ${error.message || error}`);
    }
  }

  status(text) {
    this.q(".mb-status").textContent = text;
  }

  renderSources() {
    const list = this.q(".mb-sources");
    const blind = this.blind;
    list.replaceChildren(
      ...this.order().map((index, slot) => {
        const source = this.sources[index];
        const row = document.createElement("div");
        row.className = "mb-source" + (index === this.active ? " mb-active" : "");
        const pick = document.createElement("button");
        const hidden = blind && !blind.revealed;
        const name = blind ? LETTERS[slot] + (hidden ? "" : ` = ${source.label}`) : source.label;
        pick.textContent = `${slot + 1}  ${name}`;
        pick.onclick = () => this.select(index);
        row.append(pick);
        if (blind) {
          const place = blind.ranking.indexOf(index);
          const rank = document.createElement("button");
          rank.className = "mb-rank";
          rank.textContent = place >= 0 ? `#${place + 1}` : "rank";
          rank.disabled = blind.revealed;
          rank.onclick = () => this.rank(index);
          row.append(rank);
        } else {
          const level = document.createElement("span");
          level.className = "mb-level";
          level.textContent = `${source.lufs.toFixed(1)} LUFS`;
          row.append(level);
        }
        return row;
      })
    );
    this.q(".mb-blind").textContent = blind ? "Leave blind mode" : "Blind test";
    this.q(".mb-blind").disabled = this.sources.length < 2;
    this.q(".mb-match").disabled = Boolean(blind);
    const ready = blind && !blind.revealed && blind.ranking.length === this.sources.length;
    this.q(".mb-save").style.display = blind && !blind.revealed ? "" : "none";
    this.q(".mb-save").disabled = !ready;
    this.q(".mb-hint").textContent = blind && !blind.revealed
      ? "Levels are matched. Click “rank” on each version from best to worst, then reveal."
      : "Keys: 1–9 switch version, space plays or pauses.";
  }

  render() {
    const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
    const position = this.position();
    if (this.playing && !this.loop && position >= this.duration) {
      this.halt();
      this.offset = this.duration;
    }
    this.q(".mb-play").textContent = this.playing ? "Pause" : "Play";
    this.q(".mb-play").disabled = !this.sources.length;
    this.q(".mb-time").textContent = `${clock(position)} / ${clock(this.duration)}`;
    if (document.activeElement !== this.q(".mb-seek")) {
      this.q(".mb-seek").value = this.duration ? (position / this.duration) * 1000 : 0;
    }
    this.q(".mb-loop").textContent = this.loop
      ? `Loop ${clock(this.loop[0])}–${clock(this.loop[1])}`
      : "No loop";
  }
}

const player = new ABPlayer(element.querySelector(".mb-player"));
window.masterbenchPlayer = player;
const parse = (value) => {
  try {
    return JSON.parse(value || "{}");
  } catch (_) {
    return {};
  }
};
player.load(parse(props.value));
watch("value", () => {
  player.attach(element.querySelector(".mb-player"));
  player.load(parse(props.value));
});

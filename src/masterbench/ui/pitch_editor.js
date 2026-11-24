// Piano-roll view of a vocal's detected notes over its pitch curve. Runs as the js_on_load
// body of a gr.HTML component: `element`, `props`, `watch` and `server` come from Gradio.
//
// Each note shows where it was sung (thin line) and where tuning will send it (bar).
// Click a note to leave it alone, drag it up or down to send it to another pitch, and
// click an edited note to undo. Edits are saved as they are made and used by the next tune.

const ROW = 14;
const PX_PER_S = 120;
const LABELS = 44;
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const SVG = "http://www.w3.org/2000/svg";
const noteName = (midi) => `${NAMES[((Math.round(midi) % 12) + 12) % 12]}${Math.floor(Math.round(midi) / 12) - 1}`;

class PitchEditor {
  constructor() {
    this.data = { notes: [], curve: [] };
    this.edits = {};
    this.drag = null;
  }

  make(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (parent) parent.appendChild(node);
    return node;
  }

  y(midi) {
    return (this.hi - midi) * ROW + ROW / 2;
  }

  target(note) {
    const edit = this.edits[note.index];
    return edit && edit.target != null ? edit.target : note.target;
  }

  load(root, data) {
    this.root = root;
    this.data = data && data.notes ? data : { notes: [], curve: [] };
    this.edits = { ...(this.data.edits || {}) };
    const empty = root.querySelector(".mb-pitch-empty");
    const view = root.querySelector(".mb-pitch-view");
    empty.style.display = this.data.notes.length ? "none" : "block";
    view.style.display = this.data.notes.length ? "flex" : "none";
    if (this.data.notes.length) this.draw();
    this.status();
  }

  draw() {
    const { notes, curve } = this.data;
    const pitches = notes.flatMap((n) => [n.center, n.target]);
    this.lo = Math.floor(Math.min(...pitches)) - 3;
    this.hi = Math.ceil(Math.max(...pitches)) + 3;
    const rows = this.hi - this.lo + 1;
    const width = Math.max(600, Math.ceil(this.data.duration_s * PX_PER_S));
    const height = rows * ROW;
    const scale = new Set(this.data.scale || []);

    const labels = this.root.querySelector(".mb-pitch-labels");
    labels.replaceChildren();
    labels.setAttribute("width", LABELS);
    labels.setAttribute("height", height);
    const grid = this.root.querySelector(".mb-pitch-grid");
    grid.replaceChildren();
    grid.setAttribute("width", width);
    grid.setAttribute("height", height);

    for (let midi = this.hi; midi >= this.lo; midi--) {
      const top = (this.hi - midi) * ROW;
      const inScale = scale.has(((midi % 12) + 12) % 12);
      this.make("rect", { x: 0, y: top, width, height: ROW, class: inScale ? "mb-row-in" : "mb-row-out" }, grid);
      const text = this.make("text", { x: LABELS - 6, y: top + ROW - 3, class: inScale ? "mb-label-in" : "mb-label-out" }, labels);
      text.textContent = noteName(midi);
    }
    for (let second = 0; second <= this.data.duration_s; second++) {
      this.make("line", { x1: second * PX_PER_S, x2: second * PX_PER_S, y1: 0, y2: height, class: "mb-second" }, grid);
      const mark = this.make("text", { x: second * PX_PER_S + 3, y: 10, class: "mb-time" }, grid);
      mark.textContent = `${Math.floor(second / 60)}:${String(second % 60).padStart(2, "0")}`;
    }

    // A frame far outside the notes is a tracking slip; drawing it would streak the view.
    let path = "";
    let pen = false;
    curve.forEach((midi, i) => {
      if (midi == null || midi > this.hi || midi < this.lo) {
        pen = false;
        return;
      }
      path += `${pen ? "L" : "M"}${(i * this.data.step_s * PX_PER_S).toFixed(1)} ${this.y(midi).toFixed(1)}`;
      pen = true;
    });
    this.make("path", { d: path, class: "mb-curve" }, grid);

    this.groups = {};
    for (const note of notes) {
      const group = this.make("g", { class: "mb-note", "data-index": note.index }, grid);
      const x = note.start_s * PX_PER_S;
      const w = Math.max(4, (note.end_s - note.start_s) * PX_PER_S);
      this.make("rect", { class: "mb-note-target", x, width: w, height: ROW - 2, rx: 3 }, group);
      this.make("line", { class: "mb-note-sung", x1: x, x2: x + w, y1: this.y(note.center), y2: this.y(note.center) }, group);
      const label = this.make("text", { class: "mb-note-label", x: x + 3 }, group);
      this.make("title", {}, group);
      group.addEventListener("pointerdown", (e) => this.down(e, note));
      this.groups[note.index] = { group, label };
      this.paint(note);
    }
    grid.onpointermove = (e) => this.move(e);
    grid.onpointerup = (e) => this.up(e);
    grid.onpointerleave = (e) => this.up(e);
  }

  paint(note, previewTarget) {
    const { group, label } = this.groups[note.index];
    const edit = this.edits[note.index] || {};
    const target = previewTarget != null ? previewTarget : this.target(note);
    const bar = group.querySelector(".mb-note-target");
    bar.setAttribute("y", this.y(target) - ROW / 2 + 1);
    label.setAttribute("y", this.y(target) + 4);
    const off = Math.abs(note.cents);
    const state = edit.bypass ? "bypass" : edit.target != null || previewTarget != null ? "edited" : off >= 50 ? "far" : off >= 20 ? "off" : "ok";
    group.setAttribute("data-state", state);
    label.textContent = edit.bypass ? "left alone" : `${note.index + 1}`;
    const moved = Math.round((target - note.center) * 100);
    group.querySelector("title").textContent = edit.bypass
      ? `Note ${note.index + 1}: ${note.name}, left alone`
      : `Note ${note.index + 1}: sung ${note.name} ${note.cents >= 0 ? "+" : ""}${Math.round(note.cents)} cents, goes to ${noteName(target)} (${moved >= 0 ? "+" : ""}${moved} cents)`;
  }

  down(event, note) {
    event.preventDefault();
    this.drag = { note, startY: event.clientY, steps: 0, from: Math.round(this.target(note)) };
  }

  move(event) {
    if (!this.drag) return;
    const steps = Math.round((this.drag.startY - event.clientY) / ROW);
    if (steps !== this.drag.steps) {
      this.drag.steps = steps;
      this.paint(this.drag.note, this.drag.from + steps);
    }
  }

  up() {
    if (!this.drag) return;
    const { note, steps, from } = this.drag;
    this.drag = null;
    const edit = this.edits[note.index] || {};
    if (steps === 0) {
      // A click undoes an edit; on an unedited note it means "leave this one alone".
      if (edit.bypass || edit.target != null) delete this.edits[note.index];
      else this.edits[note.index] = { bypass: true, target: null };
    } else {
      this.edits[note.index] = { bypass: false, target: from + steps };
    }
    this.paint(note);
    this.save();
  }

  status(message) {
    const count = Object.keys(this.edits).length;
    const parts = Object.entries(this.edits)
      .slice(0, 6)
      .map(([index, edit]) => (edit.bypass ? `note ${Number(index) + 1} left alone` : `note ${Number(index) + 1} to ${noteName(edit.target)}`));
    const summary = count ? `${count} edited: ${parts.join(", ")}${count > 6 ? ", ..." : ""}` : "No edits. Click a note to leave it alone; drag it to move it; click it again to undo.";
    this.root.querySelector(".mb-pitch-status").textContent = message ? `${summary} ${message}` : summary;
  }

  async save() {
    this.status("Saving...");
    try {
      const saved = await server.save_edits({ source: this.data.source, edits: this.edits });
      this.status(saved.error ? `Not saved: ${saved.error}` : "Saved; used the next time you tune.");
    } catch (error) {
      this.status(`Not saved: ${error}`);
    }
  }
}

const editor = new PitchEditor();
window.masterbenchPitchEditor = editor;
const parse = (value) => {
  try {
    return JSON.parse(value || "{}");
  } catch (_) {
    return {};
  }
};
editor.load(element.querySelector(".mb-pitch"), parse(props.value));
watch("value", () => editor.load(element.querySelector(".mb-pitch"), parse(props.value)));

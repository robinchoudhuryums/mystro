// Browser check for the Vocal tab: report, tune, the A/B player and the pitch editor.
// Needs a running UI on an empty data folder and the fixture vocal.
//
//   uv run masterbench fixtures /tmp/fx
//   MASTERBENCH_HOME=/tmp/mb uv run masterbench ui --no-browser &
//   node scripts/check_vocal.js http://127.0.0.1:7860 /tmp/fx/vocal.wav
//
// Requires Node with Playwright (npm i -g playwright) and its Chromium.

const { chromium } = require("playwright");

const [base, vocal] = process.argv.slice(2);
if (!base || !vocal) {
  console.error("usage: node scripts/check_vocal.js <base-url> <vocal.wav>");
  process.exit(2);
}

const check = (ok, message) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${message}`);
  if (!ok) process.exitCode = 1;
};

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(base, { waitUntil: "domcontentloaded" });
  await page.getByRole("tab", { name: "Vocal" }).click();
  const tab = page.locator("#mb-vocal-tab");
  const sources = () => page.evaluate(() => window.masterbenchPlayers["mb-vocal-player"].sources.map((s) => s.label));
  const edits = () => page.evaluate(() => window.masterbenchPitchEditor.edits);

  await tab.locator('input[type="file"]').first().setInputFiles(vocal);
  await tab.getByRole("button", { name: "Check pitch" }).click();
  await tab.getByText("8 notes in F minor").waitFor({ timeout: 120000 });
  check(true, "the report finds 8 notes in F minor");
  check((await tab.getByText("guessed from the sung notes").count()) > 0, "the key's source is shown");
  await page.waitForFunction(() => window.masterbenchPlayers["mb-vocal-player"].sources.length === 1);
  check((await sources())[0] === "Original", "the player holds the original");

  await tab.getByRole("button", { name: "Tune", exact: true }).click();
  await tab.getByText("Tuned 2 of 8 notes").waitFor({ timeout: 120000 });
  check(true, "Natural tunes the two sharp notes");
  await page.waitForFunction(() => window.masterbenchPlayers["mb-vocal-player"].sources.length === 2);
  check((await sources())[1] === "Natural (rubberband, 90%)", "the tuned version is in the player");

  await tab.getByRole("button", { name: "Advanced" }).click();
  await page.waitForFunction(() => document.querySelectorAll("#mb-pitch-editor .mb-note").length === 8);
  check(true, "the editor draws 8 notes");
  const states = await page.$$eval("#mb-pitch-editor .mb-note", (notes) => notes.map((n) => n.dataset.state));
  check(states.filter((s) => s === "off").length === 2, `two notes are marked as off (${states.join(" ")})`);

  // Mouse positions are relative to the viewport, so the note has to be on screen first.
  const box = async (n) => {
    const bar = page.locator(`#mb-pitch-editor .mb-note[data-index="${n}"] .mb-note-target`);
    await bar.scrollIntoViewIfNeeded();
    return bar.boundingBox();
  };
  let b = await box(5);
  await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
  await tab.getByText("Saved; used the next time you tune.").waitFor({ timeout: 20000 });
  check((await edits())[5].bypass === true, "clicking a note leaves it alone");

  b = await box(2);
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2 - 28, { steps: 4 });
  await page.mouse.up();
  await page.waitForFunction(() => window.masterbenchPitchEditor.edits[2]);
  check((await edits())[2].target === 62, "dragging a note up two rows retargets it two semitones up");
  b = await box(2);
  await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
  await page.waitForFunction(() => !window.masterbenchPitchEditor.edits[2]);
  check(true, "clicking an edited note undoes the edit");
  await tab.getByText("1 edited: note 6 left alone").waitFor({ timeout: 20000 });

  await tab.getByRole("button", { name: "Tune", exact: true }).click();
  await tab.getByText("Tuned 1 of 8 notes").waitFor({ timeout: 120000 });
  check(true, "the next tune skips the bypassed note");

  await tab.getByLabel("hard-tune").check();
  await page.waitForFunction(() => document.querySelector('input[type="radio"][value="world"]')?.checked);
  check(true, "choosing Hard-tune switches the engine to WORLD");

  await tab.getByRole("button", { name: "Compare engines" }).click();
  await tab.getByText("None of this says which sounds better").waitFor({ timeout: 240000 });
  await page.waitForFunction(() => window.masterbenchPlayers["mb-vocal-player"].sources.length >= 6);
  check(true, `both engines are lined up for a blind test (${(await sources()).join(" | ")})`);

  const vocalPlayer = page.locator("#mb-vocal-player");
  await vocalPlayer.locator(".mb-blind").click();
  const ranks = vocalPlayer.locator(".mb-rank");
  const count = await ranks.count();
  for (let i = 0; i < count; i++) await ranks.nth(i).click();
  await vocalPlayer.locator(".mb-save").click();
  await vocalPlayer.getByText("Blind test saved.").waitFor({ timeout: 20000 });
  check(count >= 6, `a blind test over all ${count} versions is saved with the track`);
  check(errors.length === 0, `no page errors (${errors.join("; ") || "none"})`);
  await page.screenshot({ path: process.env.SHOT || "/dev/null", fullPage: true }).catch(() => {});
  await browser.close();
})();

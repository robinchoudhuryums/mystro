// Browser check for the A/B/C player: sources load, switching keeps the playhead, and
// level-matched sources play at one loudness. Needs a running UI and a master run id.
//
//   uv run masterbench ui --no-browser &
//   node scripts/check_player.js http://127.0.0.1:7860 <run-id>
//
// Requires Node with Playwright (npm i -g playwright) and its Chromium.

const { chromium } = require("playwright");

const [base, runId] = process.argv.slice(2);
if (!base || !runId) {
  console.error("usage: node scripts/check_player.js <base-url> <run-id>");
  process.exit(2);
}

const check = (ok, message) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${message}`);
  if (!ok) process.exitCode = 1;
};

(async () => {
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${base}/?run=${runId}`, { waitUntil: "domcontentloaded" });
  await page.getByRole("tab", { name: "Compare" }).click();
  await page.waitForFunction(() => window.masterbenchPlayer?.sources.length >= 2, null, {
    timeout: 60000,
  });

  const state = () =>
    page.evaluate(() => {
      const p = window.masterbenchPlayer;
      return {
        labels: p.sources.map((s) => s.label),
        active: p.active,
        playing: p.playing,
        position: p.position(),
        gains: p.nodes.map((n) => n.gain.gain.value),
        matched: p.sources.map((s, i) => s.lufs + 20 * Math.log10(p.gainFor(i))),
      };
    });

  const loaded = await state();
  check(loaded.labels.length >= 2, `sources loaded: ${loaded.labels.join(", ")}`);

  await page.locator(".mb-play").click();
  await page.waitForTimeout(1200);
  const before = await state();
  await page.keyboard.press("1");
  const during = await state();
  await page.waitForTimeout(300);
  const after = await state();

  check(before.playing && before.position > 1.0, `playing, at ${before.position.toFixed(2)} s`);
  check(after.active === 0 && after.active !== before.active, "key 1 selected the first source");
  const jump = during.position - before.position;
  check(jump >= 0 && jump < 0.1, `playhead continued across the switch (+${(jump * 1000).toFixed(0)} ms)`);
  const drift = after.position - during.position;
  check(Math.abs(drift - 0.3) < 0.1, `playhead kept running afterwards (+${(drift * 1000).toFixed(0)} ms)`);
  const audible = after.gains.filter((gain) => gain > 1e-4).length;
  check(audible === 1 && after.gains[0] > 0, "only the selected source is audible");
  const spread = Math.max(...after.matched) - Math.min(...after.matched);
  check(spread < 0.01, `matched playback loudness spread ${spread.toFixed(3)} LU`);

  await page.locator(".mb-match").uncheck();
  const real = await state();
  check(
    Math.max(...real.matched) - Math.min(...real.matched) > 0.5 || loaded.labels.length < 2,
    "turning matching off restores the real level differences"
  );
  await page.keyboard.press("Space");
  await page.waitForTimeout(150);
  check(!(await state()).playing, "space pauses");
  check(errors.length === 0, `no page errors${errors.length ? ": " + errors[0] : ""}`);
  await browser.close();
})().catch((error) => {
  console.error("FAIL ", error.message);
  process.exit(1);
});

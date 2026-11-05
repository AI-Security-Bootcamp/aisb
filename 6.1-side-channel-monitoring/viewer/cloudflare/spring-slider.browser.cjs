// Native range events differ between engines, so exercise actual browser input.
// Requires Playwright with its Firefox and Chromium browsers installed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, firefox } = require('playwright');

async function check(engine) {
  const browser = await engine.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 900, height: 600 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('http://slider.test/**', route => {
      const module = route.request().url().endsWith('.mjs');
      return route.fulfill({
        contentType: module ? 'text/javascript' : 'text/html',
        body: module ? fs.readFileSync(path.join(__dirname, '../static/spring-slider.mjs'), 'utf8') : `
          <input id="slider" type="range" min="-1" max="1" step="0.01" value="0"
            style="width:400px;margin:50px">
          <script type="module">
            import { springSlider } from './spring-slider.mjs';
            window.adjustment = 0; window.finishes = 0;
            springSlider(document.querySelector('input'),
              (value, dt) => { adjustment += value * dt; }, () => { finishes++; });
            window.ready = true;
          </script>`,
      });
    });
    await page.goto('http://slider.test/');
    await page.waitForFunction(() => window.ready);
    const slider = page.locator('#slider');
    const box = await slider.boundingBox();
    const state = () => page.evaluate(() => ({
      value: document.querySelector('input').value, adjustment, finishes,
    }));
    let completed = 0;

    async function assertStopped(before) {
      await page.waitForTimeout(40);
      const released = await state();
      assert.equal(released.value, '0', 'Releasing must recenter the knob');
      assert.notEqual(released.adjustment, before, 'The adjustment must be retained');
      assert.equal(released.finishes, ++completed, 'Each interaction must finish once');
      await page.waitForTimeout(180);
      assert.deepEqual(await state(), released, 'No adjustment may continue after release');
    }

    for (const outside of [false, true]) {
      const before = (await state()).adjustment;
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * 0.85, box.y + box.height / 2, { steps: 6 });
      await page.waitForTimeout(100);
      if (outside) await page.mouse.move(box.x + box.width + 100, box.y + 100);
      await page.mouse.up();
      await assertStopped(before);
    }

    for (const key of ['ArrowLeft', 'ArrowRight']) {
      const before = (await state()).adjustment;
      await slider.focus();
      await page.keyboard.down(key);
      await page.waitForTimeout(80);
      await page.keyboard.up(key);
      await assertStopped(before);
    }

    // Track clicks also trigger Firefox's final native input event on release.
    const before = (await state()).adjustment;
    await slider.click({ position: { x: box.width * 0.2, y: box.height / 2 } });
    await assertStopped(before);
    assert.deepEqual(errors, []);
    console.log(`${engine.name()}: drag, outside release, keyboard, click, and no post-release drift passed`);
  } finally {
    await browser.close();
  }
}

(async () => {
  for (const engine of [firefox, chromium]) await check(engine);
})().catch(error => { console.error(error); process.exitCode = 1; });

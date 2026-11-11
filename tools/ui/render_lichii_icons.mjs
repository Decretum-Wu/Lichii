// Renders the Lichii theme's SVG sources to the PNGs the stylesheet and dialogs load,
// at 1x and @2x. Qt picks the @2x file on high-DPI screens. Needs @resvg/resvg-js:
//   npm i @resvg/resvg-js && node tools/ui/render_lichii_icons.mjs
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { Resvg } from "@resvg/resvg-js";

const root = new URL("../../dist/qt_themes/lichii/", import.meta.url);
const src = new URL("svg/", root);
const out = new URL("icons/", root);
mkdirSync(out, { recursive: true });
for (const file of readdirSync(src).filter((f) => f.endsWith(".svg")).sort()) {
    const svg = readFileSync(new URL(file, src));
    const width = Number(/width="(\d+)"/.exec(svg)[1]);
    for (const [suffix, scale] of [["", 1], ["@2x", 2]]) {
        const png = new Resvg(svg, { fitTo: { mode: "width", value: width * scale } }).render().asPng();
        writeFileSync(new URL(file.replace(".svg", suffix + ".png"), out), png);
    }
}

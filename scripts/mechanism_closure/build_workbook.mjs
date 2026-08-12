import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const packageDir = process.argv[2];
const previewDir = process.argv[3];
if (!packageDir || !previewDir) throw new Error("usage: node build_workbook.mjs PACKAGE_DIR PREVIEW_DIR");

const imports = [
  ["Harm", "refresh_harm_replication.csv"],
  ["Predictors", "predictor_gate_summary.csv"],
  ["Geo Events", "geo01b_event_summary.csv"],
  ["Geo Units", "geo01b_unit_outcomes.csv"],
  ["Claims", "claim_boundary.csv"],
  ["Evidence", "mechanism_evidence_ledger.csv"],
  ["Input Audit", "input_audit.csv"],
];
// CSV ingestion must happen before any collaborative formatting operations.
const wb = await Workbook.fromCSV("placeholder", { sheetName: "Summary" });
for (const [sheetName, fileName] of imports) {
  const csvText = await fs.readFile(path.join(packageDir, fileName), "utf8");
  await wb.fromCSV(csvText, { sheetName });
}
const summary = wb.worksheets.getItem("Summary");

const navy = "#17365D";
const blue = "#D9EAF7";
const green = "#E2F0D9";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const grid = "#D9E2F3";
const white = "#FFFFFF";

summary.showGridLines = false;
summary.mergeCells("A1:F1");
summary.getRange("A1").values = [["MECHANISM CLOSURE | 2026-08-05"]];
summary.getRange("A1:F1").format = {
  fill: navy,
  font: { bold: true, color: white, size: 18 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A1:F1").format.rowHeight = 32;

summary.mergeCells("A3:F4");
summary.getRange("A3").values = [[
  "Decision: close the mechanism line. Retain the replicated refresh-loss impulse; do not claim an origin-independent scalar/local-geometry explanation; do not authorize GEO-01C."
]];
summary.getRange("A3:F4").format = {
  fill: amber,
  font: { bold: true, color: "#7F6000", size: 11 },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "all", style: "thin", color: "#D6B656" },
};

summary.mergeCells("A6:F6");
summary.getRange("A6").values = [["Replicated refresh harm"]];
summary.getRange("A6:F6").format = { fill: blue, font: { bold: true, color: navy }, borders: { preset: "all", style: "thin", color: grid } };
summary.getRange("A7:F7").values = [["Event", "MDP-05 mean", "GEO-01B mean", "MDP-05 signs", "GEO-01B signs", "Adjudication"]];
summary.getRange("A8:A9").values = [["Production refresh"], ["Delayed refresh"]];
summary.getRange("B8:E9").formulas = [
  ["=Harm!E5", "=Harm!E4", "=Harm!D5&\"/\"&Harm!C5", "=Harm!D4&\"/\"&Harm!C4"],
  ["=Harm!E3", "=Harm!E2", "=Harm!D3&\"/\"&Harm!C3", "=Harm!D2&\"/\"&Harm!C2"],
];
summary.getRange("F8:F9").values = [["Replicated positive effect"], ["Replicated positive effect"]];
summary.getRange("B8:C9").setNumberFormat("0.000000");

summary.mergeCells("A12:F12");
summary.getRange("A12").values = [["GEO-01B local closure versus short-horizon generalization"]];
summary.getRange("A12:F12").format = { fill: blue, font: { bold: true, color: navy }, borders: { preset: "all", style: "thin", color: grid } };
summary.getRange("A13:D13").values = [["Metric", "Production", "Delayed", "Interpretation"]];
summary.getRange("A14:A16").values = [["Origin eta-squared (harm)"], ["Full-Taylor median relative error"], ["Curvature local error reduction"]];
summary.getRange("B14:C16").formulas = [
  ["='Geo Events'!P3", "='Geo Events'!P2"],
  ["='Geo Events'!M3", "='Geo Events'!M2"],
  ["='Geo Events'!O3", "='Geo Events'!O2"],
];
summary.getRange("D14:D16").values = [
  ["Most unit variation is carried by checkpoint origin"],
  ["Immediate line loss is numerically closed"],
  ["Curvature improves local approximation, not 16-step prediction"],
];
summary.mergeCells("D14:F14");
summary.mergeCells("D15:F15");
summary.mergeCells("D16:F16");
summary.getRange("B14:C14").setNumberFormat("0.0%");
summary.getRange("B15:C15").setNumberFormat("0.0000%");
summary.getRange("B16:C16").setNumberFormat("0.0%");
summary.getRange("A14:F16").format.rowHeight = 38;

summary.mergeCells("A19:F19");
summary.getRange("A19").values = [["Paper boundary"]];
summary.getRange("A19:F19").format = { fill: blue, font: { bold: true, color: navy }, borders: { preset: "all", style: "thin", color: grid } };
summary.getRange("A20:B22").values = [
  ["Allowed", "Refresh produces a reproducible short-horizon held-out loss impulse inside the frozen matched replay tree."],
  ["Required negative", "Shock scalars and local directional geometry were not confirmed as origin-independent quantitative mediators."],
  ["Prohibited", "Do not claim universal mechanism, multi-step curvature prediction, automatic layer selection, or GEO-01B confirmatory evidence."],
];
summary.mergeCells("B20:F20");
summary.mergeCells("B21:F21");
summary.mergeCells("B22:F22");
summary.getRange("A20:F20").format.fill = green;
summary.getRange("A21:F21").format.fill = amber;
summary.getRange("A22:F22").format.fill = red;

for (const range of ["A7:F9", "A13:D16", "A20:F22"]) {
  summary.getRange(range).format.borders = { preset: "all", style: "thin", color: grid };
  summary.getRange(range).format.wrapText = true;
}
for (const range of ["A7:F7", "A13:D13"]) {
  summary.getRange(range).format = {
    fill: navy,
    font: { bold: true, color: white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: grid },
  };
}
summary.getRange("A1:F22").format.font.name = "Aptos";
summary.getRange("A3:F22").format.wrapText = true;
summary.getRange("A1:A22").format.columnWidth = 24;
summary.getRange("B1:C22").format.columnWidth = 19;
summary.getRange("D1:E22").format.columnWidth = 20;
summary.getRange("F1:F22").format.columnWidth = 48;
summary.getRange("B20:F22").format.rowHeight = 37;
summary.freezePanes.freezeRows(1);

for (const [sheetName] of imports) {
  const sheet = wb.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  used.format.font.name = "Aptos";
  used.format.borders = { preset: "all", style: "thin", color: grid };
  used.format.wrapText = true;
  used.format.verticalAlignment = "top";
  used.getRow(0).format = {
    fill: navy,
    font: { bold: true, color: white, name: "Aptos" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: grid },
  };
  used.getRow(0).format.rowHeight = 38;
  used.format.autofitColumns();
  used.format.autofitRows();
}

// Keep narrative-heavy tables readable instead of letting long text create extremely wide sheets.
wb.worksheets.getItem("Claims").getRange("D1:F7").format.columnWidth = 48;
wb.worksheets.getItem("Evidence").getRange("E1:G19").format.columnWidth = 48;
wb.worksheets.getItem("Predictors").getRange("L1:L11").format.columnWidth = 46;
wb.worksheets.getItem("Input Audit").getRange("C1:C15").format.columnWidth = 56;
wb.worksheets.getItem("Input Audit").getRange("H1:H15").format.columnWidth = 42;
wb.worksheets.getItem("Geo Units").getRange("A1:N25").format.columnWidth = 18;
wb.worksheets.getItem("Geo Events").getRange("A1:R3").format.columnWidth = 18;
for (const [sheetName] of imports) {
  wb.worksheets.getItem(sheetName).getUsedRange().format.autofitRows();
}

await fs.mkdir(previewDir, { recursive: true });
const overview = await wb.inspect({ kind: "workbook,sheet", maxChars: 5000, tableMaxRows: 4, tableMaxCols: 6 });
await fs.writeFile(path.join(previewDir, "inspect_overview.ndjson"), overview.ndjson, "utf8");
const formulas = await wb.inspect({ kind: "formula", sheetId: "Summary", range: "A1:F22", maxChars: 5000, options: { maxResults: 50 } });
await fs.writeFile(path.join(previewDir, "inspect_formulas.ndjson"), formulas.ndjson, "utf8");
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(previewDir, "inspect_formula_errors.ndjson"), errors.ndjson, "utf8");

for (const [sheetName] of [["Summary"], ...imports]) {
  const blob = await wb.render({ sheetName, autoCrop: "all", scale: sheetName === "Summary" ? 1.2 : 0.8, format: "png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const safe = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(path.join(previewDir, `${safe}.png`), bytes);
}

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(packageDir, "mechanism_closure_workbook.xlsx"));

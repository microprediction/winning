import { runAll } from './checks.js';
const rows = runAll();
let fail = 0;
for (const r of rows) {
  const mark = r.pass ? 'PASS' : 'FAIL';
  if (!r.pass) fail += 1;
  console.log(`${mark}  ${r.id.padEnd(20)} ${r.claim}`);
  console.log(`      ${r.where}`);
  console.log(`      ${r.detail}`);
}
console.log(`\n${rows.length - fail} of ${rows.length} checks pass`);
process.exit(fail ? 1 : 0);

import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {analyze,normalize,sourceString} from '../web/engine.mjs';
const read=p=>JSON.parse(readFileSync(new URL(p,import.meta.url),'utf8'));
const model=read('../aiqiang/model.json'),checks=read('../web/checks.json'),cases=read('../tests/web_reference.json');
function close(a,b,path='') {
  if(typeof a==='number'&&typeof b==='number'){assert.ok(Math.abs(a-b)<1e-9,`${path}: ${a} vs ${b}`);return;}
  if(Array.isArray(b)){assert.equal(a.length,b.length,path);b.forEach((x,i)=>close(a[i],x,path+'.'+i));return;}
  if(b&&typeof b==='object'){for(const key of Object.keys(b))close(a[key],b[key],path+'.'+key);return;}
  assert.equal(a,b,path);
}
for(const c of cases) {
  assert.equal(normalize(c.text),c.normalized,c.name+' normalize');
  for(let i=0;i<model.feature_sets.length;i++)assert.equal(sourceString(c.text,model.feature_sets[i]),c.sources[i],c.name+' source');
  const actual=analyze(c.text,model,checks);
  // Browser ranges use UTF-16 offsets; Python uses Unicode code point offsets.
  const jsOffset=n=>[...c.text].slice(0,n).join('').length;
  const expected=c.result;
  for(const row of expected.markers)row.spans=row.spans.map(([a,b])=>[jsOffset(a),jsOffset(b)]);
  for(const p of expected.pieces){p.start=jsOffset(p.start);p.end=jsOffset(p.end);}
  for(const key of ['chars','verdict','verdict_text','ai_prob','which','which_text','baigu_index','markers','pieces','evidence'])if(key in expected&&!(key==='verdict_text'&&expected.verdict==='too_short'))close(actual[key],expected[key],c.name+'.'+key);
}
console.log(`PASS: ${cases.length} Python ↔ JavaScript cases; model probabilities, rules, evidence, Unicode offsets and long texts agree within 1e-9.`);

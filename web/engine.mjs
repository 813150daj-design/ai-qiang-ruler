export const cjkCount = t => [...t].filter(c => /[\u3400-\u4dbf\u4e00-\u9fff]/u.test(c)).length;
export function normalize(text) {
  return text.normalize('NFKC').replace(/^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$/gm,'')
    .replace(/^[ \t]*(#{1,6}[ \t]+|>[ \t]*|[-*+][ \t]+)/gm,'').replace(/\*\*|__|`/g,'')
    .replace(/(https?:\/\/|www\.)\S+/gu,' ').replace(/@[\p{L}\p{N}_一-鿿-]{1,30}/gu,' ')
    .replace(/\[[^\[\]\s]{1,8}\]/gu,' ').replace(/#[^#\n]{1,30}#/gu,' ')
    .replace(/[“”「」『』〝〞〟"]/gu,'"').replace(/[‘’']/gu,"'").replace(/…/gu,'...')
    .replace(/\.{2,}|。{2,}|⋯+|·{3,}/gu,'…').replace(/[—–―‒-]{2,}|[—–―‒]/gu,'—')
    .toLowerCase().replace(/[0-9]+(?:[.,][0-9]+)*/g,'0').replace(/[a-z]+/g,'a').replace(/\s+/gu,'');
}
export function sourceString(text, fs) {
  const norm = normalize(text);
  if(fs.source==='plain') return norm;
  const keep=new Set(fs.keep), punct=new Set(fs.punct), out=[], mask=fs.mask||'x';
  for(const c of norm) {
    if(keep.has(c)||punct.has(c)) out.push(c);
    else if(/[\p{L}\p{N}]/u.test(c)) {if(out.at(-1)!==mask) out.push(mask);}
    else out.push(c);
  }
  return out.join('');
}
export function featurize(text,fs) {
  const chars=[...sourceString(text,fs)], counts=new Map(), vals=new Map();
  for(const n of fs.n) for(let i=0;i<=chars.length-n;i++) {
    const g=chars.slice(i,i+n).join(''); counts.set(g,(counts.get(g)||0)+1);
  }
  let sum=0;
  for(const [g,tf] of counts) if(Object.hasOwn(fs.vocab,g)) {const v=(1+Math.log(tf))*fs.vocab[g][0];vals.set(g,v);sum+=v*v;}
  const len=Math.sqrt(sum); if(len) for(const [g,v] of vals) vals.set(g,v/len);
  return vals;
}
const softmax=z=>{const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
function predict(model,text) {
  const z=[...model.bias];
  for(const fs of model.feature_sets) for(const [g,v] of featurize(text,fs)) for(let c=0;c<z.length;c++) z[c]+=v*fs.vocab[g][c+1];
  const m=Math.max(...z.slice(1)), score=m+Math.log(z.slice(1).reduce((s,v)=>s+Math.exp(v-m),0))-z[0];
  return {logits:z,probs:softmax(z),ai_score:score,ai_prob:sigmoid(model.ai_calibration.a*score+model.ai_calibration.b),which:softmax(z.slice(1))};
}
export function splitPieces(text,params={}) {
  const p={min_len:500,target:300,min_piece:150,...params};
  if([...text].length<=p.min_len) return [[0,text.length]];
  const spans=[];let start=null,size=0;
  for(const m of text.matchAll(/[^。！？!?；;…\n]*(?:[。！？!?；;…]+|\n+|$)/gu)) {
    if(!m[0]) continue;if(start===null) start=m.index;size += [...m[0].trim()].length;
    if(size>=p.target) {spans.push([start,m.index+m[0].length]);start=null;size=0;}
  }
  if(start!==null) {if(spans.length&&size<p.min_piece) spans.at(-1)[1]=text.length;else spans.push([start,text.length]);}
  const out=spans.map(([a,b])=>{while(a<b&&/\s/u.test(text[a]))a++;while(b>a&&/\s/u.test(text[b-1]))b--;return [a,b];}).filter(([a,b])=>a<b);
  return out.length<2?[[0,text.length]]:out;
}
export function predictDoc(model,text) {
  let spans=splitPieces(text,model.long_text);
  if(spans.length>1) {spans=spans.filter(([a,b])=>cjkCount(text.slice(a,b))>=(model.long_text?.min_cjk||30));if(!spans.length)spans=[[0,text.length]];}
  const preds=spans.map(([a,b])=>predict(model,text.slice(a,b))),k=preds.length;
  const mean=key=>preds[0][key].map((_,i)=>preds.reduce((s,p)=>s+p[key][i],0)/k);
  const score=preds.reduce((s,p)=>s+p.ai_score,0)/k;
  return {logits:mean('logits'),probs:mean('probs'),which:mean('which'),ai_score:score,ai_prob:sigmoid(model.ai_calibration.a*score+model.ai_calibration.b),pieces:spans.map(([start,end],i)=>({start,end,ai_prob:preds[i].ai_prob}))};
}
export function markers(text,checks) {
  const positions=[];let flat='';for(let i=0;i<text.length;i++)if(text[i]!=='\n'){positions.push(i);flat+=text[i];}
  const sentences=[];let cursor=0;
  for(const match of flat.matchAll(/(?<=[。！？!?])/gu)){const end=match.index;if(flat.slice(cursor,end).trim())sentences.push({0:flat.slice(cursor,end),index:cursor});cursor=end;}
  if(flat.slice(cursor).trim())sentences.push({0:flat.slice(cursor),index:cursor});
  return checks.map(({name,pattern,hint})=>{
    let spans=[];
    if(pattern) spans=[...text.matchAll(new RegExp(pattern,'gmu'))].map(m=>[m.index,m.index+m[0].length]);
    else if(name==='结尾升华') {
      const m=sentences.at(-1);if(m&&/人生|生活|日子|岁月|时光|幸福|温柔|治愈|意义|勇气|成长|慈悲|美好|滋味|人间|福气|骄傲/u.test(m[0])) {
        const a=m.index+m[0].length-m[0].trimStart().length,b=m.index+m[0].trimEnd().length;spans=[[positions[a],positions[b-1]+1]];
      }
    } else {
      for(const sentence of sentences) {
        const parts=[...sentence[0].matchAll(/[^，、；,]+/gu)].filter(m=>m[0].trim()).map(m=>{const a=sentence.index+m.index+m[0].length-m[0].trimStart().length;return [a,sentence.index+m.index+m[0].trimEnd().length];});
        let run=1;for(let i=1;i<parts.length;i++) {run=flat[parts[i-1][0]]===flat[parts[i][0]]?run+1:1;
          if(run===3)spans.push([positions[parts[i-2][0]],positions[parts[i][1]-1]+1]);else if(run>3)spans.at(-1)[1]=positions[parts[i][1]-1]+1;
        }
      }
    }
    return {name,hint,spans,count:spans.length};
  });
}
export function analyze(text,model,checks) {
  const chars=cjkCount(text), rows=markers(text,checks), limits={min_chars:50,ai_high:.9,human_low:.2,which_min:.5,...model.thresholds};
  const out={chars,markers:rows,baigu_index:rows.filter(r=>r.count).length,evidence:{ai:[],human:[]},pieces:[],which:[],ai_prob:null,verdict:'too_short',verdict_text:`太短了，至少需要 ${limits.min_chars} 个汉字`,notes:['这把尺子只量文风，不能证明是谁写的。请别拿结果指控任何人，尤其是学生的作文。']};
  if(chars<[...text].filter(c=>!(/\s/u.test(c))).length*.3)out.notes.push('这段文字里汉字不多，结果不可靠。');
  out.which_text='';
  if(chars<limits.min_chars)return out;
  const p=predictDoc(model,text);Object.assign(out,p);
  out.verdict=p.ai_prob>=limits.ai_high?'likely_ai':p.ai_prob<=limits.human_low?'likely_human':'uncertain';
  out.verdict_text={likely_ai:'很像 AI 写的',likely_human:'更像人写的',uncertain:'不确定'}[out.verdict];
  out.which=model.classes.slice(1).map((name,i)=>({name,p:p.which[i]})).sort((a,b)=>b.p-a.p);
  out.which_text=out.verdict==='likely_human'?'更像人写的，这一栏仅供参考':out.which[0].p<limits.which_min?'三家都有点像，分不清':'最像 '+out.which[0].name;
  const top=model.classes.indexOf(out.which[0].name), contributions=new Map();
  for(const fs of model.feature_sets.filter(f=>f.source==='plain'))for(const [g,v] of featurize(text,fs))if([...g].length>=2&&cjkCount(g)===[...g].length)contributions.set(g,(contributions.get(g)||0)+v*(fs.vocab[g][1+top]-fs.vocab[g][1]));
  for(const [g,v] of [...contributions].sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])||(a[0]<b[0]?-1:a[0]>b[0]?1:0))) {
    if(!v)continue;const chosen=out.evidence[v>0?'ai':'human'];if(chosen.length<6&&!chosen.some(([old])=>g.includes(old)||old.includes(g)))chosen.push([g,Math.round(v*10000)/10000]);
  }
  if(out.pieces.length>1)out.notes.push(`长文分成 ${out.pieces.length} 段分别分析，最终概率由平均分校准。`);
  return out;
}

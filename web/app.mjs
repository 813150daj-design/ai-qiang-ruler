import {cjkCount} from './engine.mjs';
const $=s=>document.querySelector(s), input=$('#text'), button=$('#measure'), status=$('#status');
let worker, ready=false, busy=false, result=null, measuredText='',activeRule=-1,activePiece=-1,requestId=0;
const escape=t=>String(t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=p=>(Math.min(99.9,Math.max(.1,p*100))).toFixed(1);
function finishError(message){busy=false;button.disabled=false;button.firstChild.textContent='重新量一量 ';status.textContent=message;}
function getWorker(){
  if(worker)return worker;
  worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
    if(data.type==='ready'){ready=true;return;}
    if(data.id!==requestId)return;
    if(data.error){worker.terminate();worker=null;ready=false;finishError('分析失败：'+data.error+'。请检查网络后重试。');return;}
    busy=false;button.disabled=false;button.firstChild.textContent='再量一量 ';result=data.result;
    status.textContent=`已测量 ${result.chars} 个汉字。修改文字后需重新测量。`;
    activeRule=-1;activePiece=-1;render();
  };
  worker.onerror=()=>{worker.terminate();worker=null;ready=false;finishError('模型或分析脚本加载失败，请检查网络后重试。');};return worker;
}
function measure(){
  if(busy)return;if(!input.value.trim()){status.textContent='请先贴入要测量的中文文字。';input.focus();return;}
  measuredText=input.value;busy=true;button.disabled=true;button.firstChild.textContent='测量中… ';status.textContent=ready?'正在本页分析文字…':'正在加载模型（约 8.7 MB），文字不会上传…';
  $('#output').classList.remove('stale');$('#annotations').classList.remove('stale');$('#copy').hidden=true;
  getWorker().postMessage({id:++requestId,text:measuredText});
}
function render(){
  $('#empty').hidden=true;$('#output').hidden=false;$('#copy').hidden=false;$('#annotations').hidden=false;
  const r=result;
  $('#output').innerHTML=`<div class="verdict"><div><p>AI 可能性</p><h3>${escape(r.verdict_text)}</h3><p>${r.chars} 个汉字${r.pieces.length>1?' · '+r.pieces.length+' 段':''}</p></div><div class="prob">${r.ai_prob===null?'—':pct(r.ai_prob)+'<small>%</small>'}</div></div>`+
    (r.ai_prob===null?'<p class="notes">仍可在下方查看十条文风规则的命中情况。</p>':`<div class="meter" role="img" aria-label="AI 可能性 ${pct(r.ai_prob)}%"><i class="needle" style="left:${pct(r.ai_prob)}%"></i></div><div class="meter-labels"><span>更像人写的</span><span>不确定</span><span>很像 AI</span></div><div class="mini-section"><h3>更像哪家 AI？</h3><p>${escape(r.which_text)}</p>${r.which.map(row=>`<div class="model-row"><span>${escape(row.name)}</span><div class="track"><i style="width:${row.p*100}%"></i></div><span>${(row.p*100).toFixed(1)}%</span></div>`).join('')}</div>`)+
    `<div class="mini-section"><h3>AI 八股：${r.baigu_index} / 10 条</h3><p>命中条数只反映文风，长文更容易命中。仅适合比较长度相近的文字。</p></div>`+
    (r.ai_prob===null?'':`<div class="mini-section"><h3>模型用词依据</h3><p>偏向 AI 的词</p><div class="evidence">${evidence(r.evidence.ai)}</div><p>偏向人的词</p><div class="evidence">${evidence(r.evidence.human)}</div><p>这些词只是模型的统计依据，不代表用词有问题。</p></div>`)+`<p class="notes">${r.notes.map(escape).join('<br>')}</p>`;
  $('#rules').replaceChildren(...r.markers.map((row,i)=>{const b=document.createElement('button');b.className='rule'+(row.count?' hit':'');b.setAttribute('aria-pressed','false');b.title=row.hint;b.innerHTML=`<span>${escape(row.name)}</span><b>${row.count}</b>`;b.onclick=()=>{activeRule=activeRule===i?-1:i;activePiece=-1;highlight();};return b;}));
  $('#pieces').replaceChildren(...(r.pieces.length>1?r.pieces.map((p,i)=>{const b=document.createElement('button');b.className='piece';b.setAttribute('aria-pressed','false');b.textContent=`第 ${i+1} 段 · ${pct(p.ai_prob)}%`;b.onclick=()=>{activePiece=activePiece===i?-1:i;activeRule=-1;highlight();$('#highlight mark')?.scrollIntoView({block:'nearest'});};return b;}):[]));
  highlight();if(input.value!==measuredText)markStale();
}
function evidence(rows){return rows.length?rows.map(([g])=>`<span class="word">${escape(g)}</span>`).join(''):'<span class="notes">没有足够的用词依据</span>';}
function highlight(){
  [...$('#rules').children].forEach((b,i)=>b.setAttribute('aria-pressed',String(activeRule===i)));
  [...$('#pieces').children].forEach((b,i)=>b.setAttribute('aria-pressed',String(activePiece===i)));
  const rows=activeRule>=0?[result.markers[activeRule]]:result.markers;
  let spans=rows.flatMap(r=>r.spans.map(([a,b])=>({a,b,name:r.name})));
  if(activePiece>=0){const p=result.pieces[activePiece];spans=[{a:p.start,b:p.end,name:`第 ${activePiece+1} 段`}];}
  const events=new Map();
  for(const s of spans)for(const [point,delta] of [[s.a,1],[s.b,-1]]){if(!events.has(point))events.set(point,[]);events.get(point).push([s.name,delta]);}
  const points=[...new Set([0,measuredText.length,...events.keys()])].sort((a,b)=>a-b),active=new Map();
  let html='';for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1];for(const [name,delta] of events.get(a)||[]){const n=(active.get(name)||0)+delta;if(n)active.set(name,n);else active.delete(name);}const names=[...active.keys()],t=escape(measuredText.slice(a,b));html+=names.length?`<mark${activePiece>=0?' class="piece-mark"':''} title="${escape(names.join('、'))}">${t}</mark>`:t;}
  $('#highlight').innerHTML=html;
  $('#highlight-help').textContent=activePiece>=0?`选中第 ${activePiece+1} 段。分段概率仅供参考。`:activeRule>=0?`${rows[0].hint}；共命中 ${rows[0].count} 处。`:'黄色标出了所有命中位置。同一句可能命中多条规则。';
}
function markStale(){if(result&&input.value!==measuredText){$('#output').classList.add('stale');$('#annotations').classList.add('stale');$('#copy').hidden=true;status.textContent='文字已修改，请点「量一量」更新结果。';}else if(result){$('#output').classList.remove('stale');$('#annotations').classList.remove('stale');$('#copy').hidden=false;}}
input.addEventListener('input',()=>{$('#counter').textContent=cjkCount(input.value)+' 个汉字';markStale();});
button.onclick=measure;input.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();measure();}});
$('#example').onclick=()=>{input.value='下班的时候，天还没完全黑。我沿着熟悉的小路往家走，路旁的树叶在晚风里轻轻摇晃。\n\n这一刻，我忽然觉得，生活的意义不是追逐遥远的星辰，而是珍惜眼前的温柔。我们在忙碌中寻找答案，我们在孤独中学会成长，我们在平凡中发现美好。原来，日子不仅是日历上的数字，更是灵魂深处一次次微小的共振。\n\n总之，愿我们都能带着勇气前行，在岁月的缝隙中，拥抱属于自己的幸福。';input.dispatchEvent(new Event('input'));input.focus();};
$('#clear').onclick=()=>{if(busy){worker?.terminate();worker=null;ready=false;busy=false;requestId++;}input.value='';result=null;measuredText='';$('#counter').textContent='0 个汉字';$('#output').hidden=true;$('#annotations').hidden=true;$('#copy').hidden=true;$('#empty').hidden=false;status.textContent='已清空。文字不会保存在本页。';button.disabled=false;button.firstChild.textContent='量一量 ';input.focus();};
$('#copy').onclick=async()=>{
  const r=result, report=`AI 腔尺子\n${r.verdict_text}\nAI 可能性：${r.ai_prob===null?'无法判断':pct(r.ai_prob)+'%'}\nAI 八股：${r.baigu_index}/10\n${r.which.map(x=>x.name+' '+(x.p*100).toFixed(1)+'%').join(' / ')}\n${r.markers.map(x=>x.name+'：'+x.count+' 处').join('\n')}\n${r.notes.join('\n')}\n原作者：Claude大小姐在运营\nhttps://space.bilibili.com/3747572619414174\n原视频：https://www.bilibili.com/video/BV1xNsV6CE7d/`;
  try{await navigator.clipboard.writeText(report);status.textContent='测量结果已复制。';}catch{const area=document.createElement('textarea');area.value=report;area.readOnly=true;area.style.minHeight='180px';area.setAttribute('aria-label','测量报告，可手动全选复制');$('#output').append(area);area.focus();area.select();status.textContent='浏览器不允许自动复制，报告已选中，请手动复制。';}
};
